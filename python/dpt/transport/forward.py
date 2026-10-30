"""Prepared CUDA transport, sparse history scoring and mandatory completion checks.

Device arrays are caller-owned. Model buffers remain immutable for the workspace
lifetime; dynamic density/source arrays remain unchanged until the owning stream
finishes. Preparation uploads only the small declared energy grid. Repeated calls
allocate no device arrays. Explicitly unchecked calls require ``check_status``
before accepting outputs; incomplete histories never become absorbed histories.
"""

from __future__ import annotations

# Workspace internals are shared only within this package.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dpt._runtime import (
    DeviceContext,
    StatusReadback,
    load_kernels,
    prepare_context,
    require_no_tape,
)
from dpt.contracts import NumericalError

from ._constants import BUDGET_MASK, ENERGY_MASK, FAILURE_MASK, NUMERICAL_MASK
from .model import (
    IncompleteHistoryError,
    TransportError,
    TransportExecution,
    TransportPhysics,
    TransportSpec,
    _physical,
)
from .physical import PhysicalTransport, _prepare_physical_parameters
from .rng import HistoryBatch

if TYPE_CHECKING:
    import warp as wp


@dataclass(slots=True)
class TransportWorkspace:
    """One stream, immutable material tables, small diagnostics and no history tape."""

    spec: TransportSpec
    max_histories: int
    context: DeviceContext
    material_ids: Any
    absorption: Any
    scattering: Any
    voxel_density: Any
    _energies: Any = field(repr=False)
    _parameters: Any = field(repr=False)
    _status: Any = field(repr=False)
    _kernels: Any = field(repr=False)
    _status_readback: StatusReadback = field(repr=False)
    _sampling: Any = field(default=None, init=False, repr=False)
    _physical_buffers: tuple[tuple[str, Any], ...] = field(default=(), repr=False)
    physical: PhysicalTransport | None = field(default=None)

    @property
    def execution(self) -> TransportExecution:
        """Effective finite-resource policy retained at preparation."""
        return self.spec.execution

    @property
    def scratch_bytes(self) -> int:
        return (
            4
            + 8 * len(self.spec.energy_nodes)
            + (0 if self._sampling is None else self._sampling.allocated_bytes)
        )

    def clear_status(self) -> None:
        """Reset at a declared independent run boundary, after discarding failures."""
        with self.context.scope():
            self._status.zero_()

    def check_status(self) -> None:
        """Synchronise this stream and transfer only the four-byte status word."""
        self._raise_status(self._status_readback.read())

    def consume_status(self) -> None:
        """Deliver then clear a checkpoint; failed estimate outputs remain invalid."""
        self._raise_status(self._status_readback.read(reset=True))

    @property
    def host_bytes(self) -> int:
        return self._status_readback.host_bytes

    def _raise_status(self, code: int) -> None:
        if code & BUDGET_MASK:
            raise IncompleteHistoryError(
                "event, voxel-crossing or angular-rejection budget exhausted; discard the estimate"
            )
        if code & ENERGY_MASK:
            raise TransportError("a history left the supplied coefficient energy support")
        if code & ~FAILURE_MASK:
            raise TransportError("invalid device transport inputs")
        if code & NUMERICAL_MASK:
            raise NumericalError("nonfinite transport arithmetic or an invalid moment")

    def _launch(self, kernel: Any, dim: int, inputs: list[Any]) -> None:
        self.context.wp.launch(
            kernel,
            dim=dim,
            inputs=inputs,
            device=self.context.device,
            stream=self.context.stream,
            block_dim=self.spec.block_dim,
            record_tape=False,
        )

    def _model_inputs(self) -> list[Any]:
        return [
            self.material_ids,
            self.voxel_density,
            self._energies,
            self.absorption,
            self.scattering,
            self._parameters,
        ]

    def _reads(self) -> list[tuple[str, Any]]:
        return [
            ("material_ids", self.material_ids),
            ("absorption", self.absorption),
            ("scattering", self.scattering),
            ("voxel_density", self.voxel_density),
            ("energy_nodes", self._energies),
            ("workspace_status", self._status),
            *self._physical_buffers,
            *(
                []
                if self._sampling is None
                else [
                    ("sampling_words", self._sampling._words),
                    ("sampling_shift", self._sampling._shift),
                ]
            ),
        ]


def prepare_transport(
    spec: TransportSpec | TransportPhysics,
    *,
    material_ids: wp.array[wp.int32],
    absorption: wp.array[wp.float64],
    scattering: wp.array[wp.float64],
    max_histories: int,
    device: str | wp.Device = "cuda:0",
    stream: wp.Stream | None = None,
    validate: bool = True,
    voxel_density: wp.array[wp.float64] | None = None,
    execution: TransportExecution | None = None,
    physical: PhysicalTransport | None = None,
) -> TransportWorkspace:
    """Bind CPU/CUDA buffers and validate supplied immutable coefficients.

    Material IDs are int32, flattened x fastest. Coefficients are binary64,
    material-major with energy fastest. Their partial sum is total extinction;
    no inconsistent independent total coefficient is accepted. The tables are
    already linear coefficients in inverse mm. ``scattering_law`` selects
    isotropic elastic, stationary free-electron Compton or bound-electron
    Compton scattering. The bound model requires explicit shell records in
    ``physical`` and a local deposition cutoff within coefficient support.
    """
    spec = TransportSpec.resolve(spec, execution)
    if type(max_histories) is not int or not 0 < max_histories < 2**31:
        raise TransportError("max_histories must be a positive signed 32-bit integer")
    context = prepare_context(device=device, stream=stream)
    require_no_tape()
    wp = context.wp
    context.array(material_ids, "material_ids", dtype=wp.int32, shape=(math.prod(spec.grid.shape),))
    table_size = spec.grid.materials * len(spec.energy_nodes)
    if table_size >= 2**31:
        raise TransportError("material-energy table exceeds signed 32-bit indexing")
    for name, value in (("absorption", absorption), ("scattering", scattering)):
        context.array(value, name, dtype=wp.float64, shape=(table_size,))
    if voxel_density is not None:
        context.array(
            voxel_density, "voxel_density", dtype=wp.float64, shape=(math.prod(spec.grid.shape),)
        )
    kernels = load_kernels("dpt.transport.kernels")
    with context.scope():
        energies = wp.array(list(spec.energy_nodes), dtype=wp.float64, device=context.device)
        status = wp.zeros(1, dtype=wp.int32, device=context.device)
        if voxel_density is None:
            voxel_density = wp.empty(0, dtype=wp.float64, device=context.device)
    parameters = kernels.Parameters()
    parameters.origin = wp.vec3d(*spec.grid.origin_mm)
    parameters.spacing = wp.vec3d(*spec.grid.spacing_mm)
    parameters.shape = wp.vec3i(*spec.grid.shape_xyz)
    parameters.detector_spacing = wp.vec2d(*spec.detector.spacing_xy_mm)
    parameters.detector_shape = wp.vec2i(*spec.detector.shape)
    inverse_pose = spec.object_pose.inverse()
    parameters.detector_general = 1
    parameters.detector_corner = wp.vec3d(*inverse_pose.point(spec.detector.lower_corner_mm))
    parameters.detector_u = wp.vec3d(*inverse_pose.direction(spec.detector.u))
    parameters.detector_v = wp.vec3d(*inverse_pose.direction(spec.detector.v))
    parameters.detector_normal = wp.vec3d(*inverse_pose.direction(spec.detector.normal))
    parameters.inverse_rotation = wp.mat33d(*inverse_pose.rotation)
    parameters.inverse_translation = wp.vec3d(*inverse_pose.translation_mm)
    parameters.source_energy = spec.energy_kev
    parameters.energy_bins = len(spec.energy_nodes)
    parameters.max_events = spec.max_events
    parameters.max_crossings = spec.max_crossings
    parameters.max_angle_trials = spec.max_angle_trials
    parameters.compton = (
        2
        if spec.scattering_law == "bound-electron-compton"
        else int(spec.scattering_law == "free-electron-compton")
    )
    parameters.compton_packed = int(spec.compton_sampler == "packed-uniform-rejection")
    parameters.energy_score = int(spec.scoring == "energy-kev")
    parameters.continuous_absorption = int(spec.estimator == "continuous-absorption")
    assert voxel_density is not None
    parameters.voxel_density = int(voxel_density.size != 0)
    parameters.physics, physical_buffers = _prepare_physical_parameters(physical, spec, context)
    if parameters.physics.woodcock != 0:
        with context.scope():
            parameters.majorant_max_density = wp.zeros(
                spec.grid.materials, dtype=wp.float64, device=context.device
            )
        physical_buffers += (("majorant_max_density", parameters.majorant_max_density),)
    workspace = TransportWorkspace(
        spec,
        max_histories,
        context,
        material_ids,
        absorption,
        scattering,
        voxel_density,
        energies,
        parameters,
        status,
        kernels,
        StatusReadback(context, status),
        physical_buffers,
        physical,
    )
    if validate:
        workspace._launch(
            kernels.validate_model,
            max(int(material_ids.size), table_size),
            [material_ids, voxel_density, absorption, scattering, spec.grid.materials, status],
        )
        workspace.check_status()
        if physical is not None:
            workspace._launch(
                kernels.validate_physical_coefficients,
                table_size,
                [scattering, parameters.physics, status],
            )
            workspace.check_status()
    if parameters.physics.woodcock != 0:
        workspace._launch(
            kernels.prepare_majorant_density_bounds,
            int(material_ids.size),
            [material_ids, voxel_density, parameters.majorant_max_density],
        )
    return workspace


def _validate_call(
    workspace: TransportWorkspace,
    batch: HistoryBatch,
    positions: Any,
    directions: Any,
    weights: Any,
    density: Any,
    writes: list[tuple[str, Any, Any]],
    source_amplitude: float,
    stream: Any,
    validate: bool,
) -> None:
    require_no_tape()
    context = workspace.context
    context.assert_stream(stream)
    wp = context.wp
    from .sampling import _bind_sampling

    _bind_sampling(workspace, batch)
    if batch.count > workspace.max_histories:
        raise TransportError("history batch exceeds workspace capacity")
    _physical(source_amplitude, "source amplitude", nonnegative=True)
    source_arrays = [("positions", positions), ("directions", directions), ("weights", weights)]
    for name, value in source_arrays:
        context.array(
            value,
            name,
            dtype=wp.float64 if name == "weights" else wp.vec3d,
            shape=(batch.count,),
        )
    context.array(density, "density", dtype=wp.float64, shape=(workspace.spec.grid.materials,))
    for name, value, dtype in writes:
        context.array(value, name, dtype=dtype, shape=(batch.count,))
    context.disjoint(
        workspace._reads() + source_arrays + [("density", density)],
        [(name, value) for name, value, _ in writes],
    )
    # Unchecked calls retain prior failure status. A successful later launch
    # cannot launder an incomplete earlier estimate into an accepted one.
    if validate:
        workspace.check_status()
        if workspace.physical is not None and workspace.physical.majorant_mm_inverse:
            workspace._launch(
                workspace._kernels.validate_majorant,
                int(workspace.material_ids.size),
                [density, *workspace._model_inputs(), workspace._status],
            )
            workspace.check_status()
        workspace._launch(
            workspace._kernels.validate_sources,
            max(batch.count, workspace.spec.grid.materials),
            [
                positions,
                directions,
                weights,
                density,
                workspace._parameters,
                workspace._status,
            ],
        )
        workspace.check_status()


def trace_histories(
    positions: wp.array[wp.vec3d],
    directions: wp.array[wp.vec3d],
    weights: wp.array[wp.float64],
    density: wp.array[wp.float64],
    *,
    batch: HistoryBatch,
    workspace: TransportWorkspace,
    out_pixel: wp.array[wp.int32],
    out_score: wp.array[wp.float64],
    out_energy: wp.array[wp.float64],
    out_events: wp.array[wp.int32],
    out_status: wp.array[wp.int32],
    source_amplitude: float = 1.0,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    r"""``X_i = weight_i * amplitude * r(E_final_i) * 1[detector hit]``.

    Pixel ``p`` has mean ``E[weight*source_amplitude*r(E_final)*1[hit=p]]``,
    where ``r(E)=1`` for arrival scoring and ``r(E)=E`` for energy-keV scoring.
    Average sparse scores over every launched original history, including zeros
    for absorption and detector misses. The analogue walk samples flights from
    total extinction. Continuous absorption samples scattering flights and
    multiplies the final contribution by ``exp(-integral(mu_abs ds))``. All
    segments use coefficients at their current photon energy.

    With ``N=batch.count`` and ``M=workspace.spec.grid.materials``, positions and
    directions are contiguous ``wp.vec3d`` arrays of shape ``(N,)``. Positions
    are world mm and direction squared norms differ from one by at most 1e-12.
    ``weights`` is nonnegative ``wp.float64`` of shape ``(N,)``, and ``density``
    is strictly positive ``wp.float64`` of shape ``(M,)`` containing dimensionless
    material multipliers. Optional prepared voxel multipliers are dimensionless.
    They multiply both partial coefficients in mm inverse. Source positions
    must lie on the source side of the detector and batch capacity is checked.

    All five outputs have shape ``(N,)`` on the workspace device and are fully
    overwritten. ``out_pixel``, ``out_events`` and ``out_status`` are ``wp.int32``.
    A pixel of ``-1`` means no detector arrival. ``out_score`` and ``out_energy``
    are ``wp.float64`` with the declared measurement units and keV respectively.
    Continuous-absorption events count scatterings, while analogue events count
    interactions. With Woodcock sampling, the event count includes null candidates.
    An incomplete event, crossing, angular or energy-support history invalidates
    the batch rather than being converted into a physical absorption event.

    Inputs, outputs and prepared buffers must satisfy the disjoint-array contract.
    Arrays remain alive and unchanged until the prepared stream completes, and
    an explicitly supplied ``stream`` must match it. ``validate=True`` performs
    mandatory status completion, while an unchecked call requires the caller to
    invoke ``workspace.check_status()`` before accepting outputs. This function
    has no Warp tape derivative. Dedicated replay APIs differentiate the expected
    measurement for their declared active density, coefficient, spectrum and
    source-amplitude parameters while retaining the original-history identity.

    """
    wp = workspace.context.wp
    _validate_call(
        workspace,
        batch,
        positions,
        directions,
        weights,
        density,
        [
            ("out_pixel", out_pixel, wp.int32),
            ("out_score", out_score, wp.float64),
            ("out_energy", out_energy, wp.float64),
            ("out_events", out_events, wp.int32),
            ("out_status", out_status, wp.int32),
        ],
        source_amplitude,
        stream,
        validate,
    )
    workspace._launch(
        workspace._kernels.trace_histories,
        batch.count,
        [
            positions,
            directions,
            weights,
            density,
            *workspace._model_inputs(),
            wp.uint64(batch.seed),
            wp.uint64(batch.first_history),
            source_amplitude,
            out_pixel,
            out_score,
            out_energy,
            out_events,
            out_status,
            workspace._status,
        ],
    )
    if validate:
        workspace.check_status()
