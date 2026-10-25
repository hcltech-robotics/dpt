"""Optical depth L[p] = integral mu(r_p(s)) ds and its discrete VJP.

The field normally uses FP32 storage. Local quadrature arithmetic is FP32 or
FP64 according to sampling_precision; initial ray setup remains FP64. Depth
storage is selected separately. Prepared inputs own validated field and pose
snapshots for repeated evaluations on one device and stream.
"""

from __future__ import annotations

# Mathematical output names follow the optical-depth contract.
# ruff: noqa: N803
# Module-owned workspace scratch stays private to this implementation.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from dpt._reductions import ReductionTree, prepare_reduction
from dpt._runtime import (
    DeviceContext,
    HostReadback,
    StatusReadback,
    ensure_tape,
    load_kernels,
    prepare_context,
    require_no_tape,
)
from dpt.contracts import ContractError, integer, raise_device_status
from dpt.geometry import DetectorGeometry, RayBundleGeometry
from dpt.geometry_device import DeviceRayBundleGeometry
from dpt.volumes import GridSpec

_BLOCK = 128
_FORWARD_BLOCK = 256
ValidationPolicy = Literal["once", "always", "never"]
ProjectionGeometry = DetectorGeometry | RayBundleGeometry | DeviceRayBundleGeometry

if TYPE_CHECKING:
    import warp as wp

    FieldArray = wp.array[wp.float32] | wp.array[wp.float16] | wp.array[wp.bfloat16]
    DepthArray = wp.array[wp.float32] | wp.array[wp.float64]


@dataclass(frozen=True, slots=True)
class ProjectionGradients:
    """Caller-owned volume and pose destinations for one projection VJP.

    ``pose_tangent`` is binary64 (6,) in local-right tx,ty,tz,rx,ry,rz order.
    ``pose_storage`` is binary64 (12,) in packed rigid-transform order. Select
    at most one pose representation. ``volume`` follows the VJP's declared
    ``volume_gradient_precision`` and remains on the workspace device.
    """

    volume: DepthArray | None = None
    pose_tangent: wp.array[wp.float64] | None = None
    pose_storage: wp.array[wp.float64] | None = None

    def __post_init__(self) -> None:
        if self.pose_tangent is not None and self.pose_storage is not None:
            raise ContractError("request either a right-tangent or a storage pose gradient")
        if self.volume is None and self.pose_tangent is None and self.pose_storage is None:
            raise ContractError("request at least one projection gradient")


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    """Choose the discrete quadrature L[p] and its active input derivatives.

    Midpoint uses samples_per_ray. Cell Gauss ignores this count and splits at
    every interpolation plane, integrating each cubic segment with two nodes.

    sampling_precision defaults to float64. float32 uses FP32 local ray
    coordinates, interpolation and accumulation with midpoint quadrature and
    FP32 field storage. Initial ray setup and the final pose chain use FP64
    outside the sample loop. Its volume and pose VJPs use the same realised
    sample positions and weights as forward.

    mixed_float32 is an explicit numerical comparison with a
    fixed-pose midpoint approximation: ray setup/intersection, field products
    and sums remain FP64, while local marching and corner weights use FP32.
    Its volume VJP uses exactly the same rounded weights. Pose derivatives are
    unsupported in this policy; output precision remains a separate choice.

    fast_float32 also evaluates each interpolation cell using FP32 products
    and fused partial sums, then promotes that sample to the FP64 ray sum.
    Ray setup and clipping remain FP64. Its VJP uses the rounded interpolation
    weights; FP32 cell rounding introduces a measurable adjoint error.
    storage_precision selects native FP32, FP16 or BF16 device fields, with
    conversion in sampling registers. Volume cotangents use FP32 or FP64 and
    differentiate the decoded field values.

    Grid/acquisition calibration is fixed. Pose derivatives include active
    intersection bounds. At ties, grazing rays and interpolation knots the
    returned branch derivative does not assert differentiability.

    ``validate='once'`` checks owned snapshots at preparation and raw buffers
    at every evaluation. ``'always'`` rescans all inputs on every call;
    ``'never'`` uses explicit caller-managed content/status checkpoints.

    ``backend='cuda_texture'`` selects CUDA C++ texture reads and a
    voxel-footprint volume adjoint for FP32 midpoint sampling. Pose derivatives,
    other sampling policies and CPU execution use Warp. Captured forwards also
    use Warp. CUDA compilation requires nvcc for the selected GPU.
    """

    samples_per_ray: int
    active_pose: bool = True
    active_volume: bool = False
    precision: Literal["float32", "float64"] = "float32"
    integration: Literal["midpoint", "cell_gauss"] = "midpoint"
    field_domain: Literal["physical", "signed_linear"] = "physical"
    sampling_precision: Literal["float32", "float64", "mixed_float32", "fast_float32"] = "float64"
    storage_precision: Literal["float32", "float16", "bfloat16"] = "float32"
    validate: ValidationPolicy = "once"
    backend: Literal["warp", "cuda_texture"] = "warp"

    def __post_init__(self) -> None:
        integer(self.samples_per_ray, "samples_per_ray", minimum=1)
        if self.integration not in ("midpoint", "cell_gauss"):
            raise ContractError("projection integration must be midpoint or cell_gauss")
        if self.precision not in ("float32", "float64"):
            raise ContractError("projection precision must be float32 or float64")
        if self.field_domain not in ("physical", "signed_linear"):
            raise ContractError("projection field_domain must be physical or signed_linear")
        if self.storage_precision not in ("float32", "float16", "bfloat16"):
            raise ContractError("storage_precision must be float32, float16 or bfloat16")
        if self.sampling_precision not in ("float32", "float64", "mixed_float32", "fast_float32"):
            raise ContractError(
                "sampling_precision must be float32, float64, mixed_float32 or fast_float32"
            )
        if self.sampling_precision in ("mixed_float32", "fast_float32") and (
            self.integration != "midpoint" or self.active_pose
        ):
            raise ContractError("FP32 sampling requires midpoint integration and fixed pose")
        if self.sampling_precision == "float32" and (
            self.integration != "midpoint" or self.storage_precision != "float32"
        ):
            raise ContractError(
                "FP32 arithmetic requires midpoint integration and FP32 field storage"
            )
        if type(self.active_pose) is not bool or type(self.active_volume) is not bool:
            raise ContractError("active input flags must be booleans")
        if self.validate not in ("once", "always", "never"):
            raise ContractError("projection validate must be once, always or never")
        if self.backend not in ("warp", "cuda_texture"):
            raise ContractError("projection backend must be warp or cuda_texture")


@dataclass(frozen=True, slots=True, init=False)
class PreparedProjectionInputs:
    """Owned mu[voxels] and pose[12] snapshots, validated at preparation.

    Create this record with ``workspace.prepare_inputs(mu, pose)``. Changes to
    the original buffers leave the snapshot intact. Prepare another record to
    use a changed field or pose; the device arrays stay private to the operator.

    Example::

        inputs = workspace.prepare_inputs(mu, pose)
        project_optical_depth(inputs, workspace=workspace, out_L=depth)
    """

    _workspace: ProjectionWorkspace = field(repr=False)
    _metadata: tuple[Any, ...] = field(repr=False)
    _mu: Any = field(repr=False)
    _pose: Any = field(repr=False)

    def __init__(self) -> None:
        raise ContractError("create prepared inputs with workspace.prepare_inputs(mu, pose)")

    @property
    def device_bytes(self) -> int:
        """Bytes retained by the field and rigid-pose snapshots."""
        return (
            int(self._mu.size) * self._workspace.context.wp.types.type_size_in_bytes(self._mu.dtype)
            + 12 * 8
        )


@dataclass(frozen=True, slots=True, init=False)
class PreparedProjectionAdjoint:
    """Owned fixed-pose ray setup for exact midpoint voxel gathering.

    ``prepare_projection_adjoint`` captures the pose and geometry and checks
    every canonical ray and midpoint. Flat cone panels additionally retain
    conservative detector bounds. Other ray bundles use all detector rays
    with conservative sample-index clipping. Device arrays are private.

    ``candidate_rays`` and ``quadrature_nodes`` count the prepared worst case
    before zero cotangents are skipped. ``work_bound`` includes one support
    query per voxel. Rebuild the plan for a different pose or geometry.
    """

    _workspace: ProjectionWorkspace = field(repr=False)
    _metadata: tuple[Any, ...] = field(repr=False)
    _pose: Any = field(repr=False)
    _rays: Any = field(repr=False)
    _coefficients: Any = field(repr=False)
    _metrics: Any = field(repr=False)
    _configuration: Any = field(repr=False)
    _cone: bool = field(repr=False)
    _voxel_count: int = field(repr=False)
    candidate_rays: int
    quadrature_nodes: int
    fallback_voxels: int
    preparation_work_bound: int
    device_bytes: int

    def __init__(self) -> None:
        raise ContractError("create a fixed-pose plan with prepare_projection_adjoint")

    @property
    def work_bound(self) -> int:
        """Upper bound on voxel, candidate-ray and sample queries per gather."""
        return self._voxel_count + self.candidate_rays + self.quadrature_nodes


@dataclass(slots=True)
class ProjectionWorkspace:
    """Persistent O(12 P/128) FP64 reduction storage, one owning CUDA stream.

    Original volume and pose buffers must remain immutable through backward.
    One recorded forward may be outstanding per workspace. Use another
    workspace for concurrent/re-entrant evaluations. No sample history is kept.
    """

    grid: GridSpec
    geometry: ProjectionGeometry
    spec: ProjectionSpec
    context: DeviceContext
    _kernels: Any = field(repr=False)
    _geometry_kernels: Any = field(repr=False)
    _configuration: Any = field(repr=False)
    _status: Any = field(repr=False)
    _empty: Any = field(repr=False)
    _reduction: ReductionTree | None = field(repr=False)
    _empty_partials: Any = field(repr=False)
    _recorded_tape: Any = field(default=None, repr=False)
    _native: Any = field(default=None, repr=False)
    _status_readback: StatusReadback = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._status_readback = StatusReadback(self.context, self._status)

    @property
    def dtype(self) -> type[wp.float32] | type[wp.float64]:
        """Declared depth and depth-cotangent storage, separate from field storage."""
        return getattr(self.context.wp, self.spec.precision)

    @property
    def field_dtype(self) -> type[wp.float32] | type[wp.float16] | type[wp.bfloat16]:
        """Native volume storage, converted in registers during sampling."""
        return getattr(self.context.wp, self.spec.storage_precision)

    @property
    def device(self) -> wp.Device:
        return self.context.device

    @property
    def stream(self) -> wp.Stream | None:
        return self.context.stream

    @property
    def scratch_bytes(self) -> int:
        return (
            4
            + (48 * self.geometry.pixels if isinstance(self.geometry, RayBundleGeometry) else 0)
            + (self._reduction.scratch_bytes if self._reduction is not None else 0)
            + (self._native.scratch_bytes if self._native is not None else 0)
        )

    @property
    def host_bytes(self) -> int:
        return self._status_readback.host_bytes

    def clear_status(self) -> None:
        with self.context.scope():
            self._status.zero_()

    def prepare_inputs(
        self, mu: Any, pose: Any, *, stream: wp.Stream | None = None
    ) -> PreparedProjectionInputs:
        """Copy mu[voxels] and pose[12] into an owned, validated snapshot.

        Preparation scans the copied field and checks the rigid pose once.
        Copies and checks run on the owning stream. Snapshot gradients are
        available through ``projection_vjp`` with caller-owned destinations.
        """
        require_no_tape()
        if self._recorded_tape is not None:
            raise ContractError("finish or discard the outstanding projection tape before reuse")
        _inputs(mu, pose, self, stream)
        ctx = self.context
        with ctx.scope():
            owned_mu = ctx.wp.empty(mu.shape, dtype=self.field_dtype, device=self.device)
            owned_pose = ctx.wp.empty(12, dtype=ctx.wp.float64, device=self.device)
            ctx.wp.copy(owned_mu, mu, stream=self.stream)
            ctx.wp.copy(owned_pose, pose, stream=self.stream)
        _validate(owned_mu, owned_pose, self)
        prepared = object.__new__(PreparedProjectionInputs)
        object.__setattr__(prepared, "_workspace", self)
        object.__setattr__(prepared, "_metadata", _metadata(self))
        object.__setattr__(prepared, "_mu", owned_mu)
        object.__setattr__(prepared, "_pose", owned_pose)
        return prepared

    def warm(
        self,
        *,
        forward: bool = True,
        vjp: bool | None = None,
        matrix_gradient: bool = False,
        volume_gradient_precision: Literal["float32", "float64"] = "float32",
    ) -> tuple[str, ...]:
        """Compile the selected forward/VJP variants without tracing any rays.

        The default prepares the forward and all declared active gradients.
        ``matrix_gradient=True`` selects twelve pose-storage partials instead
        of the six-coordinate right tangent. Compilation allocates no field,
        image or gradient arrays and retains existing diagnostics.
        """
        require_no_tape()
        if vjp is None:
            vjp = self.spec.active_pose or self.spec.active_volume
        if any(type(value) is not bool for value in (forward, vjp, matrix_gradient)):
            raise ContractError("warm forward, vjp and matrix_gradient must be booleans")
        if volume_gradient_precision not in ("float32", "float64"):
            raise ContractError("volume_gradient_precision must be float32 or float64")
        if vjp and not (self.spec.active_pose or self.spec.active_volume):
            raise ContractError("VJP warming requires an active input")
        if matrix_gradient and not self.spec.active_pose:
            raise ContractError("matrix gradients require active pose")
        kernels: list[tuple[Any, int]] = [
            (_field_check(self), 256),
            (self._geometry_kernels.validate_pose, 256),
        ]
        if forward:
            kernels.append(
                (
                    _forward_kernel(self),
                    _FORWARD_BLOCK,
                )
            )
        if vjp:
            kernels.append(
                (
                    _vjp_kernel(
                        self,
                        self.spec.active_volume,
                        matrix_gradient,
                        self.spec.active_pose,
                        double_seed=self.spec.precision == "float64",
                        cell_gauss=self.spec.integration == "cell_gauss",
                        double_volume_gradient=volume_gradient_precision == "float64"
                        and self.spec.active_volume,
                        mixed=self.spec.sampling_precision != "float64",
                        storage_precision=self.spec.storage_precision,
                    ),
                    _vjp_block(self, active_pose=self.spec.active_pose),
                )
            )
            kernels.append(
                (self._kernels.get_validate_finite(self.spec.precision == "float64"), 256)
            )
            if self.spec.active_volume:
                kernels.append(
                    (self._kernels.get_validate_finite(volume_gradient_precision == "float64"), 256)
                )
            if self.spec.active_pose:
                assert self._reduction is not None
                kernels.append(
                    (
                        self._reduction._kernels.reduce_kernel(
                            _BLOCK, 12 if matrix_gradient else 6
                        ),
                        _BLOCK,
                    )
                )
                kernels.extend(
                    (self._reduction._kernels.finish_kernel(False, accumulate), 256)
                    for accumulate in (False, True)
                )
        modules: dict[tuple[str, int], Any] = {
            (kernel.module.name, block): kernel.module for kernel, block in kernels
        }
        with self.context.scope():
            self.check_status()
            for (_, block), module in modules.items():
                self.context.wp.load_module(module, device=self.device, block_dim=block)
        return tuple(f"{name}@block{block}" for name, block in modules)

    def validate_inputs(
        self,
        mu: FieldArray,
        pose: wp.array[wp.float64],
        *,
        cotangent: DepthArray | None = None,
        stream: wp.Stream | None = None,
    ) -> None:
        """Check current field, pose and optional cotangent before an evaluation.

        Check mutable buffers after their last change and keep them intact
        through an unchecked forward/VJP epoch. Check output diagnostics at
        the next acceptance checkpoint with ``check_status()``.

        ``cotangent`` optionally checks a depth covector of shape (pixels,) in
        the declared depth precision. Omission or None skips this optional covector check.
        """
        require_no_tape()
        _inputs(mu, pose, self, stream)
        if cotangent is not None:
            self.context.array(
                cotangent, "cotangent", dtype=self.dtype, shape=(self.geometry.pixels,)
            )
        _validate(mu, pose, self, cotangent)

    def validate_pose(self, pose: wp.array[wp.float64], *, stream: wp.Stream | None = None) -> None:
        """Check pose[12] for a finite rigid transform on the owning stream.

        This bounded check validates rotation and translation without reading
        the volume. It synchronises diagnostics before and after the check.
        """
        require_no_tape()
        ctx = self.context
        ctx.assert_stream(stream)
        ctx.array(pose, "pose", dtype=ctx.wp.float64, shape=(12,))
        self.check_status()
        ctx.wp.launch(
            self._geometry_kernels.validate_pose,
            dim=1,
            inputs=[pose],
            outputs=[self._status],
            stream=self.stream,
            device=self.device,
            record_tape=False,
        )
        self.check_status()

    def check_status(self) -> None:
        self._raise_status(self._status_readback.read())

    def enqueue_field_check(self, mu: FieldArray) -> None:
        """Check current field content without reading or clearing diagnostics.

        A composed call must consume its diagnostics before accepting outputs.
        The caller keeps the field unchanged throughout that operation.
        """
        require_no_tape()
        self.context.array(mu, "mu", dtype=self.field_dtype, shape=(self.grid.voxels,))
        self.context.wp.launch(
            _field_check(self),
            dim=self.grid.voxels,
            inputs=[mu],
            outputs=[self._status],
            stream=self.stream,
            device=self.device,
            record_tape=False,
        )

    def enqueue_cotangent_check(self, cotangent: DepthArray) -> None:
        """Check finite flat covectors, including concatenated views, asynchronously."""
        require_no_tape()
        self.context.array(cotangent, "cotangent", dtype=self.dtype, ndim=1)
        self.context.wp.launch(
            self._kernels.get_validate_finite(self.spec.precision == "float64"),
            dim=cotangent.shape[0],
            inputs=[cotangent],
            outputs=[self._status],
            stream=self.stream,
            device=self.device,
            record_tape=False,
        )

    def enqueue_volume_gradient_check(
        self,
        out_mu: DepthArray,
        *,
        volume_gradient_precision: Literal["float32", "float64"] = "float32",
        stream: wp.Stream | None = None,
    ) -> None:
        """Enqueue the final finite check for an explicit accumulation epoch.

        Deferred VJPs must keep the destination intact until this check, using
        only accumulation. A finite increment cannot recover an infinite or
        NaN accumulator. This enqueues work without resetting diagnostics or
        reading them; check_status or a composed checkpoint delivers failures.
        """
        require_no_tape()
        if volume_gradient_precision not in ("float32", "float64"):
            raise ContractError("volume_gradient_precision must be float32 or float64")
        ctx = self.context
        ctx.assert_stream(stream)
        ctx.array(
            out_mu,
            "out_mu",
            dtype=getattr(ctx.wp, volume_gradient_precision),
            shape=(self.grid.voxels,),
        )
        ctx.wp.launch(
            self._kernels.get_validate_finite(volume_gradient_precision == "float64"),
            dim=self.grid.voxels,
            inputs=[out_mu],
            outputs=[self._status],
            stream=self.stream,
            device=self.device,
            record_tape=False,
        )

    def consume_status(self) -> None:
        """Report and acknowledge pending diagnostics on the owning stream."""
        self._raise_status(self._status_readback.read(reset=True))

    @staticmethod
    def _raise_status(code: int) -> None:
        raise_device_status(
            code,
            "projection",
            messages={
                1: "invalid rigid pose coordinate map",
                2: "non-finite projection or gradient; discard these outputs",
                8: "field must be finite and non-negative",
            },
        )

    def discard_recording(self, tape: wp.Tape) -> None:
        """Release a forward abandoned by tape.reset(), after discarding its graph."""
        if self._recorded_tape is not None and self._recorded_tape is not tape:
            raise ContractError("this workspace belongs to another tape")
        if cast(Any, tape).launches:
            raise ContractError("reset the abandoned tape before releasing its input lifetime")
        self._recorded_tape = None


def prepare_projection(
    grid: GridSpec,
    geometry: ProjectionGeometry,
    spec: ProjectionSpec,
    *,
    device: str | wp.Device = "cuda:0",
    stream: wp.Stream | None = None,
) -> ProjectionWorkspace:
    """Prepare L[pixels] quadrature metadata and persistent device scratch.

    Explicit bundles copy their immutable world segments once into persistent
    FP64 device storage. Outputs remain flat and one segment has one result.
    Device bundles retain their already prepared arrays, flattened in
    view,row,column order. Supply image and cotangent outputs separately.
    """
    if not isinstance(spec, ProjectionSpec):
        raise ContractError("spec must be a ProjectionSpec")
    if not isinstance(
        cast(object, geometry), (DetectorGeometry, RayBundleGeometry, DeviceRayBundleGeometry)
    ):
        raise ContractError("projection geometry must be a detector or prepared ray bundle")
    if spec.sampling_precision != "float64" and max(grid.shape) > 2**24:
        raise ContractError("FP32 sampling requires grid axis counts at most 2**24")
    context = prepare_context(device=device, stream=stream)
    if isinstance(geometry, DeviceRayBundleGeometry):
        if geometry.device != context.device:
            raise ContractError("prepared geometry and projection must share a device")
        geometry.context.assert_stream(context.stream)
        context.array(
            geometry.sources_mm, "sources_mm", dtype=context.wp.vec3d, shape=(geometry.pixels,)
        )
        context.array(
            geometry.endpoints_mm, "endpoints_mm", dtype=context.wp.vec3d, shape=(geometry.pixels,)
        )
    wp = context.wp
    kernels = load_kernels("dpt.kernels.projection")
    geometry_kernels = load_kernels("dpt.kernels.geometry")
    config = kernels.Configuration()
    config.shape = wp.vec3i(grid.shape[2], grid.shape[1], grid.shape[0])
    # diag(1/h) G^T: spacing follows grid axes, not object/world coordinates.
    inverse_grid = tuple(
        grid.orientation[3 * j + i] / grid.spacing_mm[i] for i in range(3) for j in range(3)
    )
    if not all(math.isfinite(value) for value in inverse_grid):
        raise ContractError("grid spacing overflows the FP64 coordinate map")
    if isinstance(geometry, DetectorGeometry):
        for row in (0, geometry.shape[0] - 1):
            for column in (0, geometry.shape[1] - 1):
                endpoint = geometry.pixel_centre(row, column)
                if not all(math.isfinite(value) for value in endpoint) or not math.isfinite(
                    math.dist(endpoint, geometry.source_mm)
                ):
                    raise ContractError("detector extent exceeds the FP64 finite-ray range")
    config.object_to_grid = wp.mat33d(*inverse_grid)
    config.grid_origin = wp.vec3d(*grid.origin_mm)
    config.is_ray_bundle = isinstance(geometry, (RayBundleGeometry, DeviceRayBundleGeometry))
    if isinstance(geometry, DetectorGeometry):
        config.source = wp.vec3d(*geometry.source_mm)
        config.detector_origin = wp.vec3d(*geometry.origin_mm)
        config.column_step = wp.vec3d(*(x * geometry.spacing_mm[0] for x in geometry.u))
        config.row_step = wp.vec3d(*(x * geometry.spacing_mm[1] for x in geometry.v))
        config.width = geometry.shape[1]
    else:
        config.width = 1
    config.pixels = geometry.pixels
    config.samples = spec.samples_per_ray
    reduction = None
    with context.scope():
        if isinstance(geometry, DeviceRayBundleGeometry):
            config.ray_sources = geometry.sources_mm
            config.ray_endpoints = geometry.endpoints_mm
        elif isinstance(geometry, RayBundleGeometry):
            config.ray_sources = wp.array(
                geometry.sources_mm, dtype=wp.vec3d, device=context.device
            )
            config.ray_endpoints = wp.array(
                geometry.endpoints_mm, dtype=wp.vec3d, device=context.device
            )
        else:
            config.ray_sources = wp.empty(0, dtype=wp.vec3d, device=context.device)
            config.ray_endpoints = wp.empty(0, dtype=wp.vec3d, device=context.device)
        status = wp.zeros(1, dtype=wp.int32, device=context.device)
        empty = wp.empty(0, dtype=wp.float32, device=context.device)
        empty_partials = wp.empty(0, dtype=wp.float64, device=context.device)
        if spec.active_pose:
            reduction = prepare_reduction(context, geometry.pixels, tile=_BLOCK, components=12)
    workspace = ProjectionWorkspace(
        grid,
        geometry,
        spec,
        context,
        kernels,
        geometry_kernels,
        config,
        status,
        empty,
        reduction,
        empty_partials,
    )
    if (
        context.device.is_cuda
        and spec.backend == "cuda_texture"
        and (spec.sampling_precision == "float32")
    ):
        from dpt._cuda_texture import NativeProjection

        workspace._native = NativeProjection(workspace)
    return workspace


def _inputs(mu: Any, pose: Any, workspace: ProjectionWorkspace, stream: wp.Stream | None) -> None:
    ctx = workspace.context
    ctx.assert_stream(stream)
    ctx.array(mu, "mu", dtype=workspace.field_dtype, shape=(workspace.grid.voxels,))
    ctx.array(pose, "pose", dtype=ctx.wp.float64, shape=(12,))


def _metadata(workspace: ProjectionWorkspace) -> tuple[Any, ...]:
    return tuple(
        id(value)
        for value in (
            workspace.grid,
            workspace.geometry,
            workspace.spec,
            workspace.context,
            workspace._configuration,
        )
    )


def _resolve_inputs(
    mu: Any, pose: Any, workspace: ProjectionWorkspace, stream: wp.Stream | None
) -> tuple[Any, Any, bool]:
    prepared = isinstance(mu, PreparedProjectionInputs)
    if prepared:
        if pose is not None:
            raise ContractError("a prepared input record already owns its pose")
        if mu._workspace is not workspace or mu._metadata != _metadata(workspace):
            raise ContractError("prepared inputs belong to their original workspace configuration")
        mu, pose = mu._mu, mu._pose
    _inputs(mu, pose, workspace, stream)
    return mu, pose, prepared


def _validation_policy(
    validate: ValidationPolicy | bool | None, workspace: ProjectionWorkspace, prepared: bool
) -> ValidationPolicy:
    if validate is None:
        policy = workspace.spec.validate
        return "always" if policy == "once" and not prepared else policy
    if type(validate) is bool:
        return "always" if validate else "never"
    if validate not in ("once", "always", "never"):
        raise ContractError("validate must be once, always, never or a boolean")
    if validate == "once" and not prepared:
        raise ContractError("validate='once' requires workspace.prepare_inputs(mu, pose)")
    return validate


def _validate_evaluation(
    mu: Any, pose: Any, workspace: ProjectionWorkspace, policy: ValidationPolicy, seed: Any = None
) -> None:
    if policy == "never":
        return
    if policy == "always":
        _validate(mu, pose, workspace, seed)
        return
    # The owned snapshot has already passed its field and pose checks. Check
    # pending diagnostics before writing and any current caller-owned seed.
    if seed is None:
        workspace.check_status()
    else:
        workspace.context.wp.launch(
            workspace._kernels.get_validate_finite(workspace.spec.precision == "float64"),
            dim=workspace.geometry.pixels,
            inputs=[seed],
            outputs=[workspace._status],
            stream=workspace.stream,
            device=workspace.device,
            record_tape=False,
        )
        workspace.check_status()


def _validate(mu: Any, pose: Any, workspace: ProjectionWorkspace, seed: Any = None) -> None:
    wp = workspace.context.wp
    workspace.check_status()
    wp.launch(
        _field_check(workspace),
        dim=workspace.grid.voxels,
        inputs=[mu],
        outputs=[workspace._status],
        stream=workspace.stream,
        device=workspace.device,
        record_tape=False,
    )
    wp.launch(
        workspace._geometry_kernels.validate_pose,
        dim=1,
        inputs=[pose],
        outputs=[workspace._status],
        stream=workspace.stream,
        device=workspace.device,
        record_tape=False,
    )
    if seed is not None:
        wp.launch(
            workspace._kernels.get_validate_finite(workspace.spec.precision == "float64"),
            dim=workspace.geometry.pixels,
            inputs=[seed],
            outputs=[workspace._status],
            stream=workspace.stream,
            device=workspace.device,
            record_tape=False,
        )
    workspace.check_status()


def _field_check(workspace: ProjectionWorkspace) -> Any:
    if workspace.spec.storage_precision == "float32":
        return (
            workspace._kernels.validate_nonnegative
            if workspace.spec.field_domain == "physical"
            else workspace._kernels.validate_finite
        )
    return workspace._kernels.get_validate_field(
        workspace.spec.storage_precision, workspace.spec.field_domain == "physical"
    )


def project_optical_depth(
    mu: FieldArray | PreparedProjectionInputs,
    pose: wp.array[wp.float64] | None = None,
    *,
    workspace: ProjectionWorkspace,
    out_L: DepthArray,
    stream: wp.Stream | None = None,
    tape: wp.Tape | None = None,
    validate: ValidationPolicy | bool | None = None,
) -> None:
    """L[p] = integral mu(r_p(s)) ds integrates each clipped source-to-detector segment.

    Mu is contiguous (grid.voxels,) storage, flattened from (nz,ny,nx) with x
    fastest, in mm^-1 and the declared FP32/FP16/BF16 field dtype. Pose is FP64
    (12,), ordered row-major object-to-world rotation followed by translation in
    mm. A PreparedProjectionInputs record owns both arrays and replaces the two
    arguments. Out_L is (pixels,) in the declared depth dtype, dimensionless and
    flattened in view,row,column order. The chosen quadrature and local sampling
    arithmetic define the discrete map. Ray setup remains FP64; accumulation
    is FP32 for float32 sampling and FP64 for the other sampling policies.
    Physical fields require nonnegative samples and signed_linear permits
    finite signed mathematical fields. Grid and acquisition stay fixed.

    Checked calls validate raw inputs each time or owned snapshots once under
    the default policy, then deliver range diagnostics. Unchecked epochs require
    explicit input validation and a final workspace.check_status(). A tape may
    record raw inputs with their declared active field/pose parameters, retained
    unchanged until backward completes.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_pose: Any = cast(Any, pose)
    ensure_tape(tape)
    mu, _api_pose, prepared = _resolve_inputs(mu, _api_pose, workspace, stream)
    policy = _validation_policy(validate, workspace, prepared)
    if workspace._recorded_tape is not None:
        raise ContractError("finish or discard the outstanding projection tape before reuse")
    ctx = workspace.context
    ctx.array(out_L, "out_L", dtype=workspace.dtype, shape=(workspace.geometry.pixels,))
    ctx.disjoint([("mu", mu), ("pose", _api_pose)], [("out_L", out_L)])
    if tape is not None:
        if prepared:
            raise ContractError("prepared snapshots use explicit projection_vjp gradients")
        if workspace.spec.storage_precision != "float32":
            raise ContractError(
                "half-storage projections use explicit FP32 or FP64 VJP destinations"
            )
        _check_tape_arrays(mu, _api_pose, out_L, workspace)
    _validate_evaluation(mu, _api_pose, workspace, policy)
    if workspace._native is not None and not ctx.device.is_capturing:
        workspace._native.forward(
            mu, _api_pose, out_L, workspace.spec.precision == "float64", workspace._status
        )
    else:
        ctx.wp.launch(
            _forward_kernel(workspace),
            dim=workspace.geometry.pixels,
            inputs=[mu, _api_pose, workspace._configuration],
            outputs=[out_L, workspace._status],
            stream=workspace.stream,
            device=workspace.device,
            block_dim=_FORWARD_BLOCK,
            record_tape=False,
        )
    if policy != "never":
        workspace.check_status()
    if tape is not None:
        _record(tape, mu, _api_pose, out_L, workspace)


def _check_tape_arrays(
    mu: Any, pose: Any, output: Any, workspace: ProjectionWorkspace
) -> list[Any]:
    arrays = [output]
    if workspace.spec.active_pose:
        arrays.append(pose)
    if workspace.spec.active_volume:
        arrays.append(mu)
    if len(arrays) == 1:
        raise ContractError("recording requires at least one active input")
    if any(array.grad is None for array in arrays):
        raise ContractError("active tape arrays require requires_grad=True at allocation")
    workspace.context.disjoint(
        [("mu", mu), ("pose", pose), ("output", output)],
        [("gradient", array.grad) for array in arrays],
    )
    return arrays


def _record(tape: Any, mu: Any, pose: Any, output: Any, workspace: ProjectionWorkspace) -> None:
    arrays = _check_tape_arrays(mu, pose, output, workspace)
    workspace._recorded_tape = tape
    if workspace.context.wp.config.verify_autograd_array_access:
        mu.mark_read()
        pose.mark_read()
        output.mark_write()
        # Register fixed inputs too, so Tape.reset() clears debug read markers.
        tape.record_launch(
            workspace._kernels.get_dependency_marker(workspace.spec.precision == "float64"),
            dim=0,
            max_blocks=0,
            inputs=[mu, pose],
            outputs=[output],
            device=workspace.device,
            block_dim=_BLOCK,
        )

    def backward() -> None:
        # Array storage has twelve coordinates. Its cotangent is NOT the
        # six-coordinate right tangent accepted by the standalone optimiser.
        projection_vjp(
            mu,
            pose,
            cotangent=output.grad,
            workspace=workspace,
            out_pose_matrix=pose.grad if workspace.spec.active_pose else None,
            out_mu=mu.grad if workspace.spec.active_volume else None,
            accumulate=True,
            validate=False,
            _from_tape=True,
        )
        with workspace.context.scope():
            if not output.retain_grad:
                output.grad.zero_()
        workspace._recorded_tape = None

    tape.record_func(backward, arrays)


def projection_vjp(
    mu: FieldArray | PreparedProjectionInputs,
    pose: wp.array[wp.float64] | None = None,
    *,
    cotangent: DepthArray | None,
    workspace: ProjectionWorkspace,
    out_pose_tangent: wp.array[wp.float64] | None = None,
    out_pose_matrix: wp.array[wp.float64] | None = None,
    out_mu: DepthArray | None = None,
    outputs: ProjectionGradients | None = None,
    stream: wp.Stream | None = None,
    validate: ValidationPolicy | bool | None = None,
    accumulate: bool = False,
    volume_gradient_precision: Literal["float32", "float64"] = "float32",
    check_volume_gradient: bool = True,
    _from_tape: bool = False,
    _float32_volume_increments: bool = False,
) -> None:
    """(grad_mu, grad_pose) = J(mu,pose)^T*cotangent forms the discrete projection VJP.

    Mu is native field storage (voxels,) in mm^-1 and pose is FP64 (12,) with
    row-major R_WO followed by t_WO in mm, or both come from a prepared snapshot.
    Cotangent is (pixels,) in depth precision. Out_mu is (voxels,) in the separate
    volume_gradient_precision, FP32 by default or FP64, and uses scatter atomics.
    Out_pose_tangent is FP64 (6,) for pose @ exp(delta^), ordered translation
    x/y/z in mm then rotation x/y/z in rad. Out_pose_matrix is FP64 (12,) for
    unconstrained packed pose entries. Select one pose destination and request
    only parameters made active at preparation. ProjectionGradients groups these
    destinations.

    The volume product follows the declared sampling weights and the pose product
    includes the active clipping/interpolation branches. Grid and acquisition are
    fixed. Accumulate adds into existing destinations. Deferred volume checks
    require unchecked accumulation followed by a finite check of the entire
    destination before acceptance. Standalone calls preserve the input covector.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_pose: Any = cast(Any, pose)
    _api_out_pose_tangent: Any = cast(Any, out_pose_tangent)
    _api_out_pose_matrix: Any = cast(Any, out_pose_matrix)
    if outputs is not None:
        if not isinstance(cast(object, outputs), ProjectionGradients):
            raise ContractError("outputs must be ProjectionGradients")
        if (
            out_mu is not None
            or _api_out_pose_tangent is not None
            or _api_out_pose_matrix is not None
        ):
            raise ContractError("supply outputs or individual projection destinations")
        out_mu = outputs.volume
        _api_out_pose_tangent = outputs.pose_tangent
        _api_out_pose_matrix = outputs.pose_storage
    require_no_tape()
    mu, _api_pose, prepared = _resolve_inputs(mu, _api_pose, workspace, stream)
    policy = _validation_policy(validate, workspace, prepared)
    if type(check_volume_gradient) is not bool:
        raise ContractError("check_volume_gradient must be a boolean")
    if not check_volume_gradient and (policy != "never" or not accumulate or _from_tape):
        raise ContractError("deferred volume checks require unchecked accumulating standalone VJPs")
    if volume_gradient_precision not in ("float32", "float64"):
        raise ContractError("volume_gradient_precision must be float32 or float64")
    if type(_float32_volume_increments) is not bool or (
        _float32_volume_increments and volume_gradient_precision != "float64"
    ):
        raise ContractError("FP32 volume increments require an FP64 gradient destination")
    if _from_tape and volume_gradient_precision != "float32":
        raise ContractError("tape volume gradients must follow FP32 field storage")
    if workspace._recorded_tape is not None and not _from_tape:
        raise ContractError("the outstanding tape owns the projection inputs")
    if _api_out_pose_tangent is not None and _api_out_pose_matrix is not None:
        raise ContractError("request either a right-tangent or a storage pose gradient")
    target = _api_out_pose_tangent if _api_out_pose_tangent is not None else _api_out_pose_matrix
    if target is None and out_mu is None:
        raise ContractError("request at least one projection gradient")
    if target is not None and not workspace.spec.active_pose:
        raise ContractError("pose is fixed in this workspace")
    if out_mu is not None and not workspace.spec.active_volume:
        raise ContractError("volume is fixed in this workspace")
    ctx, wp = workspace.context, workspace.context.wp
    ctx.array(cotangent, "cotangent", dtype=workspace.dtype, shape=(workspace.geometry.pixels,))
    components = 12 if _api_out_pose_matrix is not None else 6
    writes: list[tuple[str, Any]] = []
    if target is not None:
        ctx.array(target, "pose gradient", dtype=wp.float64, shape=(components,))
        writes.append(("pose gradient", target))
    if out_mu is not None:
        ctx.array(
            out_mu,
            "out_mu",
            dtype=getattr(wp, volume_gradient_precision),
            shape=(workspace.grid.voxels,),
        )
        writes.append(("out_mu", out_mu))
    ctx.disjoint([("mu", mu), ("pose", _api_pose), ("cotangent", cotangent)], writes)
    _validate_evaluation(mu, _api_pose, workspace, policy, cotangent)
    if workspace._native is not None and out_mu is not None:
        workspace._native.adjoint(
            _api_pose,
            cotangent,
            out_mu,
            workspace.spec.precision == "float64",
            volume_gradient_precision == "float64",
            accumulate,
            workspace._status,
        )
        if check_volume_gradient:
            workspace.enqueue_volume_gradient_check(
                out_mu, volume_gradient_precision=volume_gradient_precision, stream=stream
            )
        # The existing pose kernel differentiates the same midpoint values.
        # Passing no volume destination prevents a second volume accumulation.
        if target is None:
            if policy != "never":
                workspace.check_status()
            return
        out_mu = None
    if out_mu is not None and not accumulate:
        with ctx.scope():
            out_mu.zero_()
    block = _vjp_block(workspace, active_pose=target is not None)
    count = (
        (workspace.geometry.pixels + block - 1) // block
        if ctx.device.is_cuda
        else workspace.geometry.pixels
    )
    wp.launch_tiled(
        _vjp_kernel(
            workspace,
            out_mu is not None,
            components == 12,
            target is not None,
            double_seed=workspace.spec.precision == "float64",
            cell_gauss=workspace.spec.integration == "cell_gauss",
            double_volume_gradient=volume_gradient_precision == "float64" and out_mu is not None,
            mixed=workspace.spec.sampling_precision != "float64",
            storage_precision=workspace.spec.storage_precision,
            float32_volume_increments=_float32_volume_increments,
        ),
        dim=count,
        inputs=[mu, _api_pose, workspace._configuration, cotangent],
        outputs=[
            out_mu if out_mu is not None else workspace._empty,
            workspace._reduction.partials[0]
            if workspace._reduction is not None
            else workspace._empty_partials,
            workspace._status,
        ],
        stream=workspace.stream,
        device=workspace.device,
        block_dim=block,
        record_tape=False,
    )
    if target is not None:
        assert workspace._reduction is not None
        workspace._reduction.finish(
            count,
            target,
            workspace._status,
            components=components,
            accumulate=accumulate,
        )
    if out_mu is not None and check_volume_gradient:
        # Atomic accumulation may overflow after individually finite increments.
        workspace.enqueue_volume_gradient_check(
            out_mu, volume_gradient_precision=volume_gradient_precision, stream=stream
        )
    if policy != "never":
        workspace.check_status()


def projection_pose_sensitivities(
    mu: FieldArray | PreparedProjectionInputs,
    pose: wp.array[wp.float64] | None,
    *,
    cotangent: DepthArray | None,
    out_jacobian: wp.array[wp.float64],
    workspace: ProjectionWorkspace,
    stream: wp.Stream | None = None,
    validate: ValidationPolicy | bool | None = None,
) -> None:
    """Out_jacobian[6*p+a] = cotangent[p]*dL[p]/d(delta[a]) exports weighted rows.

    Mu is native (voxels,) attenuation storage in mm^-1 and pose is FP64 (12,)
    object-to-world rotation/translation, or an owned prepared input record.
    Cotangent is (pixels,) in depth precision and out_jacobian is FP64
    (6*pixels,), with contiguous rows ordered tx,ty,tz,rx,ry,rz in mm/rad for
    pose @ exp(delta^).
    Unit weights export the unweighted optical-depth Jacobian. Missed rays and
    zero weights give zero rows. Pose must be active under FP32 or FP64 sampling,
    while the field, grid and acquisition are held fixed. Each row follows the
    selected clipping and interpolation branch. The output is caller-owned
    diagnostic storage with no device allocation or image transfer in this call.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_pose: Any = cast(Any, pose)
    _api_out_jacobian: Any = cast(Any, out_jacobian)
    require_no_tape()
    mu, _api_pose, prepared = _resolve_inputs(mu, _api_pose, workspace, stream)
    policy = _validation_policy(validate, workspace, prepared)
    if workspace._recorded_tape is not None:
        raise ContractError("the outstanding tape owns the projection inputs")
    if not workspace.spec.active_pose:
        raise ContractError("pose is fixed in this workspace")
    ctx, wp = workspace.context, workspace.context.wp
    pixels = workspace.geometry.pixels
    ctx.array(cotangent, "cotangent", dtype=workspace.dtype, shape=(pixels,))
    ctx.array(_api_out_jacobian, "out_jacobian", dtype=wp.float64, shape=(6 * pixels,))
    ctx.disjoint(
        [("mu", mu), ("pose", _api_pose), ("cotangent", cotangent)],
        [("out_jacobian", _api_out_jacobian)],
    )
    _validate_evaluation(mu, _api_pose, workspace, policy, cotangent)
    wp.launch_tiled(
        _vjp_kernel(
            workspace,
            False,
            False,
            True,
            per_ray=True,
            double_seed=workspace.spec.precision == "float64",
            cell_gauss=workspace.spec.integration == "cell_gauss",
            storage_precision=workspace.spec.storage_precision,
        ),
        dim=(pixels + _BLOCK - 1) // _BLOCK if ctx.device.is_cuda else pixels,
        inputs=[mu, _api_pose, workspace._configuration, cotangent],
        outputs=[workspace._empty, _api_out_jacobian, workspace._status],
        stream=workspace.stream,
        device=workspace.device,
        block_dim=_BLOCK,
        record_tape=False,
    )
    if policy != "never":
        workspace.check_status()


def prepare_projection_adjoint(
    pose: wp.array[wp.float64] | None,
    *,
    workspace: ProjectionWorkspace,
    max_work: int = 10_000_000,
    max_prepare_voxels: int = 16_777_216,
    max_rays_per_voxel: int = 256,
    max_bytes: int = 256 * 1024 * 1024,
    stream: wp.Stream | None = None,
) -> PreparedProjectionAdjoint:
    """Capture A(pose) ray setup and conservative strict-midpoint gather bounds.

    The owned FP64 pose and ray records remain independent of changes to the
    original buffers. Preparation validates all rays and canonical nodes,
    then counts candidate rays and nodes for every voxel. Only an accepted
    plan can write a gather destination. Flat cone detector candidates enclose
    the projected, clamped trilinear support; source-plane or ill-conditioned
    cases use the full panel. Arbitrary bundles retain every ray and clip only
    the sample indices. Each selected sample uses the original forward's
    position, scaling and corner weights.

    ``max_prepare_voxels`` and ``max_rays_per_voxel`` bound preparation's
    support queries and per-voxel ray loop; ``max_work`` bounds all-node ray
    validation and the accepted gather queries. ``max_bytes`` bounds owned
    device storage. Increase these limits explicitly for large volumes.

    Example::

        plan = prepare_projection_adjoint(pose, workspace=workspace)
        projection_adjoint_gather(
            plan, cotangent=image, workspace=workspace, out_mu=gradient
        )
    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_pose: Any = cast(Any, pose)
    require_no_tape()
    if workspace._recorded_tape is not None:
        raise ContractError("the outstanding tape owns the projection inputs")
    if workspace.spec.sampling_precision != "float64" or workspace.spec.integration != "midpoint":
        raise ContractError("prepared gather requires strict float64 midpoint sampling")
    integer(workspace.spec.samples_per_ray, "samples_per_ray", minimum=1, maximum=2**31 - 1)
    for value, name in (
        (max_work, "max_work"),
        (max_prepare_voxels, "max_prepare_voxels"),
        (max_rays_per_voxel, "max_rays_per_voxel"),
        (max_bytes, "max_bytes"),
    ):
        integer(value, name, minimum=1, maximum=2**63 - 1)
    integer(max_rays_per_voxel, "max_rays_per_voxel", minimum=1, maximum=2**31 - 1)
    voxels, pixels = workspace.grid.voxels, workspace.geometry.pixels
    setup_nodes = pixels * workspace.spec.samples_per_ray
    if voxels > max_prepare_voxels or voxels > max_work:
        raise ContractError("prepared gather exceeds max_prepare_voxels or max_work")
    if setup_nodes > max_work:
        raise ContractError("all-node ray validation exceeds max_work")
    # A counter reduction and its budget comparison sum both ray and node
    # counts. Bound that sum before launching any concurrent tile reductions.
    if voxels * pixels * (workspace.spec.samples_per_ray + 1) + voxels > 2**63 - 1:
        raise ContractError("prepared gather work counters exceed signed 64-bit range")
    ctx = workspace.context
    ctx.assert_stream(stream)
    if isinstance(_api_pose, PreparedProjectionInputs):
        _, _api_pose, _ = _resolve_inputs(_api_pose, None, workspace, stream)
    ctx.array(_api_pose, "pose", dtype=ctx.wp.float64, shape=(12,))
    kernels = load_kernels("dpt.kernels.projection_adjoint")
    ray_bytes = ctx.wp.types.type_size_in_bytes(kernels.GatherRay)
    device_bytes = pixels * ray_bytes + 12 * 8 + 4 * 24 + 6 * 8
    if device_bytes + 4 * 8 > max_bytes:
        raise ContractError(
            f"prepared gather requires {device_bytes + 32} device bytes; max_bytes is {max_bytes}"
        )
    cone = isinstance(workspace.geometry, DetectorGeometry)
    configuration = workspace._kernels.Configuration()
    # Warp copies composite scalar fields into this independent struct. The
    # source/end arrays are used only while capturing the owned ray records.
    for name in (
        "shape",
        "object_to_grid",
        "grid_origin",
        "source",
        "detector_origin",
        "column_step",
        "row_step",
        "width",
        "pixels",
        "samples",
        "is_ray_bundle",
        "ray_sources",
        "ray_endpoints",
    ):
        setattr(configuration, name, getattr(workspace._configuration, name))
    with ctx.scope():
        owned_pose = ctx.wp.empty(12, dtype=ctx.wp.float64, device=workspace.device)
        ctx.wp.copy(owned_pose, _api_pose, stream=workspace.stream)
        rays = ctx.wp.empty(pixels, dtype=kernels.GatherRay, device=workspace.device)
        coefficients = ctx.wp.zeros(4, dtype=ctx.wp.vec3d, device=workspace.device)
        metrics = ctx.wp.zeros(6, dtype=ctx.wp.float64, device=workspace.device)
        totals = ctx.wp.zeros(4, dtype=ctx.wp.int64, device=workspace.device)
        workspace.validate_pose(owned_pose, stream=stream)
        if cone:
            ctx.wp.launch(
                kernels.prepare_cone_coefficients,
                dim=1,
                inputs=[owned_pose, configuration],
                outputs=[coefficients],
                stream=workspace.stream,
                device=workspace.device,
                record_tape=False,
            )
        ctx.wp.launch(
            kernels.get_prepare_rays(cone),
            dim=pixels,
            inputs=[owned_pose, configuration, coefficients],
            outputs=[rays, metrics, workspace._status],
            stream=workspace.stream,
            device=workspace.device,
            record_tape=False,
        )
        workspace.check_status()
        ctx.wp.launch_tiled(
            kernels.get_count_gather_work(cone),
            dim=(voxels + _BLOCK - 1) // _BLOCK if ctx.device.is_cuda else voxels,
            inputs=[
                configuration,
                rays,
                coefficients,
                metrics,
                max_rays_per_voxel,
                ctx.wp.int64(max_work - voxels),
            ],
            outputs=[totals],
            block_dim=_BLOCK,
            stream=workspace.stream,
            device=workspace.device,
            record_tape=False,
        )
    candidate_rays, nodes, limited, fallbacks = (
        int(value) for value in HostReadback(ctx, totals).read()
    )
    if limited:
        raise ContractError(f"{limited} voxel bounds exceed max_rays_per_voxel")
    work = voxels + candidate_rays + nodes
    if work > max_work:
        raise ContractError(f"prepared gather needs {work} bounded queries; max_work is {max_work}")
    workspace.check_status()
    plan = object.__new__(PreparedProjectionAdjoint)
    for name, value in (
        ("_workspace", workspace),
        ("_metadata", _metadata(workspace)),
        ("_pose", owned_pose),
        ("_rays", rays),
        ("_coefficients", coefficients),
        ("_metrics", metrics),
        ("_configuration", configuration),
        ("_cone", cone),
        ("_voxel_count", voxels),
        ("candidate_rays", candidate_rays),
        ("quadrature_nodes", nodes),
        ("fallback_voxels", fallbacks),
        ("device_bytes", device_bytes),
        ("preparation_work_bound", setup_nodes + pixels + voxels * min(pixels, max_rays_per_voxel)),
    ):
        object.__setattr__(plan, name, value)
    return plan


def projection_adjoint_gather(
    pose: wp.array[wp.float64] | PreparedProjectionInputs | PreparedProjectionAdjoint | None,
    *,
    cotangent: DepthArray | None,
    workspace: ProjectionWorkspace,
    out_mu: DepthArray | None,
    volume_gradient_precision: Literal["float32", "float64"] = "float64",
    accumulate: bool = False,
    validate: ValidationPolicy | bool | None = None,
    max_work: int = 10000000,
    stream: wp.Stream | None = None,
) -> None:
    """Out_mu[v] = sum_p A(pose)[p,v]*cotangent[p] gathers the fixed-pose transpose.

    Cotangent is (pixels,) in depth precision and out_mu is (voxels,) in FP64
    by default or FP32 under volume_gradient_precision. Pose is FP64 (12,)
    object-to-world rotation/translation, a prepared input record, or a
    PreparedProjectionAdjoint containing captured pose and rays. The grid's
    (nz,ny,nx) field layout has x fastest. Each voxel sums in fixed ray/node
    order using FP64 and the forward's strict sampling weights, so its units
    are depth-covector units times mm. Only the field is differentiated.

    The raw-pose path supports strict midpoint/cell-Gauss and bounds its all-ray
    work with max_work before writing. A prepared midpoint plan reuses validated
    ray setup and conservative detector/sample culling, and checks its retained
    work bound. Accumulate reads each existing output once after its local sum.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_pose: Any = cast(Any, pose)
    require_no_tape()
    if workspace._recorded_tape is not None:
        raise ContractError("the outstanding tape owns the projection inputs")
    if workspace.spec.sampling_precision != "float64":
        raise ContractError("deterministic gather requires strict float64 sampling")
    if volume_gradient_precision not in ("float32", "float64"):
        raise ContractError("volume_gradient_precision must be float32 or float64")
    if type(accumulate) is not bool:
        raise ContractError("accumulate must be a boolean")
    integer(max_work, "max_work", minimum=1)
    plan = _api_pose if isinstance(_api_pose, PreparedProjectionAdjoint) else None
    if plan is not None and (
        plan._workspace is not workspace or plan._metadata != _metadata(workspace)
    ):
        raise ContractError("prepared adjoint belongs to its original workspace configuration")
    nodes = (
        2 * (sum(workspace.grid.shape) + 1)
        if workspace.spec.integration == "cell_gauss"
        else workspace.spec.samples_per_ray
    )
    work = (
        plan.work_bound
        if plan is not None
        else workspace.grid.voxels * workspace.geometry.pixels * nodes
    )
    if work > max_work:
        raise ContractError(
            f"deterministic gather needs at most {work} voxel/ray quadrature evaluations; "
            f"max_work is {max_work}"
        )
    prepared = plan is not None or isinstance(_api_pose, PreparedProjectionInputs)
    if plan is not None:
        _api_pose = plan._pose
    elif prepared:
        _, _api_pose, _ = _resolve_inputs(_api_pose, None, workspace, stream)
    policy = _validation_policy(validate, workspace, prepared)
    ctx = workspace.context
    ctx.assert_stream(stream)
    ctx.array(_api_pose, "pose", dtype=ctx.wp.float64, shape=(12,))
    ctx.array(cotangent, "cotangent", dtype=workspace.dtype, shape=(workspace.geometry.pixels,))
    ctx.array(
        out_mu,
        "out_mu",
        dtype=getattr(ctx.wp, volume_gradient_precision),
        shape=(workspace.grid.voxels,),
    )
    reads = [("pose", _api_pose), ("cotangent", cotangent)]
    if plan is not None:
        reads.extend(
            [
                ("prepared rays", plan._rays),
                ("prepared coefficients", plan._coefficients),
                ("prepared metrics", plan._metrics),
            ]
        )
    elif workspace._configuration.is_ray_bundle:
        reads.extend(
            [
                ("ray_sources", workspace._configuration.ray_sources),
                ("ray_endpoints", workspace._configuration.ray_endpoints),
            ]
        )
    ctx.disjoint(reads, [("out_mu", out_mu)])
    if policy != "never":
        if policy == "always":
            workspace.check_status()
            ctx.wp.launch(
                workspace._geometry_kernels.validate_pose,
                dim=1,
                inputs=[_api_pose],
                outputs=[workspace._status],
                stream=workspace.stream,
                device=workspace.device,
                record_tape=False,
            )
        ctx.wp.launch(
            workspace._kernels.get_validate_finite(workspace.spec.precision == "float64"),
            dim=workspace.geometry.pixels,
            inputs=[cotangent],
            outputs=[workspace._status],
            stream=workspace.stream,
            device=workspace.device,
            record_tape=False,
        )
        workspace.check_status()
    kernels = load_kernels("dpt.kernels.projection_adjoint")
    if plan is not None:
        ctx.wp.launch(
            kernels.get_culled_gather(
                volume_gradient_precision == "float64",
                workspace.spec.precision == "float64",
                plan._cone,
            ),
            dim=workspace.grid.voxels,
            inputs=[
                plan._configuration,
                plan._rays,
                plan._coefficients,
                plan._metrics,
                cotangent,
                accumulate,
            ],
            outputs=[out_mu, workspace._status],
            stream=workspace.stream,
            device=workspace.device,
            record_tape=False,
        )
    else:
        ctx.wp.launch(
            kernels.get_gather(
                volume_gradient_precision == "float64",
                workspace.spec.precision == "float64",
                workspace.spec.integration == "cell_gauss",
            ),
            dim=workspace.grid.voxels,
            inputs=[_api_pose, workspace._configuration, cotangent, accumulate],
            outputs=[out_mu, workspace._status],
            stream=workspace.stream,
            device=workspace.device,
            record_tape=False,
        )
    if policy != "never":
        workspace.check_status()


def _vjp_block(workspace: ProjectionWorkspace, *, active_pose: bool) -> int:
    """Choose the measured launch shape without changing pose reduction tiles."""
    return 256 if not active_pose and workspace.spec.sampling_precision == "float64" else _BLOCK


def _forward_kernel(workspace: ProjectionWorkspace) -> Any:
    """Select arithmetic once at the host boundary."""
    spec = workspace.spec
    if spec.sampling_precision == "float32":
        return load_kernels("dpt.kernels.projection_fp32").get_forward(spec.precision == "float64")
    return workspace._kernels.get_forward(
        spec.precision == "float64",
        spec.integration == "cell_gauss",
        spec.sampling_precision != "float64",
        storage_precision=spec.storage_precision,
        float32_partials=spec.sampling_precision == "fast_float32",
    )


def _vjp_kernel(workspace: ProjectionWorkspace, *args: Any, **kwargs: Any) -> Any:
    """Use the forward arithmetic for every derivative of that workspace."""
    if workspace.spec.sampling_precision == "float32":
        for option in ("cell_gauss", "mixed", "storage_precision", "float32_volume_increments"):
            kwargs.pop(option, None)
        return load_kernels("dpt.kernels.projection_fp32").get_vjp(*args, **kwargs)
    return workspace._kernels.get_vjp(*args, **kwargs)
