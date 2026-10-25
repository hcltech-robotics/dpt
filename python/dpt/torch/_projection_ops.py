"""Checked multiview Torch operators with one diagnostic checkpoint per call.

Only ordinary immutable geometry values cross the dispatcher. Native scratch
is private, bounded and keyed by device/stream; no workspace handle enters a
compiled graph. The functional operators allocate every returned tensor.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from threading import RLock
from typing import Any

import torch

from dpt._runtime import HostReadback, StatusCheckpoint, load_kernels, require_no_tape
from dpt.contracts import ContractError, NumericalError
from dpt.geometry import (
    BatchedGeometry,
    CurvedDetectorGeometry,
    DetectorGeometry,
    ParallelGeometry,
)
from dpt.geometry_device import prepare_geometry
from dpt.interop._torch_streams import stream_bridge
from dpt.interop.torch import _pack_cotangent
from dpt.projection import ProjectionSpec, ProjectionWorkspace, prepare_projection
from dpt.projection import project_optical_depth as _native_project
from dpt.projection import projection_vjp as _native_vjp
from dpt.volumes import GridSpec

_CACHE_LIMIT = 8
_CACHE: OrderedDict[tuple[str, str, int, bool, bool], _Prepared] = OrderedDict()
_CACHE_LOCK = RLock()
_TAGS = (torch.Tag.cudagraph_unsafe,)


def configuration(grid: GridSpec, geometry: BatchedGeometry, spec: ProjectionSpec) -> str:
    """Encode fixed physical values once, without process-local object handles."""
    kinds = {
        DetectorGeometry: "cone",
        ParallelGeometry: "parallel",
        CurvedDetectorGeometry: "curved",
    }
    return json.dumps(
        {
            "grid": asdict(grid),
            "views": [[kinds[type(panel)], asdict(panel)] for panel in geometry.views],
            "samples": spec.samples_per_ray,
            "precision": spec.precision,
            "backend": spec.backend,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@lru_cache(maxsize=16)
def _decode(value: str) -> tuple[GridSpec, BatchedGeometry, ProjectionSpec]:
    encoded = json.loads(value)
    constructors = {
        "cone": DetectorGeometry,
        "parallel": ParallelGeometry,
        "curved": CurvedDetectorGeometry,
    }

    # Native records require immutable tuples; JSON represents those as lists.
    def immutable(value):
        if isinstance(value, list):
            return tuple(immutable(item) for item in value)
        return value

    grid = GridSpec(**{key: immutable(item) for key, item in encoded["grid"].items()})
    geometry = BatchedGeometry(
        tuple(
            constructors[kind](**{key: immutable(item) for key, item in panel.items()})
            for kind, panel in encoded["views"]
        )
    )
    precision = encoded["precision"]
    if precision not in ("float32", "float64"):
        raise ContractError("Torch projection arithmetic must be float32 or float64")
    spec = ProjectionSpec(
        encoded["samples"],
        precision=precision,
        sampling_precision=precision,
        backend=encoded.get("backend", "warp"),
    )
    return grid, geometry, spec


@dataclass
class _ViewStatus:
    workspace: ProjectionWorkspace
    index: int

    @property
    def context(self):
        return self.workspace.context

    @property
    def _status(self):
        return self.workspace._status

    def check_status(self) -> None:
        try:
            self.workspace.check_status()
        except (ContractError, NumericalError) as error:
            error.args = (f"projection view {self.index}: {error}", *error.args[1:])
            raise

    def clear_status(self) -> None:
        self.workspace.clear_status()


@dataclass
class _Prepared:
    workspaces: tuple[ProjectionWorkspace, ...]
    statuses: Any
    checkpoint: StatusCheckpoint
    pose_kernel: Any
    forward_batch: Any = None
    status_readback: Any = None
    lock: Any = field(default_factory=RLock)

    @property
    def context(self):
        return self.workspaces[0].context

    def consume(self) -> None:
        if self.status_readback is None:
            self.checkpoint.consume()
            return
        with self.context.scope():
            codes = self.status_readback.read()
            if any(codes):
                self.checkpoint._deliver(codes, consume=True)


def prepare(field: torch.Tensor, value: str, active_volume: bool, active_pose: bool) -> _Prepared:
    """Borrow a strongly held entry for the complete checked operation."""
    import warp as wp

    wp.init()
    caller = torch.cuda.current_stream(field.device) if field.device.type == "cuda" else None
    stream_id = caller.cuda_stream if caller is not None else 0
    key = (value, str(field.device), stream_id, active_volume, active_pose)
    with _CACHE_LOCK:
        found = _CACHE.get(key)
        if found is not None:
            _CACHE.move_to_end(key)
            return found
        grid, geometry, spec = _decode(value)
        stream = wp.stream_from_torch(caller) if caller is not None else None
        spec = replace(spec, active_volume=active_volume, active_pose=active_pose)
        statuses = None
        workspaces = []
        for index, panel in enumerate(geometry.views):
            native = (
                panel
                if isinstance(panel, DetectorGeometry)
                else prepare_geometry(
                    BatchedGeometry((panel,)), device=str(field.device), stream=stream
                )
            )
            workspace = prepare_projection(
                grid, native, spec, device=str(field.device), stream=stream
            )
            if statuses is None:
                with workspace.context.scope():
                    statuses = wp.zeros(
                        len(geometry.views), dtype=wp.int32, device=workspace.device
                    )
            # Each native launch still has its ordinary one-word diagnostics;
            # the words also form one batch for rigid-pose validation.
            workspaces.append(replace(workspace, _status=statuses[index : index + 1]))
        workspaces = tuple(workspaces)
        checkpoint = StatusCheckpoint(
            workspaces[0].context,
            tuple(_ViewStatus(workspace, index) for index, workspace in enumerate(workspaces)),
        )
        found = _Prepared(
            workspaces, statuses, checkpoint, load_kernels("dpt.kernels.batched_geometry")
        )
        if 1 < len(workspaces) <= 65535 and all(
            workspace._native is not None and not workspace._configuration.is_ray_bundle
            for workspace in workspaces
        ):
            from dpt._cuda_texture import NativeBatch

            found.forward_batch = NativeBatch(tuple(workspace._native for workspace in workspaces))
            # The words already form a contiguous batch. Reading them directly
            # avoids one device copy and one zero-write launch for every view.
            found.status_readback = HostReadback(found.context, statuses)
        _CACHE[key] = found
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
        return found


def _metadata(field: torch.Tensor, poses: torch.Tensor, value: str):
    grid, geometry, spec = _decode(value)
    if (
        tuple(field.shape) != grid.shape
        or field.dtype != torch.float32
        or not field.is_contiguous()
    ):
        raise ContractError("field must use contiguous float32 storage with grid shape")
    if field.device.type not in ("cpu", "cuda"):
        raise ContractError("field must be on a CPU or CUDA device")
    if (
        tuple(poses.shape) != (len(geometry.views), 12)
        or poses.dtype != torch.float64
        or poses.device != field.device
    ):
        raise ContractError("poses must be float64 (views,12) on the field device")
    return geometry.shape, torch.float64 if spec.precision == "float64" else torch.float32


def _validate(entry: _Prepared, native_field: Any, native_poses: Any, seed: Any = None) -> None:
    workspace = entry.workspaces[0]
    workspace.enqueue_field_check(native_field)
    entry.context.wp.launch(
        entry.pose_kernel.validate_poses_per_view,
        dim=len(entry.workspaces),
        inputs=[native_poses],
        outputs=[entry.statuses],
        device=workspace.device,
        stream=workspace.stream,
        record_tape=False,
    )
    if seed is not None:
        workspace.enqueue_cotangent_check(seed)


def _forward(
    field: torch.Tensor,
    poses: torch.Tensor,
    value: str,
    active_volume: bool,
    active_pose: bool,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    require_no_tape()
    shape, dtype = _metadata(field, poses, value)
    if out is not None and (
        tuple(out.shape) != shape
        or out.dtype != dtype
        or out.device != field.device
        or not out.is_contiguous()
    ):
        raise ContractError("out must match the projector's shape, dtype and input device")
    if out is not None and any(
        out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr() for x in (field, poses)
    ):
        raise ContractError("out must not share storage with projection inputs")
    entry = prepare(field, value, active_volume, active_pose)
    inputs = (field, poses) if out is None else (field, poses, out)
    with entry.lock, torch.no_grad(), stream_bridge(torch, inputs, entry) as bridge:
        wp = entry.context.wp
        packed_poses = poses.contiguous()
        images = torch.empty(shape, dtype=dtype, device=field.device) if out is None else out
        bridge.record(packed_poses, images)
        native_field = wp.from_torch(field.reshape(-1), requires_grad=False)
        native_poses = wp.from_torch(packed_poses, requires_grad=False)
        _validate(entry, native_field, native_poses)
        if entry.forward_batch is not None and not entry.context.device.is_capturing:
            entry.forward_batch.forward(
                native_field,
                native_poses,
                wp.from_torch(images.reshape(-1), requires_grad=False),
                entry.statuses,
            )
        else:
            for index, workspace in enumerate(entry.workspaces):
                _native_project(
                    native_field,
                    wp.from_torch(packed_poses[index], requires_grad=False),
                    workspace=workspace,
                    out_L=wp.from_torch(images[index].reshape(-1), requires_grad=False),
                    stream=workspace.stream,
                    validate=False,
                )
        entry.consume()
    return images


@torch.library.custom_op("dpt::project", mutates_args=(), tags=_TAGS)
def project(
    field: torch.Tensor,
    poses: torch.Tensor,
    value: str,
    active_volume: bool,
    active_pose: bool,
) -> torch.Tensor:
    return _forward(field, poses, value, active_volume, active_pose)


@project.register_fake
def _project_fake(field, poses, value, active_volume, active_pose):
    shape, dtype = _metadata(field, poses, value)
    return torch.empty(shape, dtype=dtype, device=field.device)


@torch.library.custom_op("dpt::project_vjp", mutates_args=(), tags=_TAGS)
def project_vjp(
    field: torch.Tensor,
    poses: torch.Tensor,
    cotangent: torch.Tensor,
    value: str,
    active_volume: bool,
    active_pose: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    require_no_tape()
    shape, dtype = _metadata(field, poses, value)
    if (
        tuple(cotangent.shape) != shape
        or cotangent.dtype != dtype
        or cotangent.device != field.device
    ):
        raise ContractError("cotangent must match the projected image shape, dtype and device")
    if not active_volume and not active_pose:
        raise ContractError("request at least one projection gradient")
    entry = prepare(field, value, active_volume, active_pose)
    with (
        entry.lock,
        torch.no_grad(),
        stream_bridge(torch, (field, poses, cotangent), entry) as bridge,
    ):
        wp = entry.context.wp
        packed_poses = poses.contiguous()
        seed = _pack_cotangent(torch, cotangent).reshape(shape)
        pose_gradient = torch.zeros_like(packed_poses) if active_pose else poses.new_empty((0,))
        gradient = field.new_empty((0,))
        bridge.record(packed_poses, seed, gradient, pose_gradient)
        native_field = wp.from_torch(field.reshape(-1), requires_grad=False)
        _validate(
            entry,
            native_field,
            wp.from_torch(packed_poses, requires_grad=False),
            wp.from_torch(seed.reshape(-1), requires_grad=False),
        )
        # StackBackward visits its last input first. Keep each native scatter
        # separate and add in that same order; this candidate changes checks,
        # not the accumulation across views.
        for index in reversed(range(len(entry.workspaces))):
            workspace = entry.workspaces[index]
            increment = torch.zeros_like(field) if active_volume else None
            if increment is not None:
                bridge.record(increment)
            _native_vjp(
                native_field,
                wp.from_torch(packed_poses[index], requires_grad=False),
                workspace=workspace,
                cotangent=wp.from_torch(seed[index].reshape(-1), requires_grad=False),
                out_mu=wp.from_torch(increment.reshape(-1), requires_grad=False)
                if increment is not None
                else None,
                out_pose_matrix=wp.from_torch(pose_gradient[index], requires_grad=False)
                if active_pose
                else None,
                stream=workspace.stream,
                validate=False,
                accumulate=active_volume,
                check_volume_gradient=not active_volume,
            )
            if increment is not None:
                if gradient.numel() == 0:
                    gradient = increment
                else:
                    gradient.add_(increment)
        if active_volume:
            entry.workspaces[0].enqueue_volume_gradient_check(
                wp.from_torch(gradient.reshape(-1), requires_grad=False)
            )
        entry.checkpoint.consume()
    return gradient, pose_gradient


@project_vjp.register_fake
def _vjp_fake(field, poses, cotangent, value, active_volume, active_pose):
    shape, dtype = _metadata(field, poses, value)
    if (
        tuple(cotangent.shape) != shape
        or cotangent.dtype != dtype
        or cotangent.device != field.device
    ):
        raise ContractError("cotangent must match the projected image shape, dtype and device")
    return (
        field.new_empty(field.shape if active_volume else (0,)),
        poses.new_empty(poses.shape if active_pose else (0,)),
    )


def _setup_context(ctx, inputs, output):
    field, poses, value, _, _ = inputs
    ctx.save_for_backward(field, poses)
    ctx.configuration = value


def _backward(ctx, cotangent):
    if torch.is_grad_enabled():
        raise ContractError("the Torch projection adapter supports first-order derivatives only")
    field, poses = ctx.saved_tensors
    active_volume, active_pose = ctx.needs_input_grad[:2]
    gradient, pose_gradient = project_vjp(
        field, poses, cotangent, ctx.configuration, active_volume, active_pose
    )
    return (
        gradient if active_volume else None,
        pose_gradient if active_pose else None,
        None,
        None,
        None,
    )


project.register_autograd(_backward, setup_context=_setup_context)


@torch.library.custom_op("dpt::project_out", mutates_args={"out"}, tags=_TAGS)
def project_out(field: torch.Tensor, poses: torch.Tensor, value: str, out: torch.Tensor) -> None:
    _forward(field, poses, value, False, False, out)


@project_out.register_fake
def _out_fake(field, poses, value, out):
    shape, dtype = _metadata(field, poses, value)
    if (
        tuple(out.shape) != shape
        or out.dtype != dtype
        or out.device != field.device
        or not out.is_contiguous()
    ):
        raise ContractError("out must match the projector's shape, dtype and input device")
