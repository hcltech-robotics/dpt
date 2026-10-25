"""Prepared multiview projection as an ordinary PyTorch module."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch

from dpt.contracts import ContractError, finite_scalar, integer
from dpt.geometry import BatchedGeometry, PanelGeometry, RigidTransform
from dpt.projection import ProjectionSpec, ProjectionWorkspace
from dpt.volumes import GridSpec

from . import _projection_ops
from .geometry import compose_pose

type Precision = Literal["fp32", "fp64"]
type GeometryInput = BatchedGeometry | PanelGeometry | Sequence[PanelGeometry]

__all__ = ["Projector"]


def _projection_spec(
    precision: Precision, samples: int, backend: Literal["warp", "cuda_texture"] = "warp"
) -> ProjectionSpec:
    """The single mapping from user precision to native arithmetic."""
    if precision not in ("fp32", "fp64"):
        raise ContractError("precision must be fp32 or fp64")
    arithmetic = "float32" if precision == "fp32" else "float64"
    return ProjectionSpec(
        samples,
        active_pose=True,
        active_volume=True,
        precision=arithmetic,
        sampling_precision=arithmetic,
        backend=backend,
    )


def _geometry(geometry: GeometryInput) -> BatchedGeometry:
    if isinstance(geometry, BatchedGeometry):
        return geometry
    if isinstance(geometry, Sequence):
        return BatchedGeometry(tuple(geometry))
    return BatchedGeometry((geometry,))


def _samples(grid: GridSpec, samples_per_ray: int | None, step_mm: float | None) -> int:
    if step_mm is None:
        return integer(
            1024 if samples_per_ray is None else samples_per_ray,
            "samples_per_ray",
            minimum=1,
            maximum=2**31 - 1,
        )
    if samples_per_ray is not None:
        raise ContractError("choose samples_per_ray or step_mm")
    step = finite_scalar(step_mm, "step_mm", minimum=0.0)
    if step == 0.0:
        raise ContractError("step_mm must be positive")
    diagonal = math.hypot(
        *(count * pitch for count, pitch in zip(reversed(grid.shape), grid.spacing_mm, strict=True))
    )
    count = diagonal / step
    if not math.isfinite(count) or count > 2**31 - 1:
        raise ContractError("step_mm requires too many midpoint samples")
    return max(1, math.ceil(count))


class Projector(torch.nn.Module):
    """Project device attenuation samples to ``(views,rows,columns)`` optical depth.

    ``grid`` and ``geometry`` are fixed millimetre calibration. The input field
    is contiguous FP32 ``(nz,ny,nx)`` storage in mm^-1. ``precision`` selects
    repeated sample arithmetic and output dtype. FP64 retains FP32 field
    storage. FP32 keeps initial ray setup and the final pose chain in FP64,
    outside its FP32 sample loop. Midpoint sampling uses a fixed
    ``samples_per_ray`` count (1024 by default). ``step_mm`` resolves that count
    once from the support diagonal; each clipped ray's realised step is its
    physical length divided by this count. All views share the sampling rule.

    ``pose`` in forward is an optional FP64 six-coordinate right increment
    ``(tx,ty,tz,rx,ry,rz)`` in mm/radians, shared or ``(views,6)``. It acts about
    ``anchor`` (identity by default). ``packed_pose=`` instead supplies explicit
    FP64 row-major rotation and translation, shared ``(12,)`` or ``(views,12)``.
    Volume and pose derivatives are first order. Inputs must remain unchanged
    until backward completes; the native adapter retains their Torch storage.

    Native workspaces are cached by fixed geometry, device and calling stream.
    Every call validates mutable inputs and delivers diagnostics once after
    the complete multiview forward or backward operation. Volume gradients
    are accumulated in view order.
    Torch dispatcher registration supports FakeTensor and full-graph compilation;
    checked calls cannot be captured in a CUDA graph. ``out=`` writes directly
    into caller-owned output for inference only.

    ``backend='cuda_texture'`` selects the CUDA C++ FP32 midpoint kernels.
    Compilation requires nvcc; FP64 and CPU calculations use Warp.
    """

    anchor: torch.Tensor | None

    def __init__(
        self,
        *,
        grid: GridSpec,
        geometry: GeometryInput,
        precision: Precision = "fp32",
        samples_per_ray: int | None = None,
        step_mm: float | None = None,
        anchor: RigidTransform | torch.Tensor | None = None,
        backend: Literal["warp", "cuda_texture"] = "warp",
    ) -> None:
        super().__init__()
        if not isinstance(grid, GridSpec):
            raise ContractError("grid must be a GridSpec")
        self.grid = grid
        self.geometry = _geometry(geometry)
        self.precision = precision
        self.backend = backend
        self.samples_per_ray = _samples(grid, samples_per_ray, step_mm)
        self.step_mm = step_mm
        self._spec = _projection_spec(precision, self.samples_per_ray, backend)
        self._configuration = _projection_ops.configuration(self.grid, self.geometry, self._spec)
        self._fixed_poses: dict[str, torch.Tensor] = {}
        if isinstance(anchor, torch.Tensor):
            self._check_pose(anchor, 12, "anchor")
            self._fixed_anchor = None
            if isinstance(anchor, torch.nn.Parameter):
                self.register_parameter("anchor", anchor)
            else:
                self.register_buffer("anchor", anchor)
        else:
            if anchor is not None and not isinstance(anchor, RigidTransform):
                raise ContractError("anchor must be a RigidTransform or FP64 packed Torch tensor")
            self._fixed_anchor = RigidTransform() if anchor is None else anchor
            self.register_buffer("anchor", None)

    def _check_pose(self, value: torch.Tensor, components: int, name: str) -> None:
        shapes = ((components,), (len(self.geometry.views), components))
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
            raise ContractError(f"{name} must be a float64 Torch tensor")
        if tuple(value.shape) not in shapes:
            raise ContractError(f"{name} must have shape {shapes[0]} or {shapes[1]}")
        if not value.is_contiguous():
            raise ContractError(f"{name} must be contiguous")

    def _poses(
        self, field: torch.Tensor, pose: torch.Tensor | None, packed_pose: torch.Tensor | None
    ) -> torch.Tensor:
        if packed_pose is not None:
            if pose is not None:
                raise ContractError("choose pose or packed_pose")
            self._check_pose(packed_pose, 12, "packed_pose")
            result = packed_pose
        else:
            anchor = self.anchor
            if anchor is None:
                device = str(field.device)
                if device not in self._fixed_poses:
                    assert self._fixed_anchor is not None
                    self._fixed_poses[device] = torch.tensor(
                        self._fixed_anchor.packed(), dtype=torch.float64, device=field.device
                    )
                anchor = self._fixed_poses[device]
            if anchor.device != field.device:
                raise ContractError(
                    "move the Projector anchor to the field device with .to(device)"
                )
            self._check_pose(anchor, 12, "anchor")
            if pose is None:
                result = anchor
            else:
                self._check_pose(pose, 6, "pose")
                if pose.device != field.device:
                    raise ContractError("pose and field must share a device")
                result = compose_pose(anchor, pose)
        if result.device != field.device:
            raise ContractError("packed_pose and field must share a device")
        return result.expand(len(self.geometry.views), 12) if result.ndim == 1 else result

    def _prepare(self, field: torch.Tensor, poses: torch.Tensor) -> tuple[ProjectionWorkspace, ...]:
        # Used by the cold-start benchmark to measure preparation separately.
        return _projection_ops.prepare(
            field, self._configuration, field.requires_grad, poses.requires_grad
        ).workspaces

    def _check_output(self, out: torch.Tensor, field: torch.Tensor, poses: torch.Tensor) -> None:
        if not isinstance(out, torch.Tensor):
            raise ContractError("out must be a Torch tensor")
        dtype = torch.float64 if self.precision == "fp64" else torch.float32
        if (
            tuple(out.shape) != self.geometry.shape
            or out.dtype != dtype
            or out.device != field.device
        ):
            raise ContractError("out must match the projector's shape, dtype and input device")
        if not out.is_contiguous():
            raise ContractError("out must be contiguous")
        if out.requires_grad or (
            torch.is_grad_enabled() and (field.requires_grad or poses.requires_grad)
        ):
            raise ContractError(
                "out is an inference buffer; omit it for autograd or use torch.no_grad()"
            )

    def forward(
        self,
        field: torch.Tensor,
        pose: torch.Tensor | None = None,
        *,
        packed_pose: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return optical depth, with field and optional rigid-pose gradients."""
        if not isinstance(field, torch.Tensor) or tuple(field.shape) != self.grid.shape:
            raise ContractError("field must be a Torch tensor with grid shape (nz,ny,nx)")
        if field.dtype != torch.float32 or not field.is_contiguous():
            raise ContractError("field must use contiguous float32 storage")
        if field.device.type not in ("cuda", "cpu"):
            raise ContractError("field must be on a CUDA or CPU device")
        poses = self._poses(field, pose, packed_pose)
        if out is not None:
            self._check_output(out, field, poses)
        if out is None:
            return _projection_ops.project(
                field,
                poses,
                self._configuration,
                field.requires_grad and torch.is_grad_enabled(),
                poses.requires_grad and torch.is_grad_enabled(),
            )
        _projection_ops.project_out(field, poses, self._configuration, out)
        return out
