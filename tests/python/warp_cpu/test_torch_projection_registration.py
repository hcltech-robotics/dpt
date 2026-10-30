"""Dispatcher registration, completed checks and native multiview equivalence."""

from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch

from dpt._runtime import HostReadback, cpu_test_context
from dpt.contracts import ContractError, NumericalError, TrialDomainError
from dpt.geometry import (
    BatchedGeometry,
    CurvedDetectorGeometry,
    DetectorGeometry,
    ParallelGeometry,
    RigidTransform,
)
from dpt.interop.torch import project_optical_depth
from dpt.torch import Projector
from dpt.torch import _projection_ops as ops
from dpt.volumes import GridSpec


@pytest.fixture(autouse=True)
def _native_context(warp_device):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


@pytest.fixture
def _independent_compile():
    # Each parameter is an independent module contract, not a request to keep
    # more than Dynamo's default eight specialisations of one frame alive.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def problem(device, precision, mode="both", views=3):
    grid = GridSpec((3, 3, 4), (0.7, 0.8, 0.9), (-1.0, -1.0, -1.0))
    panel = DetectorGeometry(
        (-8.0, 0.1, -0.2),
        (8.0, -0.3, -0.4),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.2, 0.2),
        (2, 3),
    )
    geometry = BatchedGeometry(
        tuple(replace(panel, origin_mm=(8.0, -0.3 + i * 0.03, -0.4)) for i in range(views))
    )
    module = Projector(grid=grid, geometry=geometry, precision=precision, samples_per_ray=47)
    field = torch.linspace(0.1, 0.3, grid.voxels, dtype=torch.float32, device=device).reshape(
        grid.shape
    )
    field.requires_grad_(mode != "pose")
    poses = torch.tensor(RigidTransform().packed(), dtype=torch.float64, device=device).repeat(
        views, 1
    )
    poses.requires_grad_(mode != "volume")
    return module, field, poses


def legacy(module, field, poses):
    return torch.stack(
        [
            project_optical_depth(field, poses[i], workspace=workspace).reshape(
                module.geometry.shape_rows_columns
            )
            for i, workspace in enumerate(module._prepare(field, poses))
        ]
    )


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
@pytest.mark.parametrize("mode", ["volume", "pose", "both"])
@pytest.mark.parametrize("views", [1, 3, 4, 5, 8, 9, 17])
def test_registered_projection_matches_per_view_forward_and_gradients(
    warp_device, precision, mode, views
):
    module, field, poses = problem(warp_device, precision, mode, views)
    expected = legacy(module, field, poses)
    actual = module(field, packed_pose=poses)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    seed = torch.linspace(
        -0.7, 0.8, actual.numel(), dtype=actual.dtype, device=warp_device
    ).reshape(actual.shape)
    inputs = tuple(x for x in (field, poses) if x.requires_grad)
    old_gradients = torch.autograd.grad(expected, inputs, seed)
    new_gradients = torch.autograd.grad(actual, inputs, seed)
    for old, new in zip(old_gradients, new_gradients, strict=True):
        torch.testing.assert_close(new, old, rtol=5e-6, atol=5e-7)


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
@pytest.mark.parametrize("mode", ["volume", "pose", "both"])
@pytest.mark.usefixtures("_independent_compile")
def test_custom_operator_registration_and_fullgraph_backward(warp_device, precision, mode):
    module, field, poses = problem(warp_device, precision, mode)
    args = (field, poses, module._configuration, field.requires_grad, poses.requires_grad)
    result = torch.library.opcheck(ops.project, args)
    assert set(result.values()) == {"SUCCESS"}
    expected = module(field, packed_pose=poses)
    backend = "inductor" if warp_device != "cpu" else "aot_eager"
    compiled = torch.compile(module, backend=backend, fullgraph=True)
    actual = compiled(field, packed_pose=poses)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    inputs = tuple(x for x in (field, poses) if x.requires_grad)
    expected_gradients = torch.autograd.grad(expected.sum(), inputs)
    actual_gradients = torch.autograd.grad(actual.sum(), inputs)
    for old, new in zip(expected_gradients, actual_gradients, strict=True):
        torch.testing.assert_close(new, old, rtol=5e-6, atol=5e-7)


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
@pytest.mark.usefixtures("_independent_compile")
def test_fresh_compiled_tangent_pose_gradients(warp_device, precision):
    module, field, _ = problem(warp_device, precision)
    tangent = torch.tensor(
        [0.03, -0.02, 0.01, 0.04, -0.03, 0.02],
        dtype=torch.float64,
        device=warp_device,
        requires_grad=True,
    )
    expected = module(field, tangent)
    fresh = Projector(
        grid=module.grid, geometry=module.geometry, precision=precision, samples_per_ray=47
    )
    compiled = torch.compile(
        fresh, backend="aot_eager" if warp_device == "cpu" else "inductor", fullgraph=True
    )
    actual = compiled(field, tangent)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)
    before = torch.autograd.grad(expected.sum(), (field, tangent))
    after = torch.autograd.grad(actual.sum(), (field, tangent))
    for old, new in zip(before, after, strict=True):
        torch.testing.assert_close(new, old, rtol=5e-6, atol=5e-7)


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
@pytest.mark.parametrize("per_view", [False, True])
@pytest.mark.usefixtures("_independent_compile")
def test_compiled_anchor_and_pose_keep_broadcast_gradients(warp_device, precision, per_view):
    base, field, _ = problem(warp_device, precision)
    anchor = torch.tensor(RigidTransform().packed(), dtype=torch.float64, device=warp_device)
    tangent = torch.tensor(
        [0.03, -0.02, 0.01, 0.04, -0.03, 0.02], dtype=torch.float64, device=warp_device
    )
    if per_view:
        anchor = anchor.repeat(3, 1)
        tangent = tangent.repeat(3, 1)
    anchor = torch.nn.Parameter(anchor)
    tangent.requires_grad_()
    module = Projector(
        grid=base.grid,
        geometry=base.geometry,
        precision=precision,
        samples_per_ray=47,
        anchor=anchor,
    )
    expected = module(field, tangent)
    compiled = torch.compile(
        module, backend="aot_eager" if warp_device == "cpu" else "inductor", fullgraph=True
    )
    actual = compiled(field, tangent)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)
    inputs = (field, tangent, anchor)
    before = torch.autograd.grad(expected.sum(), inputs)
    after = torch.autograd.grad(actual.sum(), inputs)
    for old, new in zip(before, after, strict=True):
        torch.testing.assert_close(new, old, rtol=5e-6, atol=5e-7)


def test_first_order_boundary_rejects_create_graph(warp_device):
    module, field, poses = problem(warp_device, "fp64")
    with pytest.raises(ContractError, match="first-order"):
        torch.autograd.grad(module(field, packed_pose=poses).sum(), field, create_graph=True)


def test_one_completed_status_read_per_forward_and_backward(warp_device, monkeypatch):
    module, field, poses = problem(warp_device, "fp64")
    torch.autograd.grad(module(field, packed_pose=poses).sum(), (field, poses))
    reads = []
    original = HostReadback.read

    def observed(self):
        reads.append(self)
        return original(self)

    monkeypatch.setattr(HostReadback, "read", observed)
    image = module(field, packed_pose=poses)
    assert len(reads) == 1
    reads.clear()
    torch.autograd.grad(image.sum(), (field, poses))
    assert len(reads) == 1


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
@pytest.mark.parametrize(
    "failure",
    ["negative_field", "nan_field", "rotation", "nan_pose", "inf_pose", "nan_seed", "inf_seed"],
)
def test_deferred_errors_are_delivered_and_next_call_recovers(warp_device, precision, failure):
    module, field, poses = problem(warp_device, precision)
    expected = module(field, packed_pose=poses)
    if failure in ("negative_field", "nan_field"):
        broken = field.detach().clone()
        broken.flatten()[-1] = -1 if failure == "negative_field" else float("nan")
        with pytest.raises(TrialDomainError, match="finite and non-negative"):
            module(broken, packed_pose=poses)
    elif failure in ("rotation", "nan_pose", "inf_pose"):
        broken = poses.detach().clone()
        broken[2, 0] = {"rotation": 0, "nan_pose": float("nan"), "inf_pose": float("inf")}[failure]
        with pytest.raises(ContractError, match=r"projection view 2.*invalid rigid"):
            module(field, packed_pose=broken)
    else:
        seed = torch.ones_like(expected)
        seed[-1, -1, -1] = float("nan") if failure == "nan_seed" else float("inf")
        with pytest.raises(NumericalError, match="non-finite"):
            torch.autograd.grad(expected, (field, poses), seed, retain_graph=True)
    recovered = module(field, packed_pose=poses)
    torch.testing.assert_close(recovered, expected, rtol=0, atol=0)
    actual = torch.autograd.grad(recovered.sum(), (field, poses))
    reference = torch.autograd.grad(legacy(module, field, poses).sum(), (field, poses))
    for old, new in zip(reference, actual, strict=True):
        torch.testing.assert_close(new, old, rtol=5e-6, atol=5e-7)


@pytest.mark.parametrize("views", [2, 17])
def test_aggregate_volume_overflow_is_reported(warp_device, views):
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    panel = DetectorGeometry(
        (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (1, 1)
    )
    module = Projector(
        grid=grid, geometry=BatchedGeometry((panel,) * views), precision="fp32", samples_per_ray=1
    )
    field = torch.ones(grid.shape, dtype=torch.float32, device=warp_device, requires_grad=True)
    image = module(field)
    # Individual views fit in FP32; their sum does not.
    seed = torch.full_like(image, (1.5 / views) * torch.finfo(torch.float32).max)
    with pytest.raises(NumericalError, match="non-finite"):
        torch.autograd.grad(image, field, seed, retain_graph=True)
    recovered = torch.autograd.grad(image, field, torch.ones_like(image))[0]
    torch.testing.assert_close(recovered, torch.full_like(field, float(views)), rtol=0, atol=0)


def test_out_schema_declares_only_output_mutation(warp_device):
    module, field, poses = problem(warp_device, "fp64")
    field, poses = field.detach(), poses.detach()
    out = torch.empty(module.geometry.shape, dtype=torch.float64, device=warp_device)
    result = torch.library.opcheck(ops.project_out, (field, poses, module._configuration, out))
    assert set(result.values()) == {"SUCCESS"}
    schema = torch.ops.dpt.project_out.default._schema
    assert [arg.name for arg in schema.arguments if arg.alias_info and arg.alias_info.is_write] == [
        "out"
    ]


def test_parallel_and_curved_geometry_survive_primitive_encoding(warp_device):
    module, field, poses = problem(warp_device, "fp64")
    panel = module.geometry.views[0]
    parallel = ParallelGeometry(
        panel.origin_mm, (1.0, 0.0, 0.0), 16.0, panel.u, panel.v, panel.spacing_mm, panel.shape
    )
    # Centre the first curved pixel on its cylinder's radial direction.
    curved = CurvedDetectorGeometry(
        (-8.0, -0.3, -0.4), panel.origin_mm, panel.u, panel.v, panel.spacing_mm, panel.shape
    )
    module = Projector(
        grid=module.grid,
        geometry=BatchedGeometry((panel, parallel, curved)),
        precision="fp64",
        samples_per_ray=47,
    )
    torch.testing.assert_close(
        module(field, packed_pose=poses), legacy(module, field, poses), rtol=0, atol=0
    )


def test_saved_inputs_survive_cache_eviction_and_module_deletion(warp_device):
    module, field, poses = problem(warp_device, "fp64")
    expected = legacy(module, field, poses)
    actual = module(field, packed_pose=poses)
    for samples in range(51, 51 + ops._CACHE_LIMIT + 1):
        other = Projector(
            grid=module.grid, geometry=module.geometry, precision="fp64", samples_per_ray=samples
        )
        other(field.detach(), packed_pose=poses.detach())
    assert len(ops._CACHE) <= ops._CACHE_LIMIT
    del module
    before = torch.autograd.grad(expected.sum(), (field, poses))
    after = torch.autograd.grad(actual.sum(), (field, poses))
    for old, new in zip(before, after, strict=True):
        torch.testing.assert_close(new, old, rtol=5e-6, atol=5e-7)
