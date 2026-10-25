"""Selected public CUDA-backend numerics, ownership and fallback regressions."""

import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch
import warp as wp

from dpt.autodiff import FirstOrderPass
from dpt.contracts import ContractError, NumericalError, TrialDomainError
from dpt.geometry import BatchedGeometry, DetectorGeometry, RayBundleGeometry, RigidTransform
from dpt.projection import (
    ProjectionSpec,
    prepare_projection,
    project_optical_depth,
    projection_vjp,
)
from dpt.torch import Projector
from dpt.volumes import GridSpec

pytestmark = pytest.mark.gpu


def _scene(shape=(3, 4, 5)):
    grid = GridSpec(shape, (0.7, 0.8, 0.9), (-1.0, -1.0, -1.0))
    panel = DetectorGeometry(
        (-8.0, 0.1, -0.2),
        (8.0, -0.3, -0.4),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.2, 0.2),
        (2, 3),
    )
    return grid, panel


def _prepare(grid, geometry, *, backend="cuda_texture", stream=None, **options):
    spec = ProjectionSpec(
        47,
        active_pose=True,
        active_volume=True,
        sampling_precision="float32",
        backend=backend,
    )
    spec = replace(spec, **options)
    return prepare_projection(grid, geometry, spec, device="cuda:0", stream=stream)


def _arrays(workspace, values=None, pose=None, gradients=False):
    with workspace.context.scope():
        field = wp.array(
            np.linspace(0.01, 0.2, workspace.grid.voxels) if values is None else values,
            dtype=workspace.field_dtype,
            device=workspace.device,
            requires_grad=gradients,
        )
        packed = wp.array(
            RigidTransform().packed() if pose is None else pose,
            dtype=wp.float64,
            device=workspace.device,
            requires_grad=gradients,
        )
        output = wp.empty(
            workspace.geometry.pixels,
            dtype=workspace.dtype,
            device=workspace.device,
            requires_grad=gradients,
        )
        seed = wp.array(
            np.linspace(-0.4, 0.8, workspace.geometry.pixels),
            dtype=workspace.dtype,
            device=workspace.device,
        )
    return field, packed, output, seed


@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 3, 5), (3, 4, 5)])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("moved", [False, True])
def test_public_forward_matches_warp_for_shapes_poses_and_output_precision(shape, precision, moved):
    from dpt.geometry import compose_pose

    grid, panel = _scene(shape)
    pose = compose_pose(RigidTransform(), (0.03, -0.02, 0.01, 0.04, -0.03, 0.02))
    packed = pose.packed() if moved else RigidTransform().packed()
    results = []
    for backend in ("warp", "cuda_texture"):
        workspace = _prepare(grid, panel, backend=backend, precision=precision)
        field, transform, output, _ = _arrays(workspace, pose=packed)
        project_optical_depth(field, transform, workspace=workspace, out_L=output)
        results.append(output.numpy())
    np.testing.assert_allclose(results[1], results[0], rtol=2e-5, atol=3e-7)


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("value", [2.0**-149, 2.0**-140, 2.0**-126, 1e20, np.finfo(np.float32).max])
@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 2, 2)])
def test_texture_preserves_subnormal_and_maximum_finite_field_bits(precision, value, shape):
    grid = GridSpec(shape, (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((-0.5, 0.0, 0.0),), ((0.5, 0.0, 0.0),))
    results = []
    for backend in ("warp", "cuda_texture"):
        workspace = _prepare(grid, rays, backend=backend, samples_per_ray=1, precision=precision)
        field, pose, output, _ = _arrays(workspace, np.full(grid.voxels, value))
        project_optical_depth(field, pose, workspace=workspace, out_L=output)
        results.append(output.numpy())
    np.testing.assert_array_equal(results[1], results[0])
    assert results[1][0] == np.float32(value)


@pytest.mark.parametrize("samples", [1, 3, 17, 97])
def test_gather_tail_and_mixed_ray_masks_match_warp_bitwise(samples):
    grid = GridSpec((3, 4, 5), (1.0, 1.0, 1.0))
    sources = [(-1.0, 0.001, 0.5), (-1.0, 8.0, 8.0), (0.2, -1.0, 0.8)] * 13
    endpoints = [(4.0, 0.001, 0.5), (4.0, 8.0, 8.0), (0.4, 5.0, 1.1)] * 13
    rays = RayBundleGeometry(tuple(sources), tuple(endpoints))
    results = []
    for backend in ("warp", "cuda_texture"):
        workspace = _prepare(grid, rays, backend=backend, samples_per_ray=samples)
        field, pose, output, _ = _arrays(workspace)
        project_optical_depth(field, pose, workspace=workspace, out_L=output)
        results.append(output.numpy())
    np.testing.assert_array_equal(results[1], results[0])


def test_workspaces_share_atlas_and_repack_after_other_fields():
    grid, panel = _scene()
    stream = wp.Stream("cuda:0")
    first = _prepare(grid, panel, stream=stream)
    second = _prepare(grid, panel, stream=stream)
    field, pose, output, _ = _arrays(first)
    other, other_pose, other_output, _ = _arrays(second, np.full(grid.voxels, 0.7))
    project_optical_depth(field, pose, workspace=first, out_L=output, stream=stream)
    expected = output.numpy()
    project_optical_depth(other, other_pose, workspace=second, out_L=other_output, stream=stream)
    assert first._native.atlas is second._native.atlas
    assert first._native.atlas.nbytes >= grid.voxels * 4
    assert not np.array_equal(other_output.numpy(), expected)
    project_optical_depth(field, pose, workspace=first, out_L=output, stream=stream)
    np.testing.assert_array_equal(output.numpy(), expected)
    independent = _prepare(grid, panel, stream=wp.Stream("cuda:0"))
    isolated, isolated_pose, isolated_output, _ = _arrays(independent)
    project_optical_depth(
        isolated,
        isolated_pose,
        workspace=independent,
        out_L=isolated_output,
        stream=independent.stream,
    )
    assert independent._native.atlas is not first._native.atlas
    np.testing.assert_array_equal(isolated_output.numpy(), expected)


def test_small_endpoint_survives_high_contrast_corner():
    grid = GridSpec((1, 1, 2), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((1.0, -1.0, 0.0),), ((1.0, 1.0, 0.0),))
    workspace = _prepare(grid, rays, samples_per_ray=1)
    field, pose, output, _ = _arrays(workspace, [1e30, 2.0**-140])
    project_optical_depth(field, pose, workspace=workspace, out_L=output)
    assert output.numpy()[0] == np.float32(2.0**-140)


def test_failed_texture_replacement_can_retry_the_original_field(monkeypatch):
    from dpt import _cuda_texture

    grid, panel = _scene()
    workspace = _prepare(grid, panel)
    field, pose, output, _ = _arrays(workspace)
    project_optical_depth(field, pose, workspace=workspace, out_L=output)
    expected = output.numpy()
    replacement, _, _, _ = _arrays(workspace)

    def allocation_failure(*args):
        raise RuntimeError("injected texture allocation failure")

    with monkeypatch.context() as patch:
        patch.setattr(_cuda_texture, "_Texture", allocation_failure)
        with pytest.raises(RuntimeError, match="injected texture allocation failure"):
            project_optical_depth(replacement, pose, workspace=workspace, out_L=output)
        # Check the failure state before an invalid handle could reach the GPU.
        assert workspace._native.texture is None
    project_optical_depth(field, pose, workspace=workspace, out_L=output)
    np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("tape", [False, True])
def test_joint_pose_and_volume_reverse_matches_existing_warp(precision, tape):
    grid, panel = _scene()
    products = []
    for backend in ("warp", "cuda_texture"):
        workspace = _prepare(grid, panel, backend=backend, precision=precision)
        field, pose, output, seed = _arrays(workspace, gradients=tape)
        if tape:
            with FirstOrderPass([workspace]) as reverse:
                project_optical_depth(
                    field, pose, workspace=workspace, out_L=output, tape=reverse.tape
                )
            with pytest.raises(ContractError, match="outstanding"):
                project_optical_depth(field, pose, workspace=workspace, out_L=output)
            reverse.backward(output, seed)
            products.append((output.numpy(), field.grad.numpy(), pose.grad.numpy()))
            reverse.close()
        else:
            volume = wp.empty(grid.voxels, dtype=wp.float64, device=workspace.device)
            pose_gradient = wp.empty(12, dtype=wp.float64, device=workspace.device)
            project_optical_depth(field, pose, workspace=workspace, out_L=output)
            projection_vjp(
                field,
                pose,
                cotangent=seed,
                workspace=workspace,
                out_mu=volume,
                out_pose_matrix=pose_gradient,
                volume_gradient_precision="float64",
            )
            products.append((output.numpy(), volume.numpy(), pose_gradient.numpy()))
    np.testing.assert_allclose(products[1][0], products[0][0], rtol=2e-5, atol=3e-7)
    np.testing.assert_allclose(products[1][1], products[0][1], rtol=2e-5, atol=3e-7)
    np.testing.assert_allclose(products[1][2], products[0][2], rtol=3e-12, atol=3e-13)


def test_owned_stream_snapshot_and_raw_mutation_are_observed():
    grid, panel = _scene()
    stream = wp.Stream("cuda:0")
    workspace = _prepare(grid, panel, stream=stream)
    field, pose, output, seed = _arrays(workspace)
    snapshot = workspace.prepare_inputs(field, pose, stream=stream)
    project_optical_depth(snapshot, workspace=workspace, out_L=output, stream=stream)
    expected = output.numpy().copy()
    with workspace.context.scope():
        field.fill_(0.0)
    project_optical_depth(field, pose, workspace=workspace, out_L=output, stream=stream)
    np.testing.assert_array_equal(output.numpy(), np.zeros(panel.pixels, np.float32))
    project_optical_depth(snapshot, workspace=workspace, out_L=output, stream=stream)
    np.testing.assert_array_equal(output.numpy(), expected)
    gradient = wp.empty(grid.voxels, dtype=wp.float32, device=workspace.device)
    projection_vjp(snapshot, cotangent=seed, workspace=workspace, out_mu=gradient, stream=stream)
    assert np.any(gradient.numpy() != 0)
    with workspace.context.scope():
        field.fill_(-1.0)
    with pytest.raises(TrialDomainError):
        project_optical_depth(field, pose, workspace=workspace, out_L=output, stream=stream)
    np.testing.assert_array_equal(output.numpy(), expected)
    workspace.clear_status()


def test_unaligned_contiguous_field_view_uses_the_same_operator():
    grid, panel = _scene()
    workspace = _prepare(grid, panel)
    field, pose, output, _ = _arrays(workspace)
    backing = wp.empty(grid.voxels + 1, dtype=wp.float32, device=workspace.device)
    view = backing[1:]
    wp.copy(view, field, stream=workspace.stream)
    project_optical_depth(field, pose, workspace=workspace, out_L=output)
    expected = output.numpy().copy()
    project_optical_depth(view, pose, workspace=workspace, out_L=output)
    np.testing.assert_array_equal(output.numpy(), expected)


def test_pending_status_and_contract_errors_precede_native_output_writes():
    grid = GridSpec((1, 1, 2), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((-1.0, 0.0, 0.0),) * 2, ((3.0, 0.0, 0.0),) * 2)
    workspace = _prepare(grid, rays)
    field, pose, output, seed = _arrays(workspace)
    output.fill_(123.0)
    with pytest.raises(ContractError, match="overlap"):
        project_optical_depth(field, pose, workspace=workspace, out_L=field)
    with pytest.raises(ContractError, match="stream"):
        project_optical_depth(
            field, pose, workspace=workspace, out_L=output, stream=wp.Stream("cuda:0")
        )
    bad_pose = wp.array(
        (*RigidTransform().rotation, 1.7e308, -1.7e308, -1.7e308),
        dtype=wp.float64,
        device=workspace.device,
    )
    project_optical_depth(field, bad_pose, workspace=workspace, out_L=output, validate=False)
    output.fill_(123.0)
    with pytest.raises(NumericalError):
        project_optical_depth(field, pose, workspace=workspace, out_L=output)
    np.testing.assert_array_equal(output.numpy(), [123.0, 123.0])
    workspace.clear_status()
    seed.zero_()
    gradient = wp.empty(grid.voxels, dtype=wp.float32, device=workspace.device)
    with pytest.raises(NumericalError):
        projection_vjp(field, bad_pose, cotangent=seed, workspace=workspace, out_mu=gradient)
    workspace.clear_status()


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_native_owner_reports_to_the_current_workspace_status(reverse):
    grid, panel = _scene()
    original = _prepare(grid, panel)
    # Torch's prepared views use this same dataclass replacement to bind a
    # per-view diagnostic word while retaining the native resource owner.
    current = replace(original, _status=wp.zeros(1, dtype=wp.int32, device=original.device))
    field, _pose, output, seed = _arrays(current)
    bad_pose = wp.array(
        (*RigidTransform().rotation, 1.7e308, -1.7e308, -1.7e308),
        dtype=wp.float64,
        device=current.device,
    )
    if reverse:
        seed.zero_()
        gradient = wp.empty(grid.voxels, dtype=wp.float32, device=current.device)
        projection_vjp(
            field, bad_pose, cotangent=seed, workspace=current, out_mu=gradient, validate=False
        )
    else:
        project_optical_depth(field, bad_pose, workspace=current, out_L=output, validate=False)
    original.check_status()
    with pytest.raises(NumericalError):
        current.check_status()
    current.clear_status()


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
@pytest.mark.parametrize("storage", ["float32", "float16", "bfloat16"])
def test_strict_gauss_and_native_storage_keep_existing_warp_path(integration, storage):
    grid, panel = _scene()
    outputs = []
    for backend in ("warp", "cuda_texture"):
        workspace = _prepare(
            grid,
            panel,
            backend=backend,
            sampling_precision="float64",
            precision="float64",
            integration=integration,
            storage_precision=storage,
        )
        assert workspace._native is None
        field, pose, output, seed = _arrays(workspace)
        gradient = wp.empty(grid.voxels, dtype=wp.float64, device=workspace.device)
        project_optical_depth(field, pose, workspace=workspace, out_L=output)
        projection_vjp(
            field,
            pose,
            cotangent=seed,
            workspace=workspace,
            out_mu=gradient,
            volume_gradient_precision="float64",
        )
        outputs.append((output.numpy(), gradient.numpy()))
    np.testing.assert_array_equal(outputs[1][0], outputs[0][0])
    np.testing.assert_allclose(outputs[1][1], outputs[0][1], rtol=2e-12, atol=2e-12)


def test_torch_multiview_joint_gradients_and_mutation_boundary_match_warp():
    grid, panel = _scene()
    geometry = BatchedGeometry((panel, replace(panel, origin_mm=(8.0, -0.24, -0.31))))
    products = []
    for backend in ("warp", "cuda_texture"):
        projector = Projector(
            grid=grid, geometry=geometry, precision="fp32", samples_per_ray=47, backend=backend
        )
        field = (
            torch.linspace(0.01, 0.2, grid.voxels, device="cuda")
            .reshape(grid.shape)
            .requires_grad_()
        )
        pose = torch.tensor(
            [0.03, -0.02, 0.01, 0.04, -0.03, 0.02],
            dtype=torch.float64,
            device="cuda",
            requires_grad=True,
        )
        image = projector(field, pose)
        seed = torch.linspace(-0.4, 0.8, image.numel(), device="cuda").reshape(image.shape)
        volume_gradient, pose_gradient = torch.autograd.grad(image, (field, pose), seed)
        products.append((image.detach(), volume_gradient, pose_gradient))
        second = projector(field, pose)
        with torch.no_grad():
            field.add_(0.01)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            second.sum().backward()
    torch.testing.assert_close(products[1][0], products[0][0], rtol=2e-5, atol=3e-7)
    torch.testing.assert_close(products[1][1], products[0][1], rtol=2e-5, atol=3e-7)
    torch.testing.assert_close(products[1][2], products[0][2], rtol=3e-12, atol=3e-13)


@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 4, 5)])
def test_torch_batch_uses_each_pose_and_refreshes_mutated_field(shape):
    from dpt.geometry import compose_pose
    from dpt.torch import _projection_ops

    grid, panel = _scene(shape)
    grid = GridSpec(shape, grid.spacing_mm)
    geometry = BatchedGeometry((panel, replace(panel, origin_mm=(8.0, -0.2, -0.3)), panel))
    packed = torch.tensor(
        [
            compose_pose(RigidTransform(), (0.0, 0.0, 0.0, x, 0.0, 0.0)).packed()
            for x in (0.0, 0.3, 20.0)
        ],
        dtype=torch.float64,
        device="cuda",
    )
    field = torch.linspace(0.01, 0.2, grid.voxels, device="cuda").reshape(grid.shape)
    native = Projector(
        grid=grid, geometry=geometry, precision="fp32", samples_per_ray=17, backend="cuda_texture"
    )
    reference = Projector(grid=grid, geometry=geometry, precision="fp32", samples_per_ray=17)
    first = native(field, packed_pose=packed)
    torch.testing.assert_close(first, reference(field, packed_pose=packed), rtol=0, atol=0)
    entry = _projection_ops.prepare(field, native._configuration, False, False)
    assert entry.forward_batch is not None
    field.mul_(2)
    after = native(field, packed_pose=packed)
    torch.testing.assert_close(after, reference(field, packed_pose=packed), rtol=0, atol=0)
    assert not torch.equal(first, after)
    output = torch.empty_like(after)
    native(field, packed_pose=packed, out=output)
    torch.testing.assert_close(output, after, rtol=0, atol=0)


def test_torch_batch_reports_view_specific_failure_and_recovers():
    grid, panel = _scene()
    projector = Projector(
        grid=grid,
        geometry=BatchedGeometry((panel, panel)),
        precision="fp32",
        samples_per_ray=17,
        backend="cuda_texture",
    )
    field = torch.full(grid.shape, 0.1, device="cuda")
    packed = torch.tensor([RigidTransform().packed()] * 2, dtype=torch.float64, device="cuda")
    expected = projector(field, packed_pose=packed)
    invalid = packed.clone()
    invalid[1, 9] = 1.7e308
    with pytest.raises(NumericalError, match="projection view 1"):
        projector(field, packed_pose=invalid)
    torch.testing.assert_close(projector(field, packed_pose=packed), expected, rtol=0, atol=0)


def test_captured_field_remains_valid_after_workspace_uses_another_field():
    # Invalid texture handles can poison a CUDA context, so isolate this actual
    # capture/replay regression from the other GPU checks.
    script = """
import numpy as np
import warp as wp
from dpt.geometry import RayBundleGeometry,RigidTransform
from dpt.projection import ProjectionSpec,prepare_projection,project_optical_depth
from dpt.volumes import GridSpec
wp.init()
w=prepare_projection(GridSpec((1,1,2),(1.,1.,1.)),
    RayBundleGeometry(((.001,-1.,0.),),((.001,1.,0.),)),
    ProjectionSpec(1,active_pose=False,sampling_precision='float32',backend='cuda_texture'))
a=wp.array([0.,1.],dtype=wp.float32,device=w.device)
b=wp.array([0.,2.],dtype=wp.float32,device=w.device)
p=wp.array(RigidTransform().packed(),dtype=wp.float64,device=w.device)
out=wp.empty(1,dtype=wp.float32,device=w.device)
project_optical_depth(a,p,workspace=w,out_L=out)
with wp.ScopedCapture(device=w.device,stream=w.stream) as capture:
    project_optical_depth(a,p,workspace=w,out_L=out,validate=False)
project_optical_depth(b,p,workspace=w,out_L=out)
wp.capture_launch(capture.graph,stream=w.stream)
w.check_status()
np.testing.assert_allclose(out.numpy(),[.001],rtol=2e-5,atol=3e-7)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
