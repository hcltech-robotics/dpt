"""Compiled multiview autograd across CUDA streams and allocator reuse."""

import gc

import pytest
import torch
import warp as wp

from dpt.geometry import BatchedGeometry, DetectorGeometry
from dpt.torch import Projector
from dpt.volumes import GridSpec

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
@pytest.mark.parametrize("views", [2, 5, 9])
def test_compiled_projector_retains_inputs_across_backward_streams(precision, views, monkeypatch):
    torch.compiler.reset()
    grid = GridSpec((3, 3, 4), (1.0, 1.0, 1.0), (-1.5, -1.0, -1.0))
    panel = DetectorGeometry(
        (-8.0, 0.0, 0.0),
        (8.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0),
        (1, 1),
    )
    module = Projector(
        grid=grid,
        geometry=BatchedGeometry((panel,) * views),
        precision=precision,
        samples_per_ray=64,
    )
    compiled = torch.compile(module, backend="inductor", fullgraph=True)

    def inputs():
        field = (
            (
                0.25
                + 0.01
                * torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32, device="cuda").reshape(
                    1, 3, 1
                )
            )
            .expand(grid.shape)
            .clone()
            .requires_grad_()
        )
        pose = torch.zeros(6, dtype=torch.float64, device="cuda", requires_grad=True)
        return field, pose

    # Compile forward and backward before prohibiting runtime-wide waits.
    warm_field, warm_pose = inputs()
    torch.autograd.grad(compiled(warm_field, warm_pose).sum(), (warm_field, warm_pose))
    torch.cuda.synchronize()

    def forbidden(*args, **kwargs):
        raise AssertionError("Projector used device-wide synchronisation")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(wp, "synchronize_device", forbidden)
    monkeypatch.setattr(wp, "synchronize", forbidden)
    producer, caller, backward = (torch.cuda.Stream() for _ in range(3))
    with torch.cuda.stream(producer):
        field, pose = inputs()
    caller.wait_stream(producer)
    with torch.cuda.stream(caller):
        first = compiled(0.5 * field, pose)
        saved = first.detach().clone()
        second = compiled(0.25 * field, pose)
        loss = first.sum() + second.sum()
    del module, compiled
    gc.collect()
    backward.wait_stream(caller)
    with torch.cuda.stream(backward):
        # Unrelated allocations must not overwrite tensors saved by either call.
        churn = [torch.empty_like(field) for _ in range(16)]
        for tensor in churn:
            tensor.fill_(float("nan"))
        del churn
        field_gradient, pose_gradient = torch.autograd.grad(loss, (field, pose))
    torch.cuda.current_stream().wait_stream(backward)
    torch.testing.assert_close(first, saved, rtol=0, atol=0)
    torch.testing.assert_close(first, torch.full_like(first, 0.5), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(second, torch.full_like(second, 0.25), rtol=2e-6, atol=2e-7)
    # Each centre ray crosses four one-mm cells. The fields scale by 0.5+0.25.
    expected_field = torch.zeros_like(field)
    expected_field[1, 1, :] = 0.75 * views
    expected_pose = torch.zeros_like(pose)
    expected_pose[1] = -0.03 * views
    torch.testing.assert_close(field_gradient, expected_field, rtol=3e-6, atol=3e-7)
    torch.testing.assert_close(pose_gradient, expected_pose, rtol=3e-6, atol=3e-7)
    torch.compiler.reset()
