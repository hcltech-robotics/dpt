"""Actual CUDA stream ordering and temporary lifetime through Projector."""

import gc
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from dpt.geometry import BatchedGeometry, DetectorGeometry
from dpt.torch import Projector
from dpt.volumes import GridSpec

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_fresh_process_projector_initialises_cuda_and_differentiates(precision):
    script = textwrap.dedent(
        """
        import sys
        import torch
        from dpt.geometry import DetectorGeometry
        from dpt.torch import Projector
        from dpt.volumes import GridSpec

        # No Warp initialisation or test-suite fixtures run in this process.
        grid = GridSpec((3, 3, 4), (1.0, 1.0, 1.0), (-1.5, -1.0, -1.0))
        panel = DetectorGeometry(
            (-8.0, 0.0, 0.0), (8.0, 0.0, 0.0),
            (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (1, 1),
        )
        field = (0.25 + 0.01 * torch.tensor(
            [-1.0, 0.0, 1.0], dtype=torch.float32, device="cuda"
        ).reshape(1, 3, 1)).expand(grid.shape).clone().requires_grad_()
        pose = torch.zeros(6, dtype=torch.float64, device="cuda", requires_grad=True)
        projector = Projector(
            grid=grid, geometry=panel, precision=sys.argv[1], samples_per_ray=64
        )
        image = projector(field, pose)
        # The centre ray traverses four mm at 0.25 /mm.
        torch.testing.assert_close(image, torch.ones_like(image), rtol=2e-6, atol=2e-7)
        image.sum().backward()
        expected = torch.zeros_like(field)
        expected[1, 1, :] = 1.0
        torch.testing.assert_close(field.grad, expected, rtol=2e-6, atol=2e-7)
        # Translating the volume along y samples its linear field at -ty.
        expected_pose = torch.zeros_like(pose)
        expected_pose[1] = -0.04
        torch.testing.assert_close(pose.grad, expected_pose, rtol=3e-6, atol=2e-7)
        print("fresh CUDA forward/backward passed")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, precision],
        capture_output=True,
        text=True,
        timeout=float(os.environ.get("DPT_TEST_CLI_TIMEOUT", "180")),
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "fresh CUDA forward/backward passed" in completed.stdout


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
@pytest.mark.parametrize("views", [2, 5, 9])
def test_projector_orders_cached_workspaces_and_temporary_fields_across_cuda_streams(
    precision, views
):
    grid = GridSpec((3, 3, 4), (0.7, 0.8, 0.9), (-1.0, -1.0, -1.0))
    panel = DetectorGeometry(
        (-8.0, 0.1, -0.2),
        (8.0, -0.3, -0.4),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.2, 0.2),
        (2, 3),
    )
    module = Projector(
        grid=grid,
        geometry=BatchedGeometry((panel,) * views),
        precision=precision,
        samples_per_ray=47,
    )
    owner, caller = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(owner):
        field = torch.linspace(0.1, 0.3, grid.voxels, dtype=torch.float32, device="cuda").reshape(
            grid.shape
        )
        field.requires_grad_()
        first = module(field)
        expected_gradient = torch.autograd.grad(first.sum(), field)[0]
    caller.wait_stream(owner)
    with torch.cuda.stream(caller):
        image = module(0.5 * field)
        saved = image.detach().clone()
        second = module(0.25 * field)
        del module
        gc.collect()
        gradient = torch.autograd.grad(image.sum() + second.sum(), field)[0]
        torch.testing.assert_close(image, saved)
        torch.testing.assert_close(image, first * 0.5, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(gradient, expected_gradient * 0.75, rtol=5e-6, atol=5e-7)
    torch.cuda.current_stream().wait_stream(caller)
