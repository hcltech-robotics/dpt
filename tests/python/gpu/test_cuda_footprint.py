"""Actual CUDA footprint products against Warp and an independent midpoint matrix."""

import math

import numpy as np
import pytest
import warp as wp

from dpt.contracts import NumericalError
from dpt.geometry import DetectorGeometry, RayBundleGeometry, RigidTransform, compose_pose
from dpt.projection import (
    ProjectionSpec,
    prepare_projection,
    project_optical_depth,
    projection_vjp,
)
from dpt.validation.projection import _clip_segment
from dpt.volumes import GridSpec

pytestmark = pytest.mark.gpu


def _fma(a, b, c):
    return np.float32(float(a) * float(b) + float(c))


def _matrix(grid, rays, pose, samples):
    """Explicit rounded nodes/weights; no production traversal or CUDA helper."""
    shape = tuple(reversed(grid.shape))
    matrix = np.zeros((rays.pixels, grid.voxels), dtype=np.float64)
    for pixel in range(rays.pixels):
        origin, direction, _, _, length = _clip_segment(grid, pose, *rays.segment(pixel))
        origin, direction = np.asarray(origin, np.float32), np.asarray(direction, np.float32)
        lower, upper = np.float32(0), np.float32(1)
        for axis in range(3):
            low, high = np.float32(-0.5), np.float32(shape[axis] - 0.5)
            if direction[axis] == 0:
                if not low <= origin[axis] <= high:
                    upper = np.float32(-1)
            else:
                a = np.float32(np.float32(low - origin[axis]) / direction[axis])
                b = np.float32(np.float32(high - origin[axis]) / direction[axis])
                lower, upper = max(lower, min(a, b)), min(upper, max(a, b))
        if upper <= lower:
            continue
        step = np.float32(np.float32(upper - lower) / np.float32(samples))
        physical_step = np.float32(np.float32(length) * step)
        for sample in range(samples):
            position = _fma(np.float32(sample) + np.float32(0.5), step, lower)
            point = np.asarray([_fma(position, direction[a], origin[a]) for a in range(3)])
            if any(not -0.5 <= point[a] <= shape[a] - 0.5 for a in range(3)):
                continue
            clamped = [np.float32(min(max(point[a], 0), shape[a] - 1)) for a in range(3)]
            cell = [min(math.floor(clamped[a]), max(shape[a] - 2, 0)) for a in range(3)]
            fraction = [np.float32(clamped[a] - np.float32(cell[a])) for a in range(3)]
            for corner in range(8):
                indices = [min(cell[a] + ((corner >> a) & 1), shape[a] - 1) for a in range(3)]
                factors = [
                    fraction[a] if (corner >> a) & 1 else np.float32(1) - fraction[a]
                    for a in range(3)
                ]
                weight = np.float32(np.float32(factors[0] * factors[1]) * factors[2])
                voxel = (indices[2] * shape[1] + indices[1]) * shape[0] + indices[0]
                matrix[pixel, voxel] += float(np.float32(physical_step * weight))
    return matrix


def _workspace(grid, geometry, samples, backend, precision="float32"):
    return prepare_projection(
        grid,
        geometry,
        ProjectionSpec(
            samples,
            active_pose=False,
            active_volume=True,
            precision=precision,
            field_domain="signed_linear",
            sampling_precision="float32",
            backend=backend,
        ),
        device="cuda:0",
    )


def _products(workspace, values, pose, seeds, gradient_precision="float32", initial=None):
    field = wp.array(values, dtype=wp.float32, device=workspace.device)
    packed = wp.array(pose.packed(), dtype=wp.float64, device=workspace.device)
    seed = wp.array(seeds, dtype=workspace.dtype, device=workspace.device)
    image = wp.empty(workspace.geometry.pixels, dtype=workspace.dtype, device=workspace.device)
    gradient_dtype = getattr(wp, gradient_precision)
    gradient = (
        wp.empty(workspace.grid.voxels, dtype=gradient_dtype, device=workspace.device)
        if initial is None
        else wp.array(initial, dtype=gradient_dtype, device=workspace.device)
    )
    project_optical_depth(field, packed, workspace=workspace, out_L=image)
    projection_vjp(
        field,
        packed,
        workspace=workspace,
        cotangent=seed,
        out_mu=gradient,
        volume_gradient_precision=gradient_precision,
        accumulate=initial is not None,
    )
    return image.numpy().astype(np.float64), gradient.numpy().astype(np.float64)


def _cone_scene():
    angle = 0.19
    grid = GridSpec(
        (5, 6, 7),
        (1.2, 0.8, 1.1),
        (-3.2, -2.1, -1.7),
        (
            math.cos(angle),
            -math.sin(angle),
            0.0,
            math.sin(angle),
            math.cos(angle),
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )
    panel = DetectorGeometry(
        (-18.0, 0.17, 0.37),
        (18.0, -6.0, -5.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.2, 1.1),
        (9, 11),
    )
    pose = compose_pose(RigidTransform(), (0.07, -0.03, 0.02, 0.013, -0.021, 0.009))
    rng = np.random.default_rng(77191)
    return grid, panel, pose, rng.normal(size=grid.voxels).astype(np.float32)


@pytest.mark.parametrize("samples", [17, 97, 1024])
@pytest.mark.parametrize("gradient_precision", ["float32", "float64"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_cone_footprint_matches_warp_with_nonidentity_anisotropic_geometry(
    samples, gradient_precision, precision
):
    grid, panel, pose, values = _cone_scene()
    seeds = np.random.default_rng(9191).normal(size=panel.pixels)
    native = _workspace(grid, panel, samples, "cuda_texture", precision)
    reference = _workspace(grid, panel, samples, "warp", precision)
    actual = _products(native, values, pose, seeds, gradient_precision)
    expected = _products(reference, values, pose, seeds, gradient_precision)
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5, atol=3e-7)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-5, atol=3e-7)
    repeated = _products(native, values, pose, seeds, gradient_precision)
    np.testing.assert_array_equal(repeated[1], actual[1])
    accumulated = _products(native, values, pose, seeds, gradient_precision, initial=actual[1])
    np.testing.assert_allclose(accumulated[1], 2 * actual[1], rtol=2e-5, atol=3e-7)


@pytest.mark.parametrize("source_inside", [False, True])
def test_cone_culling_retains_independent_realised_matrix(source_inside):
    grid = GridSpec((3, 4, 5), (1.0, 1.0, 1.0))
    panel = DetectorGeometry(
        (0.25 if source_inside else -8.0, 0.01, 0.07),
        (9.0, -2.0, -2.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.7, 0.6),
        (9, 11),
    )
    rays = RayBundleGeometry(
        (panel.source_mm,) * panel.pixels,
        tuple(panel.pixel_centre(row, col) for row in range(9) for col in range(11)),
    )
    pose = RigidTransform()
    rng = np.random.default_rng(556)
    values = rng.uniform(0.01, 0.1, grid.voxels).astype(np.float32)
    seeds = rng.uniform(-1, 1, panel.pixels).astype(np.float32)
    matrix = _matrix(grid, rays, pose, 17)
    native = _workspace(grid, panel, 17, "cuda_texture")
    image, gradient = _products(native, values, pose, seeds, "float64")
    np.testing.assert_allclose(image, matrix @ values, rtol=2e-5, atol=3e-7)
    np.testing.assert_allclose(gradient, matrix.T @ seeds, rtol=2e-5, atol=3e-7)
    lhs, rhs = float(image @ seeds), float(values.astype(np.float64) @ gradient)
    assert abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12) < 1e-5


@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 3, 5), (3, 1, 5), (3, 4, 1), (3, 4, 5)])
@pytest.mark.parametrize("gradient_precision", ["float32", "float64"])
def test_clipped_boundary_and_singleton_nodes_match_independent_matrix(shape, gradient_precision):
    grid = GridSpec(shape, (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(
        (
            (-2.0, 0.001, 0.0),
            (0.1, 0.2, 0.0),
            (-0.5, -1.0, 0.0),
            (20.0, 20.0, 20.0),
            (-2.0, -0.5, -0.5),
        ),
        (
            (8.0, 0.001, 0.0),
            (0.4, 0.25, 0.0),
            (-0.5, 8.0, 0.0),
            (21.0, 20.0, 20.0),
            (8.0, -0.5, -0.5),
        ),
    )
    pose = RigidTransform()
    values = np.random.default_rng(427).uniform(0.01, 0.1, grid.voxels).astype(np.float32)
    seeds = np.asarray([0.7, -0.3, 0.2, 0.8, -0.1], dtype=np.float32)
    matrix = _matrix(grid, rays, pose, 97)
    native = _workspace(grid, rays, 97, "cuda_texture")
    image, gradient = _products(native, values, pose, seeds, gradient_precision)
    np.testing.assert_allclose(image, matrix @ values, rtol=2e-5, atol=3e-7)
    np.testing.assert_allclose(gradient, matrix.T @ seeds, rtol=2e-5, atol=3e-7)
    lhs, rhs = float(image @ seeds), float(values.astype(np.float64) @ gradient)
    assert abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12) < 1e-5


def test_fraction_below_hardware_texture_resolution_remains_in_the_transpose():
    grid = GridSpec((1, 1, 2), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((0.001, -1.0, 0.0),), ((0.001, 1.0, 0.0),))
    native = _workspace(grid, rays, 1, "cuda_texture")
    image, gradient = _products(native, [0.0, 1.0], RigidTransform(), [1.0], "float64")
    np.testing.assert_allclose(image, [0.001], rtol=2e-5, atol=3e-7)
    np.testing.assert_allclose(gradient, [0.999, 0.001], rtol=2e-5, atol=3e-7)
    # Mutating the realised weights to an eight-fractional-bit texture stencil
    # must fail the same comparison; this is not a hardware precision claim.
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(gradient, [1.0, 0.0], rtol=2e-5, atol=3e-7)


@pytest.mark.parametrize("samples", [17, 1024])
@pytest.mark.parametrize("gradient_precision", ["float32", "float64"])
def test_distant_source_retains_repeated_rounded_nodes(samples, gradient_precision):
    grid = GridSpec((1, 1, 2), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((-(2.0**20), 0.001, 0.0),), ((2.0**20, 0.001, 0.0),))
    native = _workspace(grid, rays, samples, "cuda_texture")
    reference = _workspace(grid, rays, samples, "warp")
    actual = _products(native, [0.1, 0.3], RigidTransform(), [-0.7], gradient_precision)
    expected = _products(reference, [0.1, 0.3], RigidTransform(), [-0.7], gradient_precision)
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5, atol=3e-7)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-5, atol=3e-7)


@pytest.mark.parametrize("gradient_precision", ["float32", "float64"])
def test_subnormal_terms_keep_destination_specific_atomic_semantics(gradient_precision):
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((-1.0, 0.0, 0.0),), ((1.0, 0.0, 0.0),))
    native = _workspace(grid, rays, 128, "cuda_texture")
    reference = _workspace(grid, rays, 128, "warp")
    actual = _products(native, [0.25], RigidTransform(), [1e-37], gradient_precision)[1]
    expected = _products(reference, [0.25], RigidTransform(), [1e-37], gradient_precision)[1]
    np.testing.assert_array_equal(actual, expected)
    if gradient_precision == "float32":
        assert actual[0] == 0.0
    else:
        assert actual[0] > 0.0


def test_finite_float64_seed_overflow_cannot_be_hidden_by_zero_corner_weights():
    grid = GridSpec((1, 1, 2), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((0.0, -1.0, 0.0),), ((0.0, 1.0, 0.0),))
    native = _workspace(grid, rays, 1, "cuda_texture", "float64")
    with pytest.raises(NumericalError):
        _products(native, [0.2, 0.3], RigidTransform(), [1e100], "float64")


@pytest.mark.parametrize("case", ["reversed", "near_source_plane", "wide_coefficients"])
def test_culling_interval_signs_and_extremes_keep_partial_detector_blocks(case):
    """Exercise both signs, denominator intervals crossing zero, and overflow."""
    grid = GridSpec((3, 4, 5), (1.0, 1.0, 1.0))
    if case == "reversed":
        source, origin = (9.0, 2.01, 1.07), (-8.0, 4.0, -2.0)
        column, spacing = (0.0, -1.0, 0.0), (0.43, 0.37)
    elif case == "near_source_plane":
        source, origin = (0.25, 0.01, 0.07), (0.25 + 1e-10, -2.0, -2.0)
        column, spacing = (0.0, 1.0, 0.0), (0.43, 0.37)
    else:
        source, origin = (0.25, 0.01, 0.07), (1e30, -8e10, -7e10)
        column, spacing = (0.0, 1.0, 0.0), (1e10, 1e10)
    panel = DetectorGeometry(source, origin, column, (0.0, 0.0, 1.0), spacing, (17, 19))
    values = np.random.default_rng(873).uniform(0.01, 0.1, grid.voxels).astype(np.float32)
    seeds = np.random.default_rng(882).uniform(-1, 1, panel.pixels).astype(np.float32)
    native = _workspace(grid, panel, 97, "cuda_texture")
    reference = _workspace(grid, panel, 97, "warp")
    actual = _products(native, values, RigidTransform(), seeds, "float64")
    expected = _products(reference, values, RigidTransform(), seeds, "float64")
    assert np.linalg.norm(expected[1]) > 0
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5, atol=3e-7)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-5, atol=3e-7)
    lhs, rhs = float(actual[0] @ seeds), float(values.astype(np.float64) @ actual[1])
    assert abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12) < 1e-5


@pytest.mark.parametrize("direction", [1e-40, -1e-40, 1e38])
def test_sample_reciprocal_overflow_and_subnormal_intervals_keep_contributors(direction):
    grid = GridSpec((1, 2, 5), (1.0, 1.0, 1.0))
    endpoint = (1e38, 0.001, 0.0) if direction == 1e38 else (6.0, direction, 0.0)
    source = (0.25, 0.001, 0.0) if direction == 1e38 else (-1.0, 0.0, 0.0)
    rays = RayBundleGeometry((source,), (endpoint,))
    values = np.linspace(0.01, 0.1, grid.voxels, dtype=np.float32)
    native = _workspace(grid, rays, 97, "cuda_texture")
    reference = _workspace(grid, rays, 97, "warp")
    actual = _products(native, values, RigidTransform(), [-0.7], "float64")
    expected = _products(reference, values, RigidTransform(), [-0.7], "float64")
    assert np.linalg.norm(expected[1]) > 0
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5, atol=3e-7)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-5, atol=3e-7)


@pytest.mark.parametrize("shape", [(8, 8, 8), (9, 10, 11), (17, 18, 19)])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("gradient_precision", ["float32", "float64"])
def test_tiled_footprint_keeps_halo_and_partial_tiles(shape, precision, gradient_precision):
    original, panel, pose, _ = _cone_scene()
    grid = GridSpec(shape, original.spacing_mm, original.origin_mm, original.orientation)
    values = np.random.default_rng(3517).uniform(0.01, 0.1, grid.voxels).astype(np.float32)
    seeds = np.random.default_rng(9031).uniform(-1, 1, panel.pixels)
    native = _workspace(grid, panel, 97, "cuda_texture", precision)
    reference = _workspace(grid, panel, 97, "warp", precision)
    actual = _products(native, values, pose, seeds, gradient_precision)
    expected = _products(reference, values, pose, seeds, gradient_precision)
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5, atol=3e-7)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-5, atol=3e-7)
    initial = np.linspace(-0.03, 0.07, grid.voxels)
    accumulated = _products(native, values, pose, seeds, gradient_precision, initial)
    accumulated_reference = _products(reference, values, pose, seeds, gradient_precision, initial)
    np.testing.assert_allclose(accumulated[1], accumulated_reference[1], rtol=2e-5, atol=3e-7)
    lhs, rhs = float(actual[0] @ seeds), float(values.astype(np.float64) @ actual[1])
    assert abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12) < 1e-5


def test_tiled_footprint_matches_independent_matrix():
    grid = GridSpec((9, 10, 11), (1.0, 1.0, 1.0))
    panel = DetectorGeometry(
        (-8.0, 4.01, 3.07),
        (20.0, -2.0, -2.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.3, 1.2),
        (9, 11),
    )
    rays = RayBundleGeometry(
        (panel.source_mm,) * panel.pixels,
        tuple(panel.pixel_centre(row, column) for row in range(9) for column in range(11)),
    )
    pose = RigidTransform()
    matrix = _matrix(grid, rays, pose, 17)
    values = np.random.default_rng(471).uniform(0.01, 0.1, grid.voxels).astype(np.float32)
    seeds = np.random.default_rng(913).uniform(-1, 1, panel.pixels).astype(np.float32)
    native = _workspace(grid, panel, 17, "cuda_texture")
    image, gradient = _products(native, values, pose, seeds, "float64")
    assert (
        np.linalg.norm(matrix[:, np.arange(grid.voxels).reshape(grid.shape)[:, :, 8:].ravel()]) > 0
    )
    np.testing.assert_allclose(image, matrix @ values, rtol=2e-5, atol=3e-7)
    np.testing.assert_allclose(gradient, matrix.T @ seeds, rtol=2e-5, atol=3e-7)


@pytest.mark.parametrize(
    "case", ["subnormal_terms", "subnormal_cancellation", "subnormal_cancellation_from_seeds"]
)
@pytest.mark.parametrize("gradient_precision", ["float32", "float64"])
def test_tiled_shared_sums_preserve_global_atomic_subnormal_semantics(case, gradient_precision):
    grid = GridSpec((8, 8, 8), (1.0, 1.0, 1.0))
    panel = DetectorGeometry(
        (-0.5, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1e-40, 1.0),
        (1, 2 if case == "subnormal_cancellation_from_seeds" else 1),
    )
    values = np.linspace(0.01, 0.1, grid.voxels, dtype=np.float32)
    initial = None
    if case == "subnormal_terms":
        seed, samples = 1e-37, 128
    elif case == "subnormal_cancellation":
        seed, samples = np.finfo(np.float32).tiny, 1
        initial = np.zeros(grid.voxels, dtype=np.float64)
        initial[0] = -float(np.nextafter(np.float32(seed), np.float32(np.inf)))
    else:
        seed, samples = np.finfo(np.float32).tiny, 1
    seeds = (
        [-float(np.nextafter(np.float32(seed), np.float32(np.inf))), seed]
        if case == "subnormal_cancellation_from_seeds"
        else [seed]
    )
    native = _workspace(grid, panel, samples, "cuda_texture")
    reference = _workspace(grid, panel, samples, "warp")
    actual = _products(native, values, RigidTransform(), seeds, gradient_precision, initial)[1]
    expected = _products(reference, values, RigidTransform(), seeds, gradient_precision, initial)[
        1
    ]
    np.testing.assert_array_equal(actual, expected)
    if gradient_precision == "float32":
        np.testing.assert_array_equal(actual, np.zeros(grid.voxels))
    elif case.startswith("subnormal_cancellation"):
        assert actual[0] == -float(np.nextafter(np.float32(0), np.float32(1)))
    else:
        assert actual.sum() > 0
