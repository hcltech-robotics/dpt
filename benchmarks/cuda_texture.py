"""Compare checked FP32 projection backends on resident, nonconstant fields.

The default synthetic workload is a smoke benchmark. Use --prepared with a
verified fullscan input directory for real CT data. No ASTRA speed claim follows
from this benchmark. Outputs and every timing repetition stay outside the repo.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORACLE_RTOL = 2e-5
ORACLE_ATOL = 3e-7
ORACLE_ADJOINT_LIMIT = 1e-5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(UTC).isoformat()


def command(*args):
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    return dict(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def error(actual, expected):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    delta = a - b
    return dict(
        max_abs=float(np.max(np.abs(delta))),
        rmse=float(np.sqrt(np.mean(delta * delta))),
        relative_l2=float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(b.ravel()), 1e-300)),
        bitwise_equal=bool(np.array_equal(actual, expected)),
    )


def reference_matrix(grid, rays, pose, samples):
    """CPU rounded-weight oracle used by test_projection_fp32.py, without Warp."""
    from dpt.validation.projection import _clip_segment

    def fma(a, b, c):
        return np.float32(float(a) * float(b) + float(c))

    matrix = np.zeros((rays.rays, grid.voxels), dtype=np.float64)
    shape = grid.shape[::-1]
    for pixel in range(rays.rays):
        origin, direction, _, _, length = _clip_segment(grid, pose, *rays.segment(pixel))
        origin, direction = np.float32(origin), np.float32(direction)
        enter, leave = np.float32(0), np.float32(1)
        for axis in range(3):
            low, high = np.float32(-0.5), np.float32(shape[axis] - 0.5)
            velocity = direction[axis]
            if velocity == 0:
                if origin[axis] < low or origin[axis] > high:
                    leave = np.float32(-1)
            else:
                first = np.float32(np.float32(low - origin[axis]) / velocity)
                last = np.float32(np.float32(high - origin[axis]) / velocity)
                enter, leave = max(enter, min(first, last)), min(leave, max(first, last))
        if leave <= enter:
            continue
        step = np.float32(np.float32(leave - enter) / np.float32(samples))
        physical = np.float32(np.float32(length) * step)
        for sample in range(samples):
            position = fma(np.float32(sample) + np.float32(0.5), step, enter)
            point = [fma(position, direction[a], origin[a]) for a in range(3)]
            if any(point[a] < -0.5 or point[a] > shape[a] - 0.5 for a in range(3)):
                continue
            clamped = [np.float32(min(max(point[a], 0), shape[a] - 1)) for a in range(3)]
            lower = [min(math.floor(clamped[a]), max(shape[a] - 2, 0)) for a in range(3)]
            fraction = [np.float32(clamped[a] - np.float32(lower[a])) for a in range(3)]
            for corner in range(8):
                index = [min(lower[a] + ((corner >> a) & 1), shape[a] - 1) for a in range(3)]
                factors = [
                    fraction[a] if (corner >> a) & 1 else np.float32(1) - fraction[a]
                    for a in range(3)
                ]
                weight = np.float32(np.float32(factors[0] * factors[1]) * factors[2])
                voxel = (index[2] * shape[1] + index[1]) * shape[0] + index[0]
                matrix[pixel, voxel] += float(np.float32(physical * weight))
    return matrix


class Projection:
    """One public workspace and caller-owned output buffers on the current stream."""

    def __init__(
        self, grid, geometry, values, pose, seed, samples, backend, torch, wp, shared=None
    ):
        from dpt.projection import ProjectionSpec, prepare_projection

        self.wp, self.torch = wp, torch
        self.checked = True
        self.stream = wp.stream_from_torch(torch.cuda.current_stream())
        options = dict(
            active_pose=False, active_volume=True, precision="float32", sampling_precision="float32"
        )
        if "backend" in ProjectionSpec.__dataclass_fields__:
            options["backend"] = backend
        elif backend != "warp":
            raise RuntimeError("this checkout has no ProjectionSpec backend option")
        with wp.ScopedStream(self.stream):
            if shared is None:
                self.field = wp.array(
                    np.asarray(values, dtype=np.float32).ravel(), dtype=wp.float32, device="cuda"
                )
                self.pose = wp.array(pose.packed(), dtype=wp.float64, device="cuda")
                self.gradient = wp.empty(grid.voxels, dtype=wp.float32, device="cuda")
            else:
                self.field, self.pose, self.gradient = shared.field, shared.pose, shared.gradient
            self.seed = wp.array(
                np.asarray(seed, dtype=np.float32).ravel(), dtype=wp.float32, device="cuda"
            )
            self.output = wp.empty(geometry.pixels, dtype=wp.float32, device="cuda")
        self.workspace = prepare_projection(
            grid, geometry, ProjectionSpec(samples, **options), device="cuda", stream=self.stream
        )

    def forward(self):
        from dpt.projection import project_optical_depth

        project_optical_depth(
            self.field,
            self.pose,
            workspace=self.workspace,
            out_L=self.output,
            validate=self.checked,
            stream=self.stream,
        )

    def vjp(self, *, accumulate=False):
        from dpt.projection import projection_vjp

        if not self.checked and not accumulate:
            with self.wp.ScopedStream(self.stream):
                self.gradient.zero_()
        projection_vjp(
            self.field,
            self.pose,
            workspace=self.workspace,
            cotangent=self.seed,
            out_mu=self.gradient,
            validate=self.checked,
            stream=self.stream,
            accumulate=accumulate or not self.checked,
            check_volume_gradient=self.checked,
        )

    def set_validation(self, mode):
        self.checked = mode == "checked"
        if not self.checked:
            self.workspace.validate_inputs(
                self.field, self.pose, cotangent=self.seed, stream=self.stream
            )

    def checkpoint(self, operation):
        if not self.checked:
            if operation != "forward":
                self.workspace.enqueue_volume_gradient_check(self.gradient, stream=self.stream)
            self.workspace.check_status()

    def iteration(self):
        self.forward()
        self.vjp()

    def host_results(self):
        return self.output.numpy(), self.gradient.numpy()


class PanelProjection:
    """Checked flat-panel calls sharing the field, pose and accumulated gradient."""

    def __init__(self, grid, geometry, values, pose, seed, samples, backend, torch, wp):
        self.views = []
        for index, panel in enumerate(geometry.views):
            self.views.append(
                Projection(
                    grid,
                    panel,
                    values,
                    pose,
                    seed[index],
                    samples,
                    backend,
                    torch,
                    wp,
                    shared=self.views[0] if self.views else None,
                )
            )

    def forward(self):
        for view in self.views:
            view.forward()

    def vjp(self):
        # Fix view order for both backends; the first call zeros the shared destination.
        for index, view in enumerate(reversed(self.views)):
            view.vjp(accumulate=index != 0)

    def iteration(self):
        self.forward()
        self.vjp()

    def set_validation(self, mode):
        for view in self.views:
            view.set_validation(mode)

    def checkpoint(self, operation):
        if not self.views[0].checked:
            if operation != "forward":
                view = self.views[0]
                view.workspace.enqueue_volume_gradient_check(view.gradient, stream=view.stream)
            for view in self.views:
                view.workspace.check_status()

    def host_results(self):
        return np.concatenate([view.output.numpy() for view in self.views]), self.views[
            0
        ].gradient.numpy()


def guard_cases():
    from dpt.geometry import DetectorGeometry, RayBundleGeometry, RigidTransform, compose_pose
    from dpt.volumes import GridSpec

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
    rays = RayBundleGeometry(
        (
            (-12.0, 0.17, 0.37),
            (12.0, 0.23, 0.48),
            (-0.7, -12.0, 0.29),
            (0.31, 0.41, -12.0),
            (-12.0, 30.0, 30.0),
        ),
        (
            (12.0, 0.81, 0.63),
            (-12.0, 0.89, 0.91),
            (0.2, 12.0, 0.86),
            (0.72, 0.83, 12.0),
            (12.0, 30.0, 30.0),
        ),
    )
    z, y, x = np.indices(grid.shape)
    values = np.float32(0.06 + 0.013 * x + 0.019 * y + 0.011 * z + 0.003 * x * y + 0.002 * y * z)
    pose = compose_pose(RigidTransform(), (0.07, -0.03, 0.02, 0.013, -0.021, 0.009))
    for samples in (17, 97, 1024):
        yield f"shifted_oriented_anisotropic_{samples}", grid, rays, rays, values, pose, samples
    grid = GridSpec((1, 1, 2), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((0.001, -1.0, 0.0),), ((0.001, 1.0, 0.0),))
    yield "fraction_0.001", grid, rays, rays, np.float32([0.0, 1.0]), RigidTransform(), 1
    for shape in ((1, 1, 1), (1, 3, 4), (2, 1, 3), (3, 2, 1)):
        grid = GridSpec(shape, (0.7, 1.1, 1.3), (-0.13, 0.21, -0.37))
        centre = grid.grid_to_object(tuple((n - 1) / 2 for n in shape[::-1]))
        source = (centre[0], centre[1] - 10.0, centre[2])
        endpoint = (centre[0], centre[1] + 10.0, centre[2])
        rays = RayBundleGeometry((source,), (endpoint,))
        values = np.float32(np.arange(grid.voxels) * 0.013 + 0.07)
        yield f"singleton_{shape}", grid, rays, rays, values, RigidTransform(), 97
    grid = GridSpec((3, 4, 5), (1.1, 0.8, 1.3), (-2.2, -1.2, -1.3))
    panel = DetectorGeometry(
        (0.0, -10.0, 0.0), (-1.5, 10.0, -1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (3, 4)
    )
    rays = RayBundleGeometry.from_views((panel,))
    values = np.float32(0.02 + np.arange(grid.voxels) * 0.0017)
    yield "flat_panel", grid, panel, rays, values, RigidTransform(), 97


def run_guards(backend, torch, wp):
    records = []
    for name, grid, geometry, rays, values, pose, samples in guard_cases():
        matrix = reference_matrix(grid, rays, pose, samples)
        seed = np.float32(np.resize([0.7, -0.3, 0.2, 0.6, 1.0], rays.rays))
        instance = Projection(grid, geometry, values, pose, seed, samples, backend, torch, wp)
        instance.iteration()
        actual, gradient = instance.output.numpy(), instance.gradient.numpy()
        expected, transpose = matrix @ values.ravel(), matrix.T @ seed
        lhs, rhs = (
            float(actual.astype(np.float64) @ seed),
            float(values.ravel().astype(np.float64) @ gradient),
        )
        adjoint = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
        forward_ok = bool(np.allclose(actual, expected, rtol=ORACLE_RTOL, atol=ORACLE_ATOL))
        vjp_ok = bool(np.allclose(gradient, transpose, rtol=ORACLE_RTOL, atol=ORACLE_ATOL))
        changed = np.float32(values * 0.7 + 0.013)
        with wp.ScopedStream(instance.stream):
            instance.field.assign(changed.ravel())
        instance.forward()
        mutation_ok = bool(
            np.allclose(
                instance.output.numpy(),
                matrix @ changed.ravel(),
                rtol=ORACLE_RTOL,
                atol=ORACLE_ATOL,
            )
        )
        records.append(
            dict(
                name=name,
                forward=error(actual, expected),
                vjp=error(gradient, transpose),
                adjoint_relative_error=adjoint,
                mutated_field_forward_passed=mutation_ok,
                passed=forward_ok and vjp_ok and adjoint < ORACLE_ADJOINT_LIMIT and mutation_ok,
            )
        )
        print(
            json.dumps(dict(guard=name, backend=backend, passed=records[-1]["passed"])), flush=True
        )
    return records


def measure(call, torch, warmup, repeats, checkpoint, label):
    stream = torch.cuda.current_stream()
    for index in range(warmup):
        call()
        checkpoint()
        print(json.dumps(dict(operation=label, warmup_completed=index + 1)), flush=True)
    stream.synchronize()
    wall, events = [], []
    started = utc()
    for index in range(repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        stream.synchronize()
        begin.record(stream)
        before = time.perf_counter()
        call()
        end.record(stream)
        end.synchronize()
        wall.append((time.perf_counter() - before) * 1000)
        events.append(begin.elapsed_time(end))
        # Check every completed result before another VJP can reset its destination.
        checkpoint()
        print(
            json.dumps(
                dict(
                    operation=label,
                    repetition=index + 1,
                    repetitions=repeats,
                    wall_ms=wall[-1],
                    cuda_event_ms=events[-1],
                )
            ),
            flush=True,
        )
    return dict(
        started_at_utc=started,
        finished_at_utc=utc(),
        wall_ms=wall,
        cuda_event_ms=events,
        wall_median_ms=float(np.median(wall)),
        cuda_event_median_ms=float(np.median(events)),
    )


def workload(args):
    from dpt.geometry import BatchedGeometry, ConeBeamIntrinsics
    from dpt.volumes import GridSpec

    if args.prepared:
        from benchmarks.fullscan_inputs import load_prepared

        prepared = load_prepared(args.prepared)
        if args.views > len(prepared.geometry.views):
            raise ValueError("requested more views than the prepared acquisition")
        indices = np.linspace(
            0, len(prepared.geometry.views), args.views, endpoint=False, dtype=int
        )
        geometry = BatchedGeometry(tuple(prepared.geometry.views[int(i)] for i in indices))
        grid, values = prepared.volume.grid, np.asarray(prepared.volume.values)
        provenance = dict(
            kind="prepared_real_ct",
            path=str(args.prepared.resolve()),
            metadata_sha256=sha256(args.prepared / "metadata.json"),
            view_indices=indices.tolist(),
            source=prepared.metadata,
        )
        seed_metadata = json.loads(args.cotangent.with_suffix(".json").read_text())
        if sha256(args.cotangent) != seed_metadata["sha256"]:
            raise ValueError("saved cotangent checksum mismatch")
        saved_seed = np.load(args.cotangent, mmap_mode="r")
        if saved_seed.shape != prepared.geometry.shape or saved_seed.dtype != np.float32:
            raise ValueError("saved cotangent must match the prepared FP32 acquisition")
        if (
            hashlib.sha256(memoryview(saved_seed).cast("B")).hexdigest()
            != seed_metadata["raw_sha256"]
        ):
            raise ValueError("saved cotangent raw-value checksum mismatch")
        seed = np.ascontiguousarray(saved_seed[indices])
        provenance.update(
            cotangent_file=str(args.cotangent.resolve()),
            cotangent_metadata=seed_metadata,
            cotangent_subset_renormalised=False,
        )
    else:
        grid = GridSpec((args.size,) * 3, (1.0, 1.0, 1.0), (-(args.size - 1) / 2,) * 3)
        rng = np.random.default_rng(args.seed)
        values = np.float32(0.005 + 0.025 * rng.random(grid.shape))
        geometry = ConeBeamIntrinsics(
            4.0 * args.size,
            8.0 * args.size,
            (args.detector,) * 2,
            (2.2 * args.size / args.detector,) * 2,
        ).batch(tuple(2 * np.pi * np.arange(args.views) / args.views))
        provenance = dict(kind="seeded_nonconstant_synthetic_smoke", seed=args.seed)
        row, column = np.indices(geometry.shape[1:])
        angle = 2 * np.pi * np.arange(args.views) / args.views
        seed = (
            1.0
            + 0.2 * np.sin(angle)[:, None, None]
            + 0.1 * (row[None] + 0.5) / geometry.shape[1]
            + 0.07 * (column[None] + 0.5) / geometry.shape[2]
        )
        seed = np.float32(seed / np.sum(np.abs(seed), dtype=np.float64))
    provenance.update(
        grid=asdict(grid),
        image_shape=geometry.shape,
        samples_per_ray=args.samples,
        field_raw_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
        cotangent_raw_sha256=hashlib.sha256(seed.tobytes()).hexdigest(),
        field_range=[float(values.min()), float(values.max())],
    )
    return grid, geometry, values, seed, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument(
        "--cotangent", type=Path, help="Saved fullscan cotangent; required with --prepared"
    )
    parser.add_argument("--representation", choices=("panels", "device_bundle"), default="panels")
    parser.add_argument("--backends", nargs="+", choices=("warp", "cuda_texture"), default=["warp"])
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--detector", type=int, default=128)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--guards-only", action="store_true")
    parser.add_argument("--validation", choices=("checked", "unchecked", "both"), default="checked")
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=("forward", "vjp", "iteration"),
        default=["forward", "vjp", "iteration"],
    )
    args = parser.parse_args()
    if min(args.size, args.detector, args.views, args.samples, args.repeats) < 1 or args.warmup < 0:
        parser.error(
            "sizes, views, samples and repeats must be positive; warmup must be nonnegative"
        )
    if args.prepared is not None and args.cotangent is None:
        parser.error("--prepared requires its saved --cotangent")
    if len(args.backends) != len(set(args.backends)):
        parser.error("each backend may be listed only once")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import warp as wp

    import dpt
    from dpt.geometry import RigidTransform
    from dpt.geometry_device import prepare_geometry

    if not Path(dpt.__file__).resolve().is_relative_to(ROOT / "python"):
        raise RuntimeError("DPT was imported from another checkout")
    wp.init()
    result = dict(
        started_at_utc=utc(),
        harness_sha256=sha256(__file__),
        argv=sys.argv,
        source_head=command("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        source_status=command("git", "-C", str(ROOT), "status", "--short"),
        source_files={
            str(p.relative_to(ROOT)): sha256(p)
            for p in sorted((ROOT / "python/dpt").rglob("*"))
            if p.suffix in (".py", ".cu", ".h", ".cuh", ".cpp")
        },
        environment=dict(
            python=platform.python_version(),
            platform=platform.platform(),
            torch=torch.__version__,
            warp=importlib.metadata.version("warp-lang"),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
            nvidia_smi_before=command("nvidia-smi"),
        ),
        tolerance_sources=dict(
            oracle="tests/python/warp_cpu/test_projection_fp32.py",
            oracle_sha256=sha256(ROOT / "tests/python/warp_cpu/test_projection_fp32.py"),
            oracle_rtol=ORACLE_RTOL,
            oracle_atol=ORACLE_ATOL,
            oracle_adjoint_limit=ORACLE_ADJOINT_LIMIT,
            fullscan_sha256=sha256(ROOT / "benchmarks/config/fullscan.json"),
        ),
        timing_boundary=(
            "Completed public calls on resident inputs and caller-owned outputs. Checked mode "
            "includes validation and finite checks. Unchecked mode validates inputs before the "
            "timing series and checks status plus gradient finiteness after every repetition, "
            "outside timing. Both modes include VJP destination zero. Preparation, uploads and "
            "host downloads excluded. CUDA events cover the operation including stream idle "
            "gaps between launches; they are not a sum of isolated kernel durations."
        ),
        geometry_representation=args.representation,
        measurements={},
        guards={},
        comparisons={},
    )

    def save():
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )

    outputs = {}
    try:
        for backend in args.backends:
            result["guards"][backend] = run_guards(backend, torch, wp)
            save()
            if not all(row["passed"] for row in result["guards"][backend]):
                raise AssertionError(f"{backend} independent guards failed")
        if not args.guards_only:
            grid, geometry, values, seed, result["input"] = workload(args)
            stream = wp.stream_from_torch(torch.cuda.current_stream())
            if args.representation == "device_bundle":
                geometry = prepare_geometry(geometry, device="cuda", stream=stream)
            constructor = PanelProjection if args.representation == "panels" else Projection
            for backend in args.backends:
                print(
                    json.dumps(
                        dict(backend=backend, stage="first checked iteration", views=args.views)
                    ),
                    flush=True,
                )
                before = time.perf_counter()
                instance = constructor(
                    grid,
                    geometry,
                    values,
                    RigidTransform(),
                    seed,
                    args.samples,
                    backend,
                    torch,
                    wp,
                )
                instance.iteration()
                torch.cuda.synchronize()
                result["measurements"][backend] = dict(
                    first_completed_iteration_seconds_including_preparation=time.perf_counter()
                    - before
                )
                save()
                modes = (
                    ("unchecked", "checked") if args.validation == "both" else (args.validation,)
                )
                for mode in modes:
                    instance.set_validation(mode)
                    result["measurements"][backend][mode] = {}
                    for name in args.operations:
                        measured = measure(
                            getattr(instance, name),
                            torch,
                            args.warmup,
                            args.repeats,
                            lambda name=name, instance=instance: instance.checkpoint(name),
                            f"{backend}/{mode}/{name}",
                        )
                        result["measurements"][backend][mode][name] = measured
                        save()
                        print(
                            json.dumps(
                                dict(backend=backend, validation=mode, operation=name, **measured)
                            ),
                            flush=True,
                        )
                # Save corresponding forward and VJP results even for a selected-operation run.
                instance.iteration()
                instance.checkpoint("iteration")
                image, gradient = instance.host_results()
                np.save(args.output / f"{backend}-image.npy", image)
                np.save(args.output / f"{backend}-gradient.npy", gradient)
                lhs = float(image.astype(np.float64) @ seed.ravel())
                rhs = float(values.ravel().astype(np.float64) @ gradient)
                discrepancy = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
                result["measurements"][backend]["adjoint_relative_error"] = discrepancy
                limits = json.loads((ROOT / "benchmarks/config/fullscan.json").read_text())[
                    "quality"
                ]
                if discrepancy > limits["volume_adjoint"]["relative_discrepancy_max"]:
                    raise AssertionError(f"{backend} adjoint discrepancy {discrepancy}")
                outputs[backend] = (image, gradient)
            if "warp" in outputs:
                for backend, (image, gradient) in outputs.items():
                    if backend == "warp":
                        continue
                    image_error = error(image, outputs["warp"][0])
                    gradient_error = error(gradient, outputs["warp"][1])
                    result["comparisons"][backend] = dict(
                        image=image_error, gradient=gradient_error
                    )
                    np.testing.assert_allclose(
                        image, outputs["warp"][0], rtol=ORACLE_RTOL, atol=ORACLE_ATOL
                    )
                    if (
                        gradient_error["relative_l2"]
                        > limits["field_vjp_vs_baseline"]["relative_l2_max"]
                    ):
                        raise AssertionError(f"{backend} volume VJP differs from Warp")
        result["completed"] = True
    except Exception as exc:
        result["failure"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        result["finished_at_utc"] = utc()
        result["environment"]["nvidia_smi_after"] = command("nvidia-smi")
        save()


if __name__ == "__main__":
    main()
