"""Measure checked CUDA multiview forward calls through the public Torch API.

Run from the source checkout with ``PYTHONPATH=python:. python -m
benchmarks.cuda_multiview --output /tmp/multiview.json``. Direct script execution
uses the same source environment. The default synthetic workload is a smoke
case. For full CT, supply --prepared, --cotangent, --views 360 and --samples 1024.
An optional --fp64-images file enables the existing fullscan image error gates.

Resident input upload and preparation are outside timing. Each timed functional
Projector call includes input validation, fresh volume packing, output allocation,
diagnostic delivery and stream completion. Results include every repetition;
this forward measurement does not establish backward or iteration performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.cuda_texture import workload

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(UTC).isoformat()


def emit(**data):
    print(json.dumps(data), flush=True)


def save(output, result):
    """Update only the result file opened exclusively by this invocation."""
    output.seek(0)
    json.dump(result, output, indent=2)
    output.write("\n")
    output.truncate()
    output.flush()


def assert_exact(actual, expected, label):
    import torch

    assert actual.dtype == expected.dtype == torch.float32, label
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), label
    assert bool(torch.isfinite(actual).all()), label


def projectors(grid, geometry, samples):
    from dpt.torch import Projector

    return tuple(
        Projector(
            grid=grid,
            geometry=geometry,
            precision="fp32",
            samples_per_ray=samples,
            backend=backend,
        )
        for backend in ("cuda_texture", "warp")
    )


def guard_cases():
    """Execute nonconstant, mixed-pose and invalid-input cases on the GPU."""
    import torch

    from dpt.contracts import ContractError, NumericalError
    from dpt.geometry import RigidTransform, compose_pose

    args = SimpleNamespace(prepared=None, views=3, samples=97, size=7, detector=17, seed=67123)
    grid, geometry, values, _, provenance = workload(args)
    native, reference = projectors(grid, geometry, args.samples)
    field = torch.tensor(values, device="cuda")
    poses = torch.tensor(
        [
            compose_pose(RigidTransform(), step).packed()
            for step in (
                (0, 0, 0, 0, 0, 0),
                (0.17, -0.09, 0.04, 0.021, -0.017, 0.013),
                (-0.11, 0.13, -0.07, -0.009, 0.019, 0.025),
            )
        ],
        dtype=torch.float64,
        device="cuda",
    )
    initial = native(field, packed_pose=poses)
    assert_exact(initial, reference(field, packed_pose=poses), "mixed per-view poses")
    field_pointer, pose_pointer = field.data_ptr(), poses.data_ptr()
    field.mul_(0.7).add_(0.011)
    changed = native(field, packed_pose=poses)
    assert_exact(changed, reference(field, packed_pose=poses), "same-pointer field mutation")
    assert not torch.equal(initial, changed), "field mutation must change output"
    poses[1, 9].add_(0.13)
    moved = native(field, packed_pose=poses)
    assert_exact(moved, reference(field, packed_pose=poses), "same-pointer pose mutation")
    assert not torch.equal(changed[1], moved[1]), "pose mutation must change its view"
    assert_exact(changed[0], moved[0], "other pose must remain unchanged")
    assert field.data_ptr() == field_pointer and poses.data_ptr() == pose_pointer

    destination = torch.empty_like(moved)
    assert native(field, packed_pose=poses, out=destination) is destination
    assert_exact(destination, moved, "out= public path")
    invalid = poses.clone()
    invalid[1, 9] = 1.7e308
    try:
        native(field, packed_pose=invalid)
    except NumericalError as error:
        pose_error = str(error)
        assert "projection view 1" in pose_error, pose_error
    else:
        raise AssertionError("overflowing view 1 pose was not rejected")
    assert_exact(native(field, packed_pose=poses), moved, "pose failure recovery")

    original = field.flatten()[0].item()
    field.flatten()[0] = float("nan")
    try:
        native(field, packed_pose=poses)
    except (ContractError, NumericalError) as error:
        field_error = str(error)
    else:
        raise AssertionError("mutated nonfinite field was not rejected")
    field.flatten()[0] = original
    assert_exact(native(field, packed_pose=poses), moved, "field failure recovery")

    owner = torch.cuda.Stream()
    owner.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(owner):
        streamed = native(field, packed_pose=poses)
    torch.cuda.current_stream().wait_stream(owner)
    assert_exact(streamed, moved, "nondefault stream")
    return dict(
        input=provenance,
        mixed_poses_bitwise=True,
        same_pointer_field_and_pose_mutation_bitwise=True,
        out_bitwise=True,
        invalid_pose_error=pose_error,
        invalid_field_error=field_error,
        invalid_pose_and_field_recovery_bitwise=True,
        nondefault_stream_bitwise=True,
    )


def source_hashes():
    paths = [
        path
        for path in (ROOT / "python").rglob("*")
        if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".h", ".cpp")
    ]
    paths += [
        ROOT / "benchmarks" / name
        for name in (
            "cuda_multiview.py",
            "cuda_texture.py",
            "fullscan_inputs.py",
            "config/fullscan.json",
        )
    ]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def errors_vs_fp64(actual, path):
    expected = np.load(path, mmap_mode="r")
    if expected.shape != tuple(actual.shape) or expected.dtype != np.float64:
        raise ValueError("FP64 reference must match the complete projected image shape")
    maximum = sum_squared = norm_squared = 0.0
    count = 0
    # Check every pixel while bounding CPU temporary storage.
    for view in range(actual.shape[0]):
        a = actual[view].detach().cpu().numpy().astype(np.float64)
        b = np.asarray(expected[view])
        delta = a - b
        maximum = max(maximum, float(np.abs(delta).max()))
        sum_squared += float(np.sum(delta * delta))
        norm_squared += float(np.sum(b * b))
        count += delta.size
    return dict(
        max_abs=maximum,
        rmse=float(np.sqrt(sum_squared / count)),
        relative_l2=float(np.sqrt(sum_squared) / max(np.sqrt(norm_squared), 1e-300)),
        reference_path=str(path.resolve()),
        reference_sha256=sha256(path),
    )


def run(args, output, result):
    import torch
    import warp as wp

    import dpt
    from dpt.geometry import RigidTransform
    from dpt.torch import _projection_ops

    if not Path(dpt.__file__).resolve().is_relative_to(ROOT / "python"):
        raise RuntimeError("DPT was imported from another checkout")
    hashes = source_hashes()
    config = json.loads((ROOT / "benchmarks/config/fullscan.json").read_text())
    compiler = os.environ.get("DPT_NVCC") or shutil.which("nvcc")
    if compiler is None:
        raise RuntimeError("cuda_texture requires nvcc; set DPT_NVCC to the CUDA compiler")
    result.update(
        source=str(ROOT),
        source_sha256=hashes,
        harness_sha256=sha256(__file__),
        torch=torch.__version__,
        warp=wp.__version__,
        gpu=torch.cuda.get_device_name(),
        device_properties=str(torch.cuda.get_device_properties(0)),
        compiler=subprocess.check_output([compiler, "--version"], text=True),
        timing_boundary=(
            "ordinary functional checked Projector with resident inputs; field/pose validation, "
            "fresh atlas pack, output allocation, diagnostic consumption and completed current "
            "stream included; preparation, warmup, comparison and file IO excluded"
        ),
        acceptance=config["quality"],
    )
    wp.init()
    with torch.no_grad():
        result["guards"] = guard_cases()
        save(output, result)
        emit(stage="public guards passed")
        grid, geometry, values, seed, provenance = workload(args)
        result["input"] = provenance
        del seed
        field = torch.tensor(np.asarray(values), dtype=torch.float32, device="cuda")
        poses = torch.tensor(
            [RigidTransform().packed()] * args.views, dtype=torch.float64, device="cuda"
        )
        field_pointer, pose_pointer = field.data_ptr(), poses.data_ptr()
        native, reference = projectors(grid, geometry, args.samples)
        expected = reference(field, packed_pose=poses)
        torch.cuda.synchronize()
        emit(stage="current Warp image produced", shape=list(expected.shape))
        actual = native(field, packed_pose=poses)
        assert_exact(actual, expected, "complete image versus current Warp")
        entry = _projection_ops.prepare(field, native._configuration, False, False)
        assert entry.forward_batch is not None
        result.update(
            native_library=entry.forward_batch.first.library._name,
            native_library_sha256=sha256(entry.forward_batch.first.library._name),
            batch_config_bytes=int(entry.forward_batch.configurations.size),
            atlas_bytes=entry.forward_batch.first.atlas.nbytes,
            complete_image_bitwise_equal_to_warp=True,
        )
        if args.fp64_images:
            metrics = errors_vs_fp64(actual, args.fp64_images)
            result["fp32_vs_fp64"] = metrics
            thresholds = config["quality"]["image_fp32_vs_fp64"]
            for name in ("max_abs", "rmse", "relative_l2"):
                assert metrics[name] <= thresholds[name], (name, metrics[name], thresholds[name])
        save(output, result)
        emit(stage="image guards passed", fp32_vs_fp64=result.get("fp32_vs_fp64"))

        result["rounds"] = []
        for round_index in range(args.rounds):
            warm = native(field, packed_pose=poses)
            assert_exact(warm, expected, "round warmup")
            series = dict(index=round_index, started_utc=utc(), samples=[])
            result["rounds"].append(series)
            for repeat in range(args.repeats):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                begin.record()
                started = time.perf_counter()
                actual = native(field, packed_pose=poses)
                end.record()
                end.synchronize()
                wall = (time.perf_counter() - started) * 1000
                elapsed = begin.elapsed_time(end)
                series["samples"].append(dict(index=repeat, wall_ms=wall, event_ms=elapsed))
            assert_exact(actual, expected, "timed round output")
            series["completed_utc"] = utc()
            save(output, result)
            emit(stage="timed round", **series)

        field.mul_(0.7).add_(0.011)
        changed = native(field, packed_pose=poses)
        assert_exact(changed, reference(field, packed_pose=poses), "post-timing field mutation")
        assert not torch.equal(changed, expected), "field mutation must change output"
        poses[1, 9].add_(0.13)
        moved = native(field, packed_pose=poses)
        assert_exact(moved, reference(field, packed_pose=poses), "post-timing pose mutation")
        assert not torch.equal(changed[1], moved[1]), "pose mutation must change its view"
        assert_exact(changed[0], moved[0], "unmodified pose view")
        assert field.data_ptr() == field_pointer and poses.data_ptr() == pose_pointer
        result["complete_image_after_field_and_pose_mutations_bitwise_equal_to_warp"] = True
        result["input_pointers_unchanged_by_in_place_mutations"] = True
        raw = [item for series in result["rounds"] for item in series["samples"]]
        result["summary"] = {
            name: dict(
                median=float(np.median([item[name] for item in raw])),
                p95=float(np.percentile([item[name] for item in raw], 95)),
                minimum=min(item[name] for item in raw),
                maximum=max(item[name] for item in raw),
            )
            for name in ("wall_ms", "event_ms")
        }
        result["timed_calls"] = len(raw)
        assert source_hashes() == hashes, "source changed during verification"
        result["completed_utc"] = utc()
        save(output, result)
        emit(
            stage="complete",
            output=str(args.output),
            summary=result["summary"],
            timed_calls=len(raw),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON result file")
    parser.add_argument("--prepared", type=Path, help="Verified prepared CT input directory")
    parser.add_argument("--cotangent", type=Path, help="Saved cotangent required with --prepared")
    parser.add_argument("--fp64-images", type=Path, help="FP64 images for the fixed fullscan gates")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--detector", type=int, default=128)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if min(args.size, args.detector, args.samples, args.rounds, args.repeats) < 1:
        parser.error("sizes, samples, rounds and repeats must be positive")
    if not 2 <= args.views <= 65535:
        parser.error("views must be between 2 and 65535 for a CUDA multiview batch")
    if args.samples > 2**31 - 1:
        parser.error("samples must be at most 2**31 - 1")
    if (args.prepared is None) != (args.cotangent is None):
        parser.error("--prepared and its saved --cotangent must be supplied together")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    # Exclusive creation also protects against another process creating this path.
    with args.output.open("x") as output:
        result = dict(started_utc=utc(), argv=sys.argv)
        try:
            run(args, output, result)
        except BaseException as error:
            result["failure"] = dict(type=type(error).__name__, message=str(error), utc=utc())
            save(output, result)
            raise


if __name__ == "__main__":
    main()
