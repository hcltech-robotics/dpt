"""Plan a finite set of complementary views around a supplied CT region.

Planning scores sampled region containment on the detector and directional
complementarity. Optional CUDA projections use the supplied CT and rigid pose.
The calling agent chooses the region, candidate angles, geometry and view budget.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dpt.contracts import ContractError, finite_scalar, integer
from dpt.geometry import BatchedGeometry, ConeBeamIntrinsics, DetectorGeometry, RigidTransform
from dpt.imaging import read_volume


def physical_centre(grid):
    nz, ny, nx = grid.shape
    return grid.grid_to_object(((nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2))


def read_pose(path: Path | None) -> RigidTransform:
    """Read an object-to-world rigid transform, retaining translation in mm."""
    if path is None:
        return RigidTransform()
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        value = value.get("pose_object_to_world", value)
    if isinstance(value, dict):
        if set(value) != {"rotation", "translation_mm"}:
            raise ContractError("pose needs rotation (nine entries) and translation_mm (three)")
        return RigidTransform(tuple(value["rotation"]), tuple(value["translation_mm"]))
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ContractError("pose matrix must be finite and 4 by 4")
    if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
        raise ContractError("pose matrix requires homogeneous row [0, 0, 0, 1]")
    return RigidTransform(tuple(matrix[:3, :3].ravel()), tuple(matrix[:3, 3]))


def transform_points(points, pose):
    return np.asarray(points) @ np.asarray(pose.rotation).reshape(3, 3).T + pose.translation_mm


def transform_panel(panel, pose):
    return DetectorGeometry(
        pose.point(panel.source_mm),
        pose.point(panel.origin_mm),
        pose.direction(panel.u),
        pose.direction(panel.v),
        panel.spacing_mm,
        panel.shape,
    )


def sphere_points(centre, radius_mm, count):
    """Deterministic approximately equal-area samples of the sphere surface."""
    integer(count, "roi_samples", minimum=16, maximum=4096)
    radius = finite_scalar(radius_mm, "roi_radius_mm", minimum=0)
    if radius == 0:
        raise ContractError("roi_radius_mm must be positive")
    centre = np.asarray(centre, dtype=float)
    if centre.shape != (3,) or not np.isfinite(centre).all():
        raise ContractError("ROI centre must contain three finite RAS coordinates in mm")
    k = np.arange(count)
    z = 1 - 2 * (k + 0.5) / count
    theta = k * math.pi * (3 - math.sqrt(5))
    xy = np.sqrt(1 - z * z)
    return centre + radius * np.column_stack((xy * np.cos(theta), xy * np.sin(theta), z))


def region_points(volume, *, centre, radius_mm, mask_path, mask_label, count):
    integer(count, "roi_samples", minimum=16, maximum=4096)
    grid = volume.grid
    if mask_path is not None:
        if radius_mm is not None or centre is not None:
            raise ContractError("use a sphere or a mask for the ROI")
        if mask_path.is_dir():
            raise ContractError("roi_mask must be a NIfTI file")
        mask = read_volume(mask_path, units="dimensionless")
        if mask.grid.shape != grid.shape or any(
            not np.allclose(getattr(mask.grid, key), getattr(grid, key), rtol=0, atol=1e-5)
            for key in ("spacing_mm", "origin_mm", "orientation")
        ):
            raise ContractError("ROI mask must share the CT's native physical grid")
        if np.any(mask.values < 0):
            raise ContractError("ROI mask values must be nonnegative")
        if mask_label is not None:
            integer(mask_label, "mask_label", minimum=1)
        foreground = mask.values > 0 if mask_label is None else mask.values == mask_label
        indices = np.flatnonzero(foreground)
        if not len(indices):
            raise ContractError("ROI mask selection is empty")
        total = len(indices)
        # Even ranks in x-fast storage give reproducible, bounded sampling.
        ranks = np.linspace(0, total - 1, min(count, total), dtype=np.int64)
        z, y, x = np.unravel_index(indices[ranks], grid.shape)
        points = (np.column_stack((x, y, z)) * np.asarray(grid.spacing_mm)) @ np.asarray(
            grid.orientation
        ).reshape(3, 3).T + grid.origin_mm
        description = {
            "kind": "mask",
            "mask": str(mask_path.resolve()),
            "source_sha256": list(mask.metadata.source_sha256),
            "label": mask_label,
            "foreground_voxels": total,
            "sampling": "evenly spaced ranks of foreground voxel centres in x-fast storage",
        }
    else:
        if mask_label is not None:
            raise ContractError("mask_label requires roi_mask")
        if centre is None or radius_mm is None:
            raise ContractError("a sphere requires an explicit centre and radius")
        points = sphere_points(centre, radius_mm, count)
        description = {
            "kind": "sphere",
            "centre_ras_mm": list(centre),
            "radius_mm": float(radius_mm),
            "sampling": "Fibonacci sphere surface",
        }
    indices = (points - grid.origin_mm) @ np.asarray(grid.orientation).reshape(3, 3)
    indices /= grid.spacing_mm
    low, high = grid.support
    if not np.all((indices >= np.asarray(low) - 1e-7) & (indices <= np.asarray(high) + 1e-7)):
        raise ContractError("ROI sample points extend outside the CT's physical support")
    description.update(sample_count=len(points), coordinate_frame="CT RAS millimetres")
    return points, description


def detector_coverage(points, panel):
    """Count ROI samples between source/panel whose projected points hit the panel."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ContractError("ROI points must have shape (N,3) with N positive")
    if not np.isfinite(points).all():
        raise ContractError("ROI points must be finite")
    source, origin = np.asarray(panel.source_mm), np.asarray(panel.origin_mm)
    u, v = np.asarray(panel.u), np.asarray(panel.v)
    normal = np.cross(u, v)
    rays = points - source
    fraction = (rays @ normal) / ((origin - source) @ normal)
    between = (fraction > 0) & (fraction <= 1)
    factor = np.zeros(len(points))
    np.divide(1, fraction, out=factor, where=between)
    hits = source + rays * factor[:, None] - origin
    columns = (hits @ u) / panel.spacing_mm[0]
    rows = (hits @ v) / panel.spacing_mm[1]
    contained = (
        between
        & (columns >= -0.5 - 1e-9)
        & (columns <= panel.shape[1] - 0.5 + 1e-9)
        & (rows >= -0.5 - 1e-9)
        & (rows <= panel.shape[0] - 0.5 + 1e-9)
    )
    count = int(contained.sum())
    return {
        "in_detector_samples": count,
        "sample_count": len(points),
        "fraction": count / len(points),
    }


def validate_search(angles, views, max_combinations):
    if not 1 <= len(angles) <= 24:
        raise ContractError("provide 1 to 24 candidate angles")
    angles = tuple(finite_scalar(a, "candidate angle") for a in angles)
    if any(not -180 <= a < 180 for a in angles) or len(set(angles)) != len(angles):
        raise ContractError("candidate angles must be unique and in [-180, 180) degrees")
    integer(views, "views", minimum=1, maximum=len(angles))
    integer(max_combinations, "max_combinations", minimum=1, maximum=100000)
    count = math.comb(len(angles), views)
    if count > max_combinations:
        raise ContractError(f"{count} subsets exceed max_combinations={max_combinations}")
    return tuple(sorted(angles)), count


def select_views(records, views, min_coverage):
    """Exhaustive bounded selection with explicit coverage and deterministic ties."""
    eligible = [row for row in records if row["roi_coverage"]["fraction"] >= min_coverage]
    best, best_key, evaluated = None, None, 0
    for chosen in itertools.combinations(eligible, views):
        matrix = sum(np.asarray(row["direction_matrix"]) for row in chosen)
        determinant = max(0.0, float(np.linalg.det(matrix)))
        angles = [row["angle_degrees"] for row in chosen]
        coverage = min(row["roi_coverage"]["fraction"] for row in chosen)
        key = (-round(determinant, 10), -coverage, sum(abs(a) for a in angles), tuple(angles))
        evaluated += 1
        if best_key is None or key < best_key:
            best_key = key
            best = {
                "candidate_indices": [row["index"] for row in chosen],
                "angles_degrees": angles,
                "direction_matrix": matrix.tolist(),
                "determinant": determinant,
                "minimum_eigenvalue": float(np.linalg.eigvalsh(matrix)[0]),
                "minimum_roi_coverage_fraction": coverage,
                "order": "ascending candidate angle",
            }
    return best, len(eligible), evaluated


def evaluate(args):
    angles, combinations = validate_search(args.angles_degrees, args.views, args.max_combinations)
    minimum = finite_scalar(args.min_coverage, "min_coverage", minimum=0)
    if minimum > 1:
        raise ContractError("min_coverage must be between zero and one")
    integer(args.roi_samples, "roi_samples", minimum=16, maximum=4096)
    for count in args.detector_shape:
        integer(count, "detector dimension", minimum=1, maximum=2048)
    if math.prod(args.detector_shape) > 1048576:
        raise ContractError("detector exceeds 1048576 pixels per view")
    volume = read_volume(args.ct, units=args.units, series_uid=args.series_uid)
    centre = physical_centre(volume.grid)
    roi_centre = centre if args.roi_at_volume_centre else args.roi_centre_mm
    points, roi = region_points(
        volume,
        centre=roi_centre,
        radius_mm=args.roi_radius_mm,
        mask_path=args.roi_mask,
        mask_label=args.mask_label,
        count=args.roi_samples,
    )
    pose = read_pose(args.pose_json)
    isocentre = tuple(args.isocentre_mm) if args.isocentre_mm is not None else centre
    intrinsics = ConeBeamIntrinsics(
        args.sod_mm,
        args.sdd_mm,
        tuple(args.detector_shape),
        tuple(args.pixel_spacing_mm),
        centre_mm=isocentre,
    )
    panels = [transform_panel(intrinsics.view(math.radians(a)), pose) for a in angles]
    world_points = transform_points(points, pose)
    world_isocentre = np.asarray(pose.point(isocentre))
    records = []
    for index, (angle, panel) in enumerate(zip(angles, panels, strict=True)):
        direction = np.asarray(panel.source_mm) - world_isocentre
        direction /= np.linalg.norm(direction)
        coverage = detector_coverage(world_points, panel)
        records.append(
            {
                "index": index,
                "angle_degrees": angle,
                "geometry_world_mm": asdict(panel),
                "source_direction_world": direction.tolist(),
                "direction_matrix": (np.eye(3) - np.outer(direction, direction)).tolist(),
                "roi_coverage": coverage,
                "eligible": coverage["fraction"] >= minimum,
            }
        )
    selected, eligible_count, evaluated = select_views(records, args.views, minimum)
    result = {
        "schema_version": 1,
        "status": "planned" if selected else "no_feasible_set",
        "ct": {
            "path": str(args.ct.resolve()),
            "source_sha256": list(volume.metadata.source_sha256),
            "value_units": volume.metadata.value_units,
            "grid": asdict(volume.grid),
        },
        "roi": roi,
        "settings": {
            "intrinsics_ct_ras": asdict(intrinsics),
            "pose_object_to_world": asdict(pose),
            "angle_convention": "source = isocentre + SOD * (sin(angle), -cos(angle), 0) in CT RAS",
            "candidate_frame": "CT RAS; rigid pose transforms both panels and ROI into world",
            "geometry_provenance": "declared example or user-supplied circular geometry",
            "angles_degrees": list(angles),
            "views": args.views,
            "minimum_roi_coverage_fraction": minimum,
            "max_combinations": args.max_combinations,
        },
        "objective": {
            "coverage": (
                "fraction of sampled ROI points between source and detector "
                "projecting inside detector edges"
            ),
            "selection": (
                "maximise det(sum(I - d d^T)) over eligible subsets; "
                "d is unit isocentre-to-source direction"
            ),
            "quantity": "dimensionless directional complementarity surrogate",
            "ties": (
                "determinant rounded to 10 decimals, then higher minimum coverage, "
                "smaller sum of absolute angles, ascending angles"
            ),
        },
        "search": {
            "candidate_count": len(records),
            "eligible_count": eligible_count,
            "possible_subsets": combinations,
            "evaluated_subsets": evaluated,
        },
        "candidates": records,
        "selection": selected,
    }
    return result, volume, points, panels, pose


def validate_render_budget(args):
    if not args.device.startswith("cuda"):
        raise ContractError("rendering requires --device cuda:N")
    integer(args.samples_per_ray, "samples_per_ray", minimum=1, maximum=4096)
    pixels = math.prod(args.detector_shape) * args.views
    if pixels * args.samples_per_ray > 2**31:
        raise ContractError(
            "rendering exceeds 2^31 ray samples; reduce views, panel shape or samples"
        )


def render_selected(args, result, volume, panels, pose):
    import torch

    from dpt import project
    from dpt.imaging import ct_to_attenuation
    from dpt.rendering import save_gray_png

    if volume.metadata.value_units == "HU":
        attenuation = ct_to_attenuation(volume, energy_kev=60)
        model = "60 keV water-scaled HU; mu_water = 0.02059 mm^-1"
    elif volume.metadata.value_units == "mm^-1" and np.all(volume.values >= 0):
        attenuation, model = volume, "supplied nonnegative attenuation in mm^-1"
    else:
        raise ContractError("rendering needs HU or nonnegative mm^-1 values")
    indices = result["selection"]["candidate_indices"]
    geometry = BatchedGeometry(tuple(panels[i] for i in indices))
    packed = torch.tensor(pose.packed(), dtype=torch.float64, device=args.device)
    images = project(
        attenuation,
        geometry,
        packed_pose=packed,
        device=args.device,
        precision="fp32",
        samples_per_ray=args.samples_per_ray,
    )
    depth = images.detach().cpu().numpy()
    if not np.isfinite(depth).all():
        raise ContractError("selected-view projections contain non-finite values")
    np.save(args.output / "optical_depth.npy", depth, allow_pickle=False)
    high = max(float(depth.max()), float(np.finfo(np.float32).eps))
    paths = []
    for position, index in enumerate(indices):
        name = f"selected-{position:02d}-candidate-{index:02d}.png"
        save_gray_png(args.output / name, np.flipud(depth[position]), low=0, high=high)
        paths.append(name)
    result["projections"] = {
        "optical_depth": "optical_depth.npy",
        "png_paths": paths,
        "candidate_indices": indices,
        "samples_per_ray": args.samples_per_ray,
        "precision": "fp32",
        "attenuation_model": model,
        "display_window_optical_depth": [0, high],
        "png_rows": "superior CT direction at top",
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    for command in ("inspect", "plan"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--ct", type=Path, required=True)
        sub.add_argument("--units", choices=("HU", "mm^-1"), default="HU")
        sub.add_argument("--series-uid")
        roi = sub.add_mutually_exclusive_group(required=True)
        roi.add_argument("--roi-centre-mm", type=float, nargs=3)
        roi.add_argument("--roi-at-volume-centre", action="store_true")
        roi.add_argument("--roi-mask", type=Path)
        sub.add_argument("--roi-radius-mm", type=float)
        sub.add_argument("--mask-label", type=int)
        sub.add_argument("--roi-samples", type=int, default=256)
        sub.add_argument(
            "--angles-degrees", type=float, nargs="+", default=[-90, -60, -30, 0, 30, 60, 90]
        )
        sub.add_argument("--views", type=int, default=3)
        sub.add_argument("--min-coverage", type=float, default=1)
        sub.add_argument("--max-combinations", type=int, default=100000)
        sub.add_argument("--sod-mm", type=float, default=353.72339)
        sub.add_argument("--sdd-mm", type=float, default=702)
        sub.add_argument(
            "--detector-shape", type=int, nargs=2, default=[256, 256], metavar=("ROWS", "COLS")
        )
        sub.add_argument(
            "--pixel-spacing-mm", type=float, nargs=2, default=[1.2, 1.2], metavar=("COL", "ROW")
        )
        sub.add_argument("--isocentre-mm", type=float, nargs=3)
        sub.add_argument("--pose-json", type=Path)
        if command == "plan":
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--render", action="store_true")
            sub.add_argument("--samples-per-ray", type=int, default=768)
            sub.add_argument("--device", default="cuda:0")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.command == "plan" and args.output.exists():
                raise ContractError("output directory already exists; choose a new directory")
            if args.command == "plan" and args.render:
                validate_render_budget(args)
            result, volume, points, panels, pose = evaluate(args)
            if args.command == "plan":
                args.output.mkdir(parents=True, exist_ok=False)
                np.save(args.output / "roi_points_ras_mm.npy", points, allow_pickle=False)
                result["roi"]["sample_points_file"] = "roi_points_ras_mm.npy"
                if args.render and result["selection"] is not None:
                    render_selected(args, result, volume, panels, pose)
                result["output"] = str(args.output.resolve())
                (args.output / "plan.json").write_text(
                    json.dumps(result, indent=2, allow_nan=False) + "\n"
                )
        print(json.dumps(result, allow_nan=False))
        return 0
    except (ContractError, FileNotFoundError, FileExistsError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
