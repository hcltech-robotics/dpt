"""Physical ROI projection, finite view selection and agent-facing file contracts."""

import importlib.util
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dpt.contracts import ContractError
from dpt.geometry import ConeBeamIntrinsics, RigidTransform
from dpt.imaging import read_volume

_PATH = Path(__file__).resolve().parents[2] / "examples" / "agentic_view_planning.py"
_SPEC = importlib.util.spec_from_file_location("agentic_view_planning", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


def _nifti(tmp_path, name="ct.nii.gz", values=None, affine=None):
    nib = pytest.importorskip("nibabel")
    if values is None:
        values = np.zeros((11, 13, 15), dtype=np.int16)
    if affine is None:
        affine = np.array([[0, -1.2, 0, 100], [0.8, 0, 0, -200], [0, 0, 2.5, 300], [0, 0, 0, 1]])
    image = nib.Nifti1Image(values, affine)
    image.header.set_xyzt_units("mm")
    path = tmp_path / name
    nib.save(image, path)
    return path


def _args(path, *extra):
    return example.parser().parse_args(
        ["inspect", "--ct", str(path), "--roi-at-volume-centre", "--roi-radius-mm", "2", *extra]
    )


def test_detector_coverage_counts_edges_and_rejects_source_backside_and_beyond_panel():
    panel = ConeBeamIntrinsics(100, 200, (10, 10), (2, 2)).view(0)
    points = np.array(
        [
            [0, 0, 0],
            [5, 0, 0],
            [5.1, 0, 0],
            [0, -101, 0],
            [0, 101, 0],
            [0, -100, 0],
        ]
    )
    assert example.detector_coverage(points, panel) == {
        "in_detector_samples": 2,
        "sample_count": 6,
        "fraction": 1 / 3,
    }


def test_coverage_is_invariant_under_rigid_world_transform():
    panel = ConeBeamIntrinsics(100, 200, (20, 30), (2, 3)).view(0.41)
    points = example.sphere_points((3, 2, 1), 12, 256)
    pose = RigidTransform((0, -1, 0, 0, 0, -1, 1, 0, 0), (314, -25, 740))
    assert example.detector_coverage(points, panel) == example.detector_coverage(
        example.transform_points(points, pose), example.transform_panel(panel, pose)
    )


def test_sphere_surface_samples_are_deterministic_and_have_declared_radius():
    points = example.sphere_points((7, -4, 10), 35, 256)
    np.testing.assert_allclose(np.linalg.norm(points - [7, -4, 10], axis=1), 35)
    np.testing.assert_array_equal(points, example.sphere_points((7, -4, 10), 35, 256))


def test_seven_angle_selection_matches_recorded_directional_criterion(tmp_path):
    result, *_ = example.evaluate(_args(_nifti(tmp_path)))
    assert result["selection"]["angles_degrees"] == [-60, 0, 60]
    assert result["selection"]["determinant"] == pytest.approx(6.75)
    assert result["selection"]["minimum_eigenvalue"] == pytest.approx(1.5)
    assert result["search"]["evaluated_subsets"] == 35
    assert all(row["roi_coverage"]["fraction"] == 1 for row in result["candidates"])


def test_selected_subset_matches_independent_planar_closed_form_oracle(tmp_path):
    angles = [-71, -23, 7, 32, 84]
    args = _args(_nifti(tmp_path), "--angles-degrees", *(str(a) for a in angles))
    result, *_ = example.evaluate(args)
    scores = []
    for chosen in itertools.combinations(angles, 3):
        # In the axial plane det(sum(I-dd^T)) = n * sum sin²(pairwise angle).
        determinant = 3 * sum(
            math.sin(math.radians(a - b)) ** 2 for a, b in itertools.combinations(chosen, 2)
        )
        scores.append((determinant, chosen))
    expected, chosen = max(scores)
    assert result["selection"]["angles_degrees"] == list(chosen)
    assert result["selection"]["determinant"] == pytest.approx(expected)


def test_coverage_threshold_excludes_clipped_candidates_and_reports_empty_search(tmp_path):
    args = _args(_nifti(tmp_path), "--detector-shape", "1", "1", "--pixel-spacing-mm", "0.1", "0.1")
    result, *_ = example.evaluate(args)
    assert result["status"] == "no_feasible_set"
    assert result["selection"] is None
    assert result["search"]["eligible_count"] == 0
    assert result["search"]["evaluated_subsets"] == 0


def test_oblique_native_mask_samples_use_world_affine_and_label(tmp_path):
    ct_path = _nifti(tmp_path)
    values = np.zeros((11, 13, 15), dtype=np.int16)
    values[2:9, 3:10, 4:11] = 2
    values[0, 0, 0] = 1
    mask_path = _nifti(tmp_path, "mask.nii.gz", values)
    volume = read_volume(ct_path, units="HU")
    points, description = example.region_points(
        volume,
        centre=None,
        radius_mm=None,
        mask_path=mask_path,
        mask_label=2,
        count=16,
    )
    assert len(points) == 16
    assert description["foreground_voxels"] == 343
    assert description["label"] == 2
    grid_indices = np.array([volume.grid.object_to_grid(tuple(point)) for point in points])
    np.testing.assert_allclose(grid_indices, np.round(grid_indices), atol=1e-6)
    assert np.all((grid_indices >= [2, 3, 4]) & (grid_indices <= [8, 9, 10]))
    assert len(description["source_sha256"][0]) == 64


def test_mask_grid_mismatch_and_empty_region_are_rejected(tmp_path):
    volume = read_volume(_nifti(tmp_path), units="HU")
    wrong = _nifti(tmp_path, "wrong.nii.gz", np.ones((11, 13, 15), dtype=np.int16), np.eye(4))
    with pytest.raises(ContractError, match="physical grid"):
        example.region_points(
            volume, centre=None, radius_mm=None, mask_path=wrong, mask_label=None, count=16
        )
    empty = _nifti(tmp_path, "empty.nii.gz")
    with pytest.raises(ContractError, match="empty"):
        example.region_points(
            volume, centre=None, radius_mm=None, mask_path=empty, mask_label=None, count=16
        )


def test_sphere_extending_outside_ct_is_rejected(tmp_path):
    with pytest.raises(ContractError, match="outside"):
        example.evaluate(_args(_nifti(tmp_path), "--roi-radius-mm", "100"))


@pytest.mark.parametrize(
    "angles,views,budget,match",
    [
        ([0, 0, 90], 2, 100, "unique"),
        ([0, 180], 2, 100, "unique"),
        ([0, float("nan")], 1, 100, "finite"),
        ([0, 90], 3, 100, "views"),
        (list(range(24)), 12, 100000, "subsets"),
        (list(range(25)), 2, 100000, "24"),
        ([0, 90], 1, 100001, "max_combinations"),
    ],
)
def test_search_validation_bounds_candidate_and_combination_work(angles, views, budget, match):
    with pytest.raises(ContractError, match=match):
        example.validate_search(angles, views, budget)


@pytest.mark.parametrize(
    "extra,match",
    [
        (["--min-coverage", "1.1"], "min_coverage"),
        (["--roi-samples", "4097"], "roi_samples"),
        (["--roi-radius-mm", "0"], "positive"),
        (["--sdd-mm", "200"], "source-detector"),
        (["--pixel-spacing-mm", "0", "1"], "positive"),
        (["--detector-shape", "2048", "2048"], "pixels"),
        (["--mask-label", "2"], "roi_mask"),
    ],
)
def test_invalid_settings_fail_before_cuda(tmp_path, extra, match):
    with pytest.raises(ContractError, match=match):
        example.evaluate(_args(_nifti(tmp_path), *extra))


def test_supplied_pose_moves_candidate_geometry_and_roi_together(tmp_path):
    path = _nifti(tmp_path)
    baseline, *_ = example.evaluate(_args(path))
    matrix = [[0, -1, 0, 300], [1, 0, 0, -25], [0, 0, 1, 50], [0, 0, 0, 1]]
    pose_path = tmp_path / "pose.json"
    pose_path.write_text(json.dumps(matrix))
    result, *_ = example.evaluate(_args(path, "--pose-json", str(pose_path)))
    assert result["selection"]["angles_degrees"] == baseline["selection"]["angles_degrees"]
    assert result["selection"]["determinant"] == pytest.approx(baseline["selection"]["determinant"])
    source = np.asarray(baseline["candidates"][0]["geometry_world_mm"]["source_mm"])
    expected = np.asarray(matrix)[:3, :3] @ source + np.asarray(matrix)[:3, 3]
    np.testing.assert_allclose(result["candidates"][0]["geometry_world_mm"]["source_mm"], expected)
    pose_path.write_text(
        json.dumps({"pose_object_to_world": result["settings"]["pose_object_to_world"]})
    )
    assert example.read_pose(pose_path) == example.read_pose(None).compose(
        example.read_pose(pose_path)
    )


def test_pose_reflection_is_rejected(tmp_path):
    path = tmp_path / "pose.json"
    path.write_text(json.dumps(np.diag([-1, 1, 1, 1]).tolist()))
    with pytest.raises(ContractError):
        example.read_pose(path)


def test_cli_plan_emits_json_and_preserves_existing_output(tmp_path):
    ct = _nifti(tmp_path)
    output = tmp_path / "run"
    command = [
        sys.executable,
        str(_PATH),
        "plan",
        "--ct",
        str(ct),
        "--roi-at-volume-centre",
        "--roi-radius-mm",
        "2",
        "--output",
        str(output),
    ]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    completed = subprocess.run(command, text=True, capture_output=True, env=environment, check=True)
    response = json.loads(completed.stdout)
    assert response["status"] == "planned"
    assert json.loads((output / "plan.json").read_text()) == response
    points = np.load(output / "roi_points_ras_mm.npy", allow_pickle=False)
    assert points.shape == (256, 3)
    before = (output / "plan.json").read_bytes()
    rerun = subprocess.run(command, text=True, capture_output=True, env=environment)
    assert rerun.returncode == 2
    assert json.loads(rerun.stdout)["status"] == "error"
    assert (output / "plan.json").read_bytes() == before


def test_cli_inspect_is_json_only_and_writes_no_outputs(tmp_path):
    ct = _nifti(tmp_path)
    before = set(tmp_path.iterdir())
    completed = subprocess.run(
        [
            sys.executable,
            str(_PATH),
            "inspect",
            "--ct",
            str(ct),
            "--roi-at-volume-centre",
            "--roi-radius-mm",
            "2",
        ],
        text=True,
        capture_output=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        check=True,
    )
    assert json.loads(completed.stdout)["search"]["candidate_count"] == 7
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize(
    "flags,match",
    [
        (["--device", "cpu"], "CUDA|cuda"),
        (["--samples-per-ray", "0"], "samples_per_ray"),
        (["--samples-per-ray", "4096", "--detector-shape", "1024", "1024"], "ray samples"),
    ],
)
def test_render_budget_failure_does_not_create_output(tmp_path, capsys, flags, match):
    output = tmp_path / "output"
    status = example.main(
        [
            "plan",
            "--ct",
            str(tmp_path / "unused.nii.gz"),
            "--roi-at-volume-centre",
            "--roi-radius-mm",
            "2",
            "--output",
            str(output),
            "--render",
            *flags,
        ]
    )
    response = json.loads(capsys.readouterr().out)
    assert status == 2
    assert response["status"] == "error"
    import re

    assert re.search(match, response["error"])
    assert not output.exists()


def test_dicom_ct_directory_is_read_into_ras_for_planning(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    series = tmp_path / "dicom"
    series.mkdir()
    uid = pydicom.uid.generate_uid()
    reference_uid = pydicom.uid.generate_uid()
    for z in range(8):
        meta = pydicom.dataset.FileMetaDataset()
        meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        data = pydicom.dataset.FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        data.SOPClassUID, data.SOPInstanceUID = (
            meta.MediaStorageSOPClassUID,
            meta.MediaStorageSOPInstanceUID,
        )
        data.SeriesInstanceUID, data.Modality = uid, "CT"
        data.FrameOfReferenceUID = reference_uid
        data.Rows, data.Columns = 12, 10
        data.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        data.ImagePositionPatient = [20, -40, 2 * z]
        data.PixelSpacing = [1.2, 0.8]
        data.SliceThickness = 2
        data.SamplesPerPixel, data.PhotometricInterpretation = 1, "MONOCHROME2"
        data.BitsAllocated, data.BitsStored, data.HighBit, data.PixelRepresentation = 16, 16, 15, 1
        data.RescaleSlope, data.RescaleIntercept, data.RescaleType = 1, -1000, "HU"
        data.PixelData = np.zeros((12, 10), dtype="<i2").tobytes()
        pydicom.dcmwrite(series / f"{z:03d}.dcm", data, enforce_file_format=True)
    result, *_ = example.evaluate(_args(series))
    assert result["status"] == "planned"
    assert result["ct"]["grid"]["origin_mm"] == (-20, 40, 0)
    assert result["ct"]["grid"]["shape"] == (8, 12, 10)
    assert len(result["ct"]["source_sha256"]) == 8
    assert result["selection"]["angles_degrees"] == [-60, 0, 60]
