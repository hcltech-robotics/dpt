"""Direct majorant flights, boundary ownership and replay on CPU and CUDA."""

# Warp kernel annotations are evaluated as runtime DSL expressions.
# pyright: reportInvalidTypeForm=false
import importlib
import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from dpt.contracts import NumericalError
from dpt.materials import Provenance
from dpt.transport import kernels
from dpt.transport.derivatives import TransportParameter, derivative_histories
from dpt.transport.energy_balance import trace_energy_balance
from dpt.transport.forward import prepare_transport, trace_histories
from dpt.transport.model import (
    HistoryStatus,
    IncompleteHistoryError,
    MaterialGrid,
    PlanarDetector,
    TransportSpec,
)
from dpt.transport.physical import AtomicScatteringTable, FluorescenceLine, PhysicalTransport
from dpt.transport.readout import TransportReadout, prepare_transport_readout, trace_readout
from dpt.transport.rng import HistoryBatch

wp: Any = importlib.import_module("warp")


@wp.kernel
def _flights(
    positions: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    depths: wp.array(dtype=wp.float64),
    majorants: wp.array(dtype=wp.float64),
    p: kernels.Parameters,  # pyright: ignore[reportGeneralTypeIssues]
    out_positions: wp.array(dtype=wp.vec3d),
    out_cells: wp.array(dtype=wp.vec3i),
    out_status: wp.array(dtype=wp.vec2i),
):
    index = wp.tid()
    result = kernels.woodcock_flight(
        positions[index], directions[index], depths[index], majorants[index], p
    )
    out_positions[index] = result.position
    out_cells[index] = result.cell
    out_status[index] = wp.vec2i(result.alive, result.status)


def _physical(majorant):
    atomic = AtomicScatteringTable(
        1,
        (0.0, 10.0),
        (1.0, 1.0),
        (0.0, 0.0),
        Provenance("constant analytic angular table", "0" * 64, "CC0-1.0", "Test fixture."),
    )
    return PhysicalTransport((atomic,), (0.0,), majorant_mm_inverse=(majorant,))


def _slab(device, *, count=4096, cells=64, depth=1.0, rate=0.3, majorant=2.0, events=128):
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, depth / cells), (cells, 1, 1), 1),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), 2.0),
        80.0,
        "analytic absorbing slab",
        max_crossings=1,
        max_events=events,
    )
    workspace = prepare_transport(
        spec,
        material_ids=wp.zeros(cells, dtype=wp.int32, device=device),
        absorption=wp.array([rate], dtype=wp.float64, device=device),
        scattering=wp.zeros(1, dtype=wp.float64, device=device),
        max_histories=count,
        physical=_physical(majorant),
        device=device,
    )
    inputs = (
        wp.array(np.tile((0.0, 0.0, -1.0), (count, 1)), dtype=wp.vec3d, device=device),
        wp.array(np.tile((0.0, 0.0, 1.0), (count, 1)), dtype=wp.vec3d, device=device),
        wp.ones(count, dtype=wp.float64, device=device),
        wp.ones(1, dtype=wp.float64, device=device),
    )
    outputs = {
        "out_" + name: wp.empty(count, dtype=dtype, device=device)
        for name, dtype in (
            ("pixel", wp.int32),
            ("score", wp.float64),
            ("energy", wp.float64),
            ("events", wp.int32),
            ("status", wp.int32),
        )
    }
    return workspace, inputs, outputs


def test_candidate_and_exit_distances_preserve_faces_and_extreme_majorants(warp_device):
    p = kernels.Parameters()
    p.origin = wp.vec3d(0.0, 0.0, 0.0)
    p.spacing = wp.vec3d(0.25, 0.5, 1.0)
    p.shape = wp.vec3i(8, 4, 2)
    positions = [(0.5, 0.5, 0.5)] * 7 + [(2.0, 0.5, 0.5)]
    diagonal = 1 / math.sqrt(3)
    directions = [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (diagonal, diagonal, diagonal),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
    ]
    depths = [0.5, 0.25, 10.0, 1.0, 1.0, 1.5, 1.0, 0.25]
    majorants = [1.0, 1.0, 1.0, 1.5e308, 1e-320, 1.0, 0.0, 1.0]
    count = len(depths)
    out_positions = wp.empty(count, dtype=wp.vec3d, device=warp_device)
    out_cells = wp.empty(count, dtype=wp.vec3i, device=warp_device)
    status = wp.empty(count, dtype=wp.vec2i, device=warp_device)
    wp.launch(
        _flights,
        dim=count,
        inputs=[
            wp.array(positions, dtype=wp.vec3d, device=warp_device),
            wp.array(directions, dtype=wp.vec3d, device=warp_device),
            wp.array(depths, dtype=wp.float64, device=warp_device),
            wp.array(majorants, dtype=wp.float64, device=warp_device),
            p,
            out_positions,
            out_cells,
            status,
        ],
        device=warp_device,
    )
    np.testing.assert_array_equal(status.numpy()[:, 0], [1, 1, 0, 1, 0, 0, 0, 1])
    np.testing.assert_array_equal(
        status.numpy()[:, 1], [0, 0, 0, 0, 0, 0, int(HistoryStatus.NUMERICAL_FAILURE), 0]
    )
    np.testing.assert_array_equal(
        out_positions.numpy(),
        [
            (1.0, 0.5, 0.5),
            (0.25, 0.5, 0.5),
            (2.0, 2.0, 2.0),
            (0.5, 0.5, 0.5),
            (2.0, 0.5, 0.5),
            (2.0, 0.5, 0.5),
            (0.5, 0.5, 0.5),
            (1.75, 0.5, 0.5),
        ],
    )
    np.testing.assert_array_equal(
        out_cells.numpy()[[0, 1, 3, 7]], [[4, 1, 0], [0, 1, 0], [2, 1, 0], [6, 1, 0]]
    )


def test_parallel_coordinates_survive_the_largest_finite_exit(warp_device):
    p = kernels.Parameters()
    p.origin = wp.vec3d(0.0, 0.0, 0.0)
    p.spacing = wp.vec3d(np.finfo(np.float64).max, 1.0, 1.0)
    p.shape = wp.vec3i(1, 1, 1)
    output = wp.empty(1, dtype=wp.vec3d, device=warp_device)
    cells = wp.empty(1, dtype=wp.vec3i, device=warp_device)
    status = wp.empty(1, dtype=wp.vec2i, device=warp_device)
    wp.launch(
        _flights,
        dim=1,
        inputs=[
            wp.array([(0.0, 0.5, 0.5)], dtype=wp.vec3d, device=warp_device),
            wp.array([(1.0, 0.0, 0.0)], dtype=wp.vec3d, device=warp_device),
            wp.array([1.0], dtype=wp.float64, device=warp_device),
            wp.array([1e-320], dtype=wp.float64, device=warp_device),
            p,
            output,
            cells,
            status,
        ],
        device=warp_device,
    )
    np.testing.assert_array_equal(output.numpy(), [(np.finfo(np.float64).max, 0.5, 0.5)])
    np.testing.assert_array_equal(status.numpy(), [[0, 0]])


@pytest.mark.parametrize("rate", [0.0, 0.3])
def test_many_cell_flights_do_not_spend_crossing_budget_and_replay_density(rate, warp_device):
    workspace, inputs, outputs = _slab(warp_device, rate=rate)
    batch = HistoryBatch(73194, 0, 4096)
    trace_histories(*inputs, batch=batch, workspace=workspace, **outputs)
    status = outputs["out_status"].numpy()
    events = outputs["out_events"].numpy()
    assert np.all(np.isin(status, [int(HistoryStatus.ESCAPED), int(HistoryStatus.ABSORBED)]))
    score = outputs["out_score"].numpy()
    exact = math.exp(-rate)
    assert abs(score.mean() - exact) < 7 * math.sqrt(exact * (1 - exact) / batch.count) + 1e-15
    derivative = wp.empty(batch.count, dtype=wp.float64, device=warp_device)
    derivative_histories(
        *inputs,
        batch=batch,
        workspace=workspace,
        parameter=TransportParameter("log-material-density", 0),
        out_pixel=outputs["out_pixel"],
        out_derivative=derivative,
        out_status=outputs["out_status"],
    )
    observed = derivative.numpy()
    standard_error = observed.std(ddof=1) / math.sqrt(batch.count)
    assert abs(observed.mean() + rate * exact) < 7 * standard_error + 1e-15
    expected_replay = -rate / (2.0 - rate) * events * score
    np.testing.assert_allclose(observed, expected_replay, atol=3e-14, rtol=0)
    np.testing.assert_array_equal(outputs["out_status"].numpy(), status)


def test_ordinary_flights_still_enforce_crossing_budget(warp_device):
    physical, inputs, outputs = _slab(warp_device, rate=0.0)
    workspace = prepare_transport(
        physical.spec,
        material_ids=physical.material_ids,
        absorption=physical.absorption,
        scattering=physical.scattering,
        max_histories=physical.max_histories,
        device=warp_device,
    )
    with pytest.raises(IncompleteHistoryError):
        trace_histories(*inputs, batch=HistoryBatch(73194, 0, 4096), workspace=workspace, **outputs)
    assert np.all(outputs["out_status"].numpy() == int(HistoryStatus.CROSSING_BUDGET))


def test_null_candidates_still_exhaust_the_event_budget(warp_device):
    workspace, inputs, outputs = _slab(warp_device, rate=0.0, majorant=100.0, events=1)
    with pytest.raises(IncompleteHistoryError):
        trace_histories(*inputs, batch=HistoryBatch(73194, 0, 4096), workspace=workspace, **outputs)
    assert np.any(outputs["out_status"].numpy() == int(HistoryStatus.EVENT_BUDGET))
    assert np.all(outputs["out_events"].numpy() <= 1)


def test_connection_integrals_still_enforce_crossing_budget(warp_device):
    transport, inputs, _outputs = _slab(warp_device, count=1)
    readout = prepare_transport_readout(transport, spec=TransportReadout(next_event=True))
    output = wp.empty(1, dtype=wp.float64, device=warp_device)
    with pytest.raises(IncompleteHistoryError):
        trace_readout(*inputs, batch=HistoryBatch(73194, 0, 1), workspace=readout, out_mean=output)
    np.testing.assert_array_equal(
        readout._history_status.numpy(), [int(HistoryStatus.CROSSING_BUDGET)]
    )


def test_energy_changes_and_local_deposition_with_direct_flights(warp_device):
    base, inputs, outputs = _slab(warp_device)
    physical = _physical(2.0)
    line = FluorescenceLine(40.0, 60.0, (0.0, 1.0), physical.scattering_tables[0].provenance)
    workspace = prepare_transport(
        replace(base.spec, coefficient_energies_kev=(40.0, 80.0)),
        material_ids=base.material_ids,
        absorption=wp.array([0.2, 0.5], dtype=wp.float64, device=warp_device),
        scattering=wp.zeros(2, dtype=wp.float64, device=warp_device),
        physical=replace(
            physical,
            majorant_mm_inverse=(1.0, 2.0),
            rayleigh_mm_inverse=(0.0, 0.0),
            fluorescence=((line,),),
        ),
        max_histories=base.max_histories,
        device=warp_device,
    )
    initial = wp.empty(base.max_histories, dtype=wp.float64, device=warp_device)
    deposited = wp.empty_like(initial)
    spectrum = wp.empty(base.max_histories, dtype=wp.int32, device=warp_device)
    local = wp.empty(math.prod(base.spec.grid.shape), dtype=wp.float64, device=warp_device)
    trace_energy_balance(
        *inputs,
        batch=HistoryBatch(819141, 0, base.max_histories),
        workspace=workspace,
        **outputs,
        out_initial_energy=initial,
        out_deposited_energy=deposited,
        out_spectrum_index=spectrum,
        out_local_deposition=local,
    )
    final = outputs["out_energy"].numpy()
    np.testing.assert_array_equal(initial.numpy(), final + deposited.numpy())
    assert np.any((final == 40.0) & (deposited.numpy() == 40.0))
    assert np.count_nonzero(local.numpy()) > 5
    assert local.numpy().sum() == deposited.numpy().sum()


def test_mixed_majorant_violation_is_reported_even_without_a_candidate(warp_device):
    rate = 0.90000003
    workspace, inputs, _outputs = _slab(
        warp_device,
        count=1,
        cells=1,
        depth=1e-12,
        rate=rate,
        majorant=math.nextafter(rate, math.inf),
    )
    output = wp.empty(1, dtype=wp.float64, device=warp_device)
    ordinary = prepare_transport_readout(workspace, spec=TransportReadout())
    trace_readout(*inputs, batch=HistoryBatch(51231, 0, 1), workspace=ordinary, out_mean=output)
    np.testing.assert_array_equal(output.numpy(), [1.0])
    mixed = prepare_transport_readout(workspace, spec=TransportReadout(precision="mixed-rates"))
    with pytest.raises(NumericalError):
        trace_readout(*inputs, batch=HistoryBatch(51231, 0, 1), workspace=mixed, out_mean=output)


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("voxel_scale", [None, 1.0])
def test_interpolated_majorant_violation_survives_candidate_free_escape(
    validate, voxel_scale, warp_device
):
    base, inputs, outputs = _slab(warp_device, count=1, cells=1, depth=1e-9)
    physical = _physical(1.0)
    density = 2.817970875024506
    workspace = prepare_transport(
        replace(
            base.spec,
            energy_kev=0.7975224520079335,
            coefficient_energies_kev=(0.25, 1.25),
        ),
        material_ids=base.material_ids,
        absorption=wp.array(
            [0.07826134497545546, 0.2200903750777381], dtype=wp.float64, device=warp_device
        ),
        scattering=wp.array(
            [0.16745305550035292, 0.052694590349504265], dtype=wp.float64, device=warp_device
        ),
        voxel_density=(
            None
            if voxel_scale is None
            else wp.array([voxel_scale], dtype=wp.float64, device=warp_device)
        ),
        physical=replace(
            physical,
            rayleigh_mm_inverse=(0.0, 0.0),
            majorant_mm_inverse=(0.6924160241149357, 0.7687000877185358),
        ),
        max_histories=1,
        device=warp_device,
    )
    batch = HistoryBatch(51231, 0, 1)
    trace_histories(*inputs, batch=batch, workspace=workspace, **outputs)
    np.testing.assert_array_equal(outputs["out_score"].numpy(), [1.0])
    inputs[3].fill_(density)
    # Both coefficient nodes retain strict slack, but their interpolated rate
    # rounds one ULP above the interpolated majorant.
    with pytest.raises(NumericalError):
        trace_histories(*inputs, batch=batch, workspace=workspace, validate=validate, **outputs)
        workspace.check_status()
    np.testing.assert_array_equal(outputs["out_events"].numpy(), [0])
    np.testing.assert_array_equal(
        outputs["out_status"].numpy(), [int(HistoryStatus.NUMERICAL_FAILURE)]
    )


def test_density_bounds_cover_each_material_and_ignore_absent_materials(warp_device):
    base, inputs, outputs = _slab(warp_device, count=1, depth=1e-9)
    physical = _physical(10.0)
    cells = math.prod(base.spec.grid.shape)
    workspace = prepare_transport(
        replace(
            base.spec,
            grid=replace(base.spec.grid, materials=3),
            coefficient_energies_kev=(40.0, 120.0),
        ),
        material_ids=wp.array(np.arange(cells) % 2, dtype=wp.int32, device=warp_device),
        absorption=wp.array([0.1, 0.2, 0.2, 0.1, 1e6, 1e6], dtype=wp.float64, device=warp_device),
        scattering=wp.zeros(6, dtype=wp.float64, device=warp_device),
        voxel_density=wp.array(
            np.tile([2.0, 4.0, 0.0, 0.25], cells // 4), dtype=wp.float64, device=warp_device
        ),
        physical=replace(
            physical,
            scattering_tables=physical.scattering_tables * 3,
            rayleigh_mm_inverse=(0.0,) * 6,
            majorant_mm_inverse=(10.0, 10.0),
        ),
        max_histories=1,
        device=warp_device,
    )
    trace_histories(
        *inputs[:3],
        wp.ones(3, dtype=wp.float64, device=warp_device),
        batch=HistoryBatch(51231, 0, 1),
        workspace=workspace,
        **outputs,
    )
    np.testing.assert_array_equal(outputs["out_score"].numpy(), [1.0])
    np.testing.assert_array_equal(outputs["out_events"].numpy(), [0])
    np.testing.assert_array_equal(workspace._parameters.majorant_max_density.numpy(), [2, 4, 0])
