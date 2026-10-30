"""Trace photon histories and replay their likelihood-ratio derivatives.

Each thread traces one original history. Sparse output contains one detector
contribution per history; derivative calls replay the same flight function and
random coordinates.
"""

# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false
# Integer constructors declare mutable Warp loop variables.
# ruff: noqa: UP018, RUF046
import warp as wp

from dpt.kernels.numerics import attenuated_product4, product3, product4
from dpt.kernels.random import random4

from ._constants import (
    ABSORBED,
    ANGLE_BUDGET,
    CROSSING_BUDGET,
    ELECTRON_REST_ENERGY_KEV,
    ENERGY_SUPPORT,
    ESCAPED,
    EVENT_BUDGET,
    INPUT_MASK,
    LOG_SOURCE_AMPLITUDE,
    NUMERICAL_FAILURE,
    NUMERICAL_MASK,
    RANDOM_NAMESPACE_BIT,
    ROULETTE_TERMINATED,
    SOURCE_AMPLITUDE,
)
from .bound_compton_kernels import bound_energy_event, bound_phase_pdf, bound_scatter_event
from .physical_kernels import (
    PhysicalParameters,
    energy_table_value,
    fluorescence_event,
    rayleigh_phase_pdf,
    rayleigh_scatter_event,
    record_deposition,
    spectrum_index,
)
from .sampling_kernels import RandomDesign, transport_random4

wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})

ABSORBED = wp.constant(ABSORBED)
ANGLE_BUDGET = wp.constant(ANGLE_BUDGET)
CROSSING_BUDGET = wp.constant(CROSSING_BUDGET)
ENERGY_SUPPORT = wp.constant(ENERGY_SUPPORT)
ESCAPED = wp.constant(ESCAPED)
EVENT_BUDGET = wp.constant(EVENT_BUDGET)
INPUT_MASK = wp.constant(INPUT_MASK)
NUMERICAL_FAILURE = wp.constant(NUMERICAL_FAILURE)
NUMERICAL_MASK = wp.constant(NUMERICAL_MASK)

MAX_DISTANCE = wp.constant(wp.float64(1.7976931348623157e308))
TWO_PI = wp.constant(wp.float64(6.2831853071795864769))
ELECTRON_REST_ENERGY_KEV = wp.constant(wp.float64(ELECTRON_REST_ENERGY_KEV))
MISS_PIXEL = wp.constant(-1)
INVALID_PIXEL = wp.constant(-2)


@wp.func_native("""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    // Every non-exited lane reaches this ballot, including misses and invalid
    // histories. Membership is established before the participating branch.
    const unsigned members = __ballot_sync(0xffffffffu, participating != 0);
    if (!participating) return wp::vec_t<2, double>(0.0, 0.0);
    const unsigned peers = __match_any_sync(members, target);
    const int leader = __ffs(peers) - 1;
    const int lane = threadIdx.x & 31;
    const int count = __popc(peers);
    double total = value;
    for (int delta = 1; delta < count; delta <<= 1) {
        // __fns selects the delta-th participating lane after this lane, even
        // for interleaved detector addresses or inactive/missing histories.
        const unsigned source = __fns(peers, lane, delta + 1);
        const double next = __shfl_sync(peers, total, source < 32 ? source : lane);
        if (source < 32)
            total = maximum ? fmax(total, next) : total + next;
    }
    return wp::vec_t<2, double>(total, lane == leader ? double(count) : 0.0);
#else
    return wp::vec_t<2, double>(participating ? value : 0.0, participating ? 1.0 : 0.0);
#endif
""")
def grouped_tally(value: wp.float64, target: int, maximum: int, participating: int) -> wp.vec2d:
    """Combine equal detector addresses selected by a converged warp ballot.

    Call unconditionally from every live kernel lane, after computing its input
    and participation flag. Every lane named by a peer mask then executes the
    same shuffles. Misses and the final partial warp contribute nothing. Only the
    peer leader updates global storage; unrelated detector addresses stay separate.
    Inputs are nonnegative magnitudes or bounded normalised scores/deviations.
    Summation order remains unspecified, as it was for per-history atomics.
    CUDA targets below SM 70 retain individual atomics; no unsupported match
    intrinsic is emitted there. The measured optimisation targets SM 121.
    """
    ...


@wp.struct
class Parameters:
    origin: wp.vec3d
    spacing: wp.vec3d
    shape: wp.vec3i
    detector_lower: wp.vec2d
    detector_spacing: wp.vec2d
    detector_shape: wp.vec2i
    detector_z: wp.float64
    detector_general: int
    detector_corner: wp.vec3d
    detector_u: wp.vec3d
    detector_v: wp.vec3d
    detector_normal: wp.vec3d
    inverse_rotation: wp.mat33d
    inverse_translation: wp.vec3d
    source_energy: wp.float64
    energy_bins: int
    max_events: int
    max_crossings: int
    max_angle_trials: int
    compton: int
    compton_packed: int
    energy_score: int
    continuous_absorption: int
    voxel_density: int
    majorant_max_density: wp.array(dtype=wp.float64)
    physics: PhysicalParameters
    sampling: RandomDesign


@wp.struct
class Trace:
    pixel: int
    events: int
    real_events: int
    status: int
    energy: wp.float64
    score: wp.float64
    density_score: wp.float64
    absorption_depth: wp.float64
    density_scores: wp.vec4d
    initial_energy: wp.float64
    deposited_energy: wp.float64
    spectrum_index: int
    coefficient_likelihood: wp.float64
    coefficient_pathwise: wp.float64


@wp.struct
class CoefficientRecords:
    enabled: int
    material: int
    node: int
    partial: int


@wp.struct
class VoxelRecords:
    """Opt-in sparse complete-history voxel scores; disabled legacy calls own no buffers."""

    enabled: int
    capacity: int
    history: int
    active: wp.array(dtype=wp.int32)
    voxel: wp.array(dtype=wp.int32)
    score: wp.array(dtype=wp.float64)
    status: wp.array(dtype=wp.int32)


@wp.func
def record_voxel_score(records: VoxelRecords, voxel: int, contribution: wp.float64):
    if records.enabled != 0:
        if records.active[voxel] != 0:
            slot = int((wp.uint32(voxel) * wp.uint32(2654435761)) & wp.uint32(records.capacity - 1))
            found = int(0)
            for _probe in range(records.capacity):
                index = records.history * records.capacity + slot
                target = records.voxel[index]
                if target == voxel or target == -1:
                    records.voxel[index] = voxel
                    value = records.score[index] + contribution
                    records.score[index] = value
                    if not wp.isfinite(value):
                        wp.atomic_or(records.status, 0, 2)
                    found = 1
                    break
                slot = (slot + 1) & (records.capacity - 1)
            if found == 0:
                # This is derivative workspace exhaustion, not a photon terminal
                # status. The seven physical-history codes remain unchanged.
                wp.atomic_or(records.status, 0, 1)


@wp.struct
class ReadoutRecords:
    enabled: int
    history: int
    pixels: int
    bins: int
    mode: int
    next_event: int
    branch: int
    mixed_rates: int
    roulette_threshold: wp.float64
    roulette_survival: wp.float64
    energy_edges: wp.array(dtype=wp.float64)
    scores: wp.array(dtype=wp.float64)
    derivative_enabled: int
    derivative_kind: int
    spectrum_bin: int
    spectrum_probability: wp.float64
    spectrum_score: wp.float64
    coefficient_material: int
    coefficient_node: int
    coefficient_partial: int
    derivatives: wp.array(dtype=wp.float64)


@wp.struct
class GridEntry:
    position: wp.vec3d
    cell: wp.vec3i
    alive: int
    status: int


@wp.struct
class FaceCrossing:
    position: wp.vec3d
    cell: wp.vec3i
    inside: int


@wp.struct
class ComptonEvent:
    cosine: wp.float64
    azimuth: wp.float64
    energy: wp.float64
    status: int


@wp.func
def detector_pixel(position: wp.vec3d, direction: wp.vec3d, p: Parameters) -> int:
    pixel = int(MISS_PIXEL)
    denominator = direction[2]
    numerator = p.detector_z - position[2]
    if p.detector_general != 0:
        denominator = wp.dot(direction, p.detector_normal)
        numerator = wp.dot(p.detector_corner - position, p.detector_normal)
    if denominator > wp.float64(0.0):
        distance = numerator / denominator
        if not wp.isfinite(distance):
            return INVALID_PIXEL
        if distance >= wp.float64(0.0):
            x = position[0] + distance * direction[0] - p.detector_lower[0]
            y = position[1] + distance * direction[1] - p.detector_lower[1]
            if p.detector_general != 0:
                offset = position + distance * direction - p.detector_corner
                x = wp.dot(offset, p.detector_u)
                y = wp.dot(offset, p.detector_v)
            if not wp.isfinite(x) or not wp.isfinite(y):
                return INVALID_PIXEL
            # Test support before converting potentially huge floating coordinates.
            if (
                x >= wp.float64(0.0)
                and y >= wp.float64(0.0)
                and x < wp.float64(p.detector_shape[0]) * p.detector_spacing[0]
                and y < wp.float64(p.detector_shape[1]) * p.detector_spacing[1]
            ):
                # Physical support has already been tested; quotient rounding
                # at its upper edge must not alias a neighbouring row.
                column = wp.min(int(wp.floor(x / p.detector_spacing[0])), p.detector_shape[0] - 1)
                row = wp.min(int(wp.floor(y / p.detector_spacing[1])), p.detector_shape[1] - 1)
                pixel = row * p.detector_shape[0] + column
    return pixel


@wp.func_native("""
// Grid coordinates supplied by Python/NumPy use separately rounded multiply
// and add. Explicit intrinsics also preserve that lattice when another module
// inlines this helper with contraction enabled.
#if defined(__CUDA_ARCH__)
return __dadd_rn(origin, __dmul_rn(static_cast<double>(index), spacing));
#else
const double offset = static_cast<double>(index) * spacing;
return origin + offset;
#endif
""")
def grid_face(origin: wp.float64, spacing: wp.float64, index: int) -> wp.float64:
    """Construct o+i*s with two binary64 roundings on every device."""
    ...


@wp.func
def cell_index(position: wp.vec3d, direction: wp.vec3d, p: Parameters) -> wp.vec3i:
    cell = wp.vec3i(0)
    for axis in range(3):
        coordinate = (position[axis] - p.origin[axis]) / p.spacing[axis]
        index = int(wp.floor(coordinate))
        # A point on a face belongs to the downstream cell. No epsilon displaces
        # a physical path or silently changes a thin voxel's optical thickness.
        # Quotient rounding can put an exactly constructed lattice face in
        # its neighbour. Compare against the same o + i*s faces used by DDA.
        face = grid_face(p.origin[axis], p.spacing[axis], index)
        if position[axis] < face:
            index -= 1
        elif position[axis] >= grid_face(p.origin[axis], p.spacing[axis], index + 1):
            index += 1
        face = grid_face(p.origin[axis], p.spacing[axis], index)
        if direction[axis] < wp.float64(0.0) and position[axis] == face:
            index -= 1
        cell[axis] = wp.clamp(index, 0, p.shape[axis] - 1)
    return cell


@wp.func
def energy_coordinate(
    energy: wp.float64,
    energies: wp.array(dtype=wp.float64),
    p: Parameters,
) -> wp.vec2d:
    if p.energy_bins == 1:
        return wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    lower = int(0)
    upper = p.energy_bins - 1
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if energies[middle] <= energy:
            lower = middle
        else:
            upper = middle
    fraction = (energy - energies[lower]) / (energies[upper] - energies[lower])
    return wp.vec2d(wp.float64(lower), fraction)


@wp.func
def coefficients_at(
    material: int,
    coordinate: wp.vec2d,
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    p: Parameters,
) -> wp.vec2d:
    offset = material * p.energy_bins
    if p.energy_bins == 1:
        return wp.vec2d(absorption[offset], scattering[offset])
    lower = int(coordinate[0])
    upper = lower + 1
    fraction = coordinate[1]
    return wp.vec2d(
        (wp.float64(1.0) - fraction) * absorption[offset + lower]
        + fraction * absorption[offset + upper],
        (wp.float64(1.0) - fraction) * scattering[offset + lower]
        + fraction * scattering[offset + upper],
    )


@wp.func
def direct_majorant_flight_allowed(
    coordinate: wp.vec2d,
    majorant: wp.float64,
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    p: Parameters,
) -> int:
    if p.majorant_max_density.shape[0] != density.shape[0]:
        return 0
    for material in range(density.shape[0]):
        voxel_scale = p.majorant_max_density[material]
        if voxel_scale > wp.float64(0.0):
            interaction = coefficients_at(material, coordinate, absorption, scattering, p)
            rate = density[material] * (interaction[0] + interaction[1])
            if p.voxel_density != 0:
                rate = product3(density[material], voxel_scale, interaction[0] + interaction[1])
            if not wp.isfinite(rate) or rate >= majorant:
                return 0
    return 1


@wp.func
def coefficients_mixed(
    material: int,
    coordinate: wp.vec2d,
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    p: Parameters,
) -> wp.vec2d:
    offset = material * p.energy_bins
    a = wp.float32(absorption[offset])
    scattering_value = wp.float32(scattering[offset])
    if p.energy_bins > 1:
        node = int(coordinate[0])
        fraction = wp.float32(coordinate[1])
        complement = wp.float32(1.0) - fraction
        a = complement * wp.float32(absorption[offset + node]) + fraction * wp.float32(
            absorption[offset + node + 1]
        )
        scattering_value = complement * wp.float32(
            scattering[offset + node]
        ) + fraction * wp.float32(scattering[offset + node + 1])
    return wp.vec2d(wp.float64(a), wp.float64(scattering_value))


@wp.func
def coefficients(
    material: int,
    energy: wp.float64,
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    p: Parameters,
) -> wp.vec2d:
    return coefficients_at(
        material, energy_coordinate(energy, energies, p), absorption, scattering, p
    )


@wp.func
def scatter_direction(direction: wp.vec3d, cosine: wp.float64, azimuth: wp.float64) -> wp.vec3d:
    # Choose the auxiliary axis away from collinearity; polar singularities do
    # not justify dividing by sin(theta) of the incoming direction.
    auxiliary = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    if wp.abs(direction[2]) > wp.float64(0.9):
        auxiliary = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
    tangent = wp.normalize(wp.cross(auxiliary, direction))
    bitangent = wp.cross(direction, tangent)
    sine = wp.sqrt(wp.max(wp.float64(0.0), wp.float64(1.0) - cosine * cosine))
    return wp.normalize(
        cosine * direction + sine * (wp.cos(azimuth) * tangent + wp.sin(azimuth) * bitangent)
    )


@wp.func
def _compton_event_domain(
    energy: wp.float64,
    seed: wp.uint64,
    history: wp.uint64,
    event: wp.uint32,
    max_trials: int,
    domain: wp.uint32,
    sampling: RandomDesign,
) -> ComptonEvent:
    """Cosine, azimuth, outgoing energy and status from the conditional free-electron law."""
    result = ComptonEvent()
    result.cosine = wp.float64(0.0)
    result.azimuth = wp.float64(0.0)
    result.energy = energy
    result.status = ANGLE_BUDGET
    for trial in range(max_trials):
        angular = transport_random4(seed, history, event, domain + wp.uint32(trial), sampling)
        cosine = wp.float64(2.0) * angular[0] - wp.float64(1.0)
        ratio = wp.float64(1.0) / (
            wp.float64(1.0) + energy / ELECTRON_REST_ENERGY_KEV * (wp.float64(1.0) - cosine)
        )
        # Klein-Nishina density divided by its envelope 2. This conditional
        # scattering law is independent of the active material-density scale.
        acceptance = wp.float64(0.5) * (
            ratio * ratio * ratio + ratio - ratio * ratio * (wp.float64(1.0) - cosine * cosine)
        )
        if angular[1] < acceptance:
            result.cosine = cosine
            result.azimuth = TWO_PI * angular[2]
            result.energy = energy * ratio
            result.status = ESCAPED
            break
    return result


@wp.func
def _compton_packed_event_domain(
    energy: wp.float64,
    seed: wp.uint64,
    history: wp.uint64,
    event: wp.uint32,
    max_trials: int,
    domain: wp.uint32,
    azimuth_uniform: wp.float64,
    sampling: RandomDesign,
) -> ComptonEvent:
    """Two bounded KN rejection proposals per block with independent flight azimuth.

    Trial budget counts proposals. An odd budget never consumes the final
    block's second proposal. The explicit packed layout assigns all four block
    lanes, while the already generated collision lane two supplies azimuth.
    """
    result = ComptonEvent()
    result.cosine = wp.float64(0.0)
    result.azimuth = wp.float64(0.0)
    result.energy = energy
    result.status = ANGLE_BUDGET
    alpha = energy / ELECTRON_REST_ENERGY_KEV
    for block in range((max_trials + 1) // 2):
        angular = transport_random4(seed, history, event, domain + wp.uint32(block), sampling)
        for member in range(2):
            if 2 * block + member < max_trials and result.status != ESCAPED:
                cosine = wp.float64(2.0) * angular[2 * member] - wp.float64(1.0)
                ratio = wp.float64(1.0) / (wp.float64(1.0) + alpha * (wp.float64(1.0) - cosine))
                acceptance = wp.float64(0.5) * (
                    ratio * ratio * ratio
                    + ratio
                    - ratio * ratio * (wp.float64(1.0) - cosine * cosine)
                )
                if angular[2 * member + 1] < acceptance:
                    result.cosine = cosine
                    result.azimuth = TWO_PI * azimuth_uniform
                    result.energy = energy * ratio
                    result.status = ESCAPED
        if result.status == ESCAPED:
            break
    return result


@wp.func
def _compton_event(
    energy: wp.float64,
    seed: wp.uint64,
    history: wp.uint64,
    event: wp.uint32,
    max_trials: int,
) -> ComptonEvent:
    return _compton_event_domain(
        energy,
        seed,
        history,
        event,
        max_trials,
        wp.uint32(RANDOM_NAMESPACE_BIT),
        RandomDesign(),
    )


@wp.func
def compton_scatter(
    energy: wp.float64,
    seed: wp.uint64,
    history: wp.uint64,
    event: wp.uint32,
    max_trials: int,
) -> wp.vec4d:
    """Keep the established packed sampler output for existing kernel callers."""
    result = _compton_event(energy, seed, history, event, max_trials)
    return wp.vec4d(result.cosine, result.azimuth, result.energy, wp.float64(result.status))


@wp.func
def enter_grid(position: wp.vec3d, direction: wp.vec3d, p: Parameters) -> GridEntry:
    """Intersect a forward ray and assign its downstream cell without an epsilon."""
    result = GridEntry()
    result.status = ESCAPED
    result.cell = wp.vec3i(0)
    entry = wp.float64(0.0)
    exit_distance = wp.float64(MAX_DISTANCE)
    intersects = int(1)
    for axis in range(3):
        lower = p.origin[axis]
        upper = grid_face(lower, p.spacing[axis], p.shape[axis])
        if direction[axis] == wp.float64(0.0):
            if position[axis] < lower or position[axis] >= upper:
                intersects = 0
        else:
            first = (lower - position[axis]) / direction[axis]
            second = (upper - position[axis]) / direction[axis]
            if not wp.isfinite(first) or not wp.isfinite(second):
                result.status = NUMERICAL_FAILURE
                intersects = 0
                break
            entry = wp.max(entry, wp.min(first, second))
            exit_distance = wp.min(exit_distance, wp.max(first, second))
    if exit_distance <= entry:
        intersects = 0
    result.alive = intersects
    if result.alive != 0:
        position += entry * direction
        # Only round the intersection coordinate to a face, not the travelled
        # distance. The entry came from these same faces and cannot be outside.
        for axis in range(3):
            position[axis] = wp.clamp(
                position[axis],
                p.origin[axis],
                grid_face(p.origin[axis], p.spacing[axis], p.shape[axis]),
            )
    if result.alive != 0:
        result.cell = cell_index(position, direction, p)
    result.position = position
    return result


@wp.func
def distances_to_faces(
    position: wp.vec3d, direction: wp.vec3d, cell: wp.vec3i, p: Parameters
) -> wp.vec3d:
    """Distances to the next cell face on each axis, retaining exact ties."""
    distances = wp.vec3d(MAX_DISTANCE)
    for axis in range(3):
        if direction[axis] != wp.float64(0.0):
            face_index = cell[axis]
            if direction[axis] > wp.float64(0.0):
                face_index += 1
            face = grid_face(p.origin[axis], p.spacing[axis], face_index)
            distance = (face - position[axis]) / direction[axis]
            if wp.isfinite(distance):
                distance = wp.max(distance, wp.float64(0.0))
            distances[axis] = distance
    return distances


@wp.func
def cross_faces(
    position: wp.vec3d,
    direction: wp.vec3d,
    cell: wp.vec3i,
    face_distances: wp.vec3d,
    distance: wp.float64,
    p: Parameters,
) -> FaceCrossing:
    """Cross every tied face and snap only its coordinate to the known cell face."""
    result = FaceCrossing()
    result.position = position
    result.cell = cell
    result.inside = 1
    for axis in range(3):
        if face_distances[axis] == distance:
            if direction[axis] > wp.float64(0.0):
                result.cell[axis] += 1
                result.position[axis] = grid_face(
                    p.origin[axis], p.spacing[axis], result.cell[axis]
                )
            else:
                result.position[axis] = grid_face(
                    p.origin[axis], p.spacing[axis], result.cell[axis]
                )
                result.cell[axis] -= 1
        if result.cell[axis] < 0 or result.cell[axis] >= p.shape[axis]:
            result.inside = 0
    return result


@wp.func
def woodcock_flight(
    position: wp.vec3d,
    direction: wp.vec3d,
    optical_distance: wp.float64,
    majorant: wp.float64,
    p: Parameters,
) -> GridEntry:
    """Advance to a majorant candidate or the outer grid boundary."""
    result = GridEntry()
    result.position = position
    result.status = ESCAPED
    result.alive = 0
    if not wp.isfinite(majorant) or majorant <= wp.float64(0.0):
        result.status = NUMERICAL_FAILURE
        return result
    distances = wp.vec3d(MAX_DISTANCE)
    faces = wp.vec3d(wp.float64(0.0))
    for axis in range(3):
        if direction[axis] != wp.float64(0.0):
            face = p.origin[axis]
            if direction[axis] > wp.float64(0.0):
                face = grid_face(p.origin[axis], p.spacing[axis], p.shape[axis])
            faces[axis] = face
            distances[axis] = wp.max((face - position[axis]) / direction[axis], wp.float64(0.0))
    exit_distance = wp.min(distances[0], wp.min(distances[1], distances[2]))
    if not wp.isfinite(exit_distance):
        result.status = NUMERICAL_FAILURE
        return result
    # A tiny majorant can overflow the candidate distance; that flight exits.
    distance = optical_distance / majorant
    if distance < exit_distance:
        result.position = position + distance * direction
        result.cell = cell_index(result.position, direction, p)
        result.alive = 1
    else:
        result.position = position + exit_distance * direction
        for axis in range(3):
            if direction[axis] != wp.float64(0.0) and distances[axis] == exit_distance:
                result.position[axis] = faces[axis]
    return result


@wp.func
def record_readout(
    records: ReadoutRecords,
    pixel: int,
    energy: wp.float64,
    score: wp.float64,
    scattered: int,
    derivative: wp.float64,
):
    if pixel >= 0 and (
        records.mode == 0
        or (records.mode == 1 and scattered == 0)
        or (records.mode == 2 and scattered != 0)
    ):
        channel = int(0)
        if records.energy_edges.shape[0] != 0:
            channel = int(-1)
            for b in range(records.bins):
                if energy >= records.energy_edges[b] and energy < records.energy_edges[b + 1]:
                    channel = b
        if channel >= 0:
            index = (records.history * records.bins + channel) * records.pixels + pixel
            records.scores[index] += score
            if records.derivative_enabled != 0:
                records.derivatives[index] += derivative + score * records.spectrum_score


@wp.func
def connection_depth(
    position: wp.vec3d,
    direction: wp.vec3d,
    energy: wp.float64,
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    p: Parameters,
    active: CoefficientRecords,
    mixed_rates: int,
) -> wp.vec3d:
    if energy < energies[0] or energy > energies[p.energy_bins - 1]:
        return wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(ENERGY_SUPPORT))
    entry = enter_grid(position, direction, p)
    current = entry.position
    cell = entry.cell
    alive = entry.alive
    depth = wp.float64(0.0)
    derivative_depth = wp.float64(0.0)
    status = entry.status
    crossings = int(0)
    coordinate = energy_coordinate(energy, energies, p)
    while alive != 0:
        flat = (cell[2] * p.shape[1] + cell[1]) * p.shape[0] + cell[0]
        material = material_ids[flat]
        interaction = coefficients_at(material, coordinate, absorption, scattering, p)
        if mixed_rates != 0:
            interaction = coefficients_mixed(material, coordinate, absorption, scattering, p)
        rate = density[material] * (interaction[0] + interaction[1])
        if p.voxel_density != 0:
            rate *= voxel_density[flat]
        if mixed_rates != 0:
            scale = wp.float32(density[material])
            if p.voxel_density != 0:
                scale *= wp.float32(voxel_density[flat])
            rate = wp.float64(scale * (wp.float32(interaction[0]) + wp.float32(interaction[1])))
        face = distances_to_faces(current, direction, cell, p)
        distance = wp.min(face[0], wp.min(face[1], face[2]))
        depth += rate * distance
        if active.enabled != 0 and material == active.material:
            basis = wp.float64(0.0)
            if p.energy_bins == 1:
                basis = wp.float64(1.0)
            elif active.node == int(coordinate[0]):
                basis = wp.float64(1.0) - coordinate[1]
            elif active.node == int(coordinate[0]) + 1:
                basis = coordinate[1]
            node = absorption[material * p.energy_bins + active.node]
            if active.partial != 0:
                node = scattering[material * p.energy_bins + active.node]
            derivative_rate = density[material] * basis * node
            if p.voxel_density != 0:
                derivative_rate *= voxel_density[flat]
            derivative_depth += derivative_rate * distance
        if not wp.isfinite(depth) or not wp.isfinite(distance):
            status = NUMERICAL_FAILURE
            break
        current += distance * direction
        crossed = cross_faces(current, direction, cell, face, distance, p)
        current = crossed.position
        cell = crossed.cell
        alive = crossed.inside
        crossings += 1
        if alive != 0 and crossings >= p.max_crossings:
            status = CROSSING_BUDGET
            break
    return wp.vec3d(depth, derivative_depth, wp.float64(status))


@wp.func
def free_electron_phase_pdf(energy: wp.float64, cosine: wp.float64) -> wp.float64:
    alpha = energy / ELECTRON_REST_ENERGY_KEV
    ratio = wp.float64(1.0) / (wp.float64(1.0) + alpha * (wp.float64(1.0) - cosine))
    shape = ratio * ratio * (ratio + wp.float64(1.0) / ratio - (wp.float64(1.0) - cosine * cosine))
    # The analytic total and its convergent small-energy series integrate the
    # same conditional Klein-Nishina shape. The series avoids cancellation.
    normal = wp.float64(8.0) / wp.float64(3.0)
    if alpha > wp.float64(0.05):
        logterm = wp.log(wp.float64(1.0) + wp.float64(2.0) * alpha)
        normal = wp.float64(2.0) * (
            (wp.float64(1.0) + alpha)
            / (alpha * alpha)
            * (
                wp.float64(2.0)
                * (wp.float64(1.0) + alpha)
                / (wp.float64(1.0) + wp.float64(2.0) * alpha)
                - logterm / alpha
            )
            + logterm / (wp.float64(2.0) * alpha)
            - (wp.float64(1.0) + wp.float64(3.0) * alpha)
            / (
                (wp.float64(1.0) + wp.float64(2.0) * alpha)
                * (wp.float64(1.0) + wp.float64(2.0) * alpha)
            )
        )
    else:
        # For alpha<=0.05 the omitted tail starts beyond order19.
        normal = wp.float64(-10880905.198268399)
        normal = wp.float64(5175602.943358396) + alpha * normal
        normal = wp.float64(-2455262.1660818714) + alpha * normal
        normal = wp.float64(1161297.018231854) + alpha * normal
        normal = wp.float64(-547445.1241830066) + alpha * normal
        normal = wp.float64(257100.29803921568) + alpha * normal
        normal = wp.float64(-120227.35238095238) + alpha * normal
        normal = wp.float64(55945.658608058606) + alpha * normal
        normal = wp.float64(-25885.069597069596) + alpha * normal
        normal = wp.float64(11896.540792540793) + alpha * normal
        normal = wp.float64(-5424.09696969697) + alpha * normal
        normal = wp.float64(2449.3252525252524) + alpha * normal
        normal = wp.float64(-1092.9777777777779) + alpha * normal
        normal = wp.float64(480.5079365079365) + alpha * normal
        normal = wp.float64(-207.23809523809524) + alpha * normal
        normal = wp.float64(87.16190476190476) + alpha * normal
        normal = wp.float64(-35.46666666666667) + alpha * normal
        normal = wp.float64(13.866666666666667) + alpha * normal
        normal = wp.float64(-5.333333333333333) + alpha * normal
        normal = wp.float64(2.6666666666666665) + alpha * normal
    return shape / (TWO_PI * normal)


@wp.func
def next_event_readout(
    position: wp.vec3d,
    direction: wp.vec3d,
    energy: wp.float64,
    material: int,
    channel: int,
    source_weight: wp.float64,
    source_amplitude: wp.float64,
    absorption_depth: wp.float64,
    branch_weight: wp.float64,
    coefficient_score: wp.float64,
    active: CoefficientRecords,
    seed: wp.uint64,
    history: wp.uint64,
    event: wp.uint32,
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    p: Parameters,
    records: ReadoutRecords,
) -> int:
    draws = transport_random4(
        seed,
        history,
        event,
        wp.uint32(0x40000000) + wp.uint32(records.branch * 65536),
        p.sampling,
    )
    width = wp.float64(p.detector_shape[0]) * p.detector_spacing[0]
    height = wp.float64(p.detector_shape[1]) * p.detector_spacing[1]
    target = p.detector_corner + draws[0] * width * p.detector_u + draws[1] * height * p.detector_v
    offset = target - position
    distance = wp.length(offset)
    outgoing = offset / distance
    cosine = wp.clamp(wp.dot(direction, outgoing), wp.float64(-1.0), wp.float64(1.0))
    projection = wp.dot(outgoing, p.detector_normal)
    out_energy = energy
    phase = wp.float64(1.0) / (wp.float64(2.0) * TWO_PI)
    if channel == 1:
        phase = rayleigh_phase_pdf(energy, material, cosine, p.physics)
    elif p.compton == 2 and channel == 0:
        phase = bound_phase_pdf(energy, cosine, material, p.physics)
        if phase == wp.float64(0.0):
            return ESCAPED
        broadening = bound_energy_event(
            energy, cosine, material, seed, history, event, 1, p.physics
        )
        if broadening[3] != wp.float64(0.0):
            return ANGLE_BUDGET
        out_energy = broadening[0]
    elif p.compton != 0 and channel == 0:
        out_energy = energy / (
            wp.float64(1.0) + energy / ELECTRON_REST_ENERGY_KEV * (wp.float64(1.0) - cosine)
        )
        phase = free_electron_phase_pdf(energy, cosine)
    if p.physics.enabled != 0 and out_energy <= p.physics.cutoff:
        return ESCAPED
    integral = connection_depth(
        position,
        outgoing,
        out_energy,
        material_ids,
        voxel_density,
        energies,
        absorption,
        scattering,
        density,
        p,
        active,
        records.mixed_rates,
    )
    status = int(integral[2])
    if status == ESCAPED and projection > wp.float64(0.0):
        factor = branch_weight * phase * width * height * projection / (distance * distance)
        if p.energy_score != 0:
            factor *= out_energy
        score = attenuated_product4(
            absorption_depth + integral[0],
            source_weight,
            source_amplitude,
            factor,
            wp.float64(1.0),
        )
        pixel = int(draws[1] * wp.float64(p.detector_shape[1])) * p.detector_shape[0] + int(
            draws[0] * wp.float64(p.detector_shape[0])
        )
        record_readout(
            records,
            pixel,
            out_energy,
            score,
            1,
            score * (coefficient_score - integral[1]),
        )
        if not wp.isfinite(score):
            status = NUMERICAL_FAILURE
    return status


def _make_walk(recording: bool, coefficient_recording: bool = False, readout: bool = False):
    @wp.func(
        name="walk_readout"
        if readout
        else "walk_coefficient"
        if coefficient_recording
        else ("walk_recorded" if recording else "walk_unrecorded")
    )
    def implementation(
        initial_position: wp.vec3d,
        initial_direction: wp.vec3d,
        source_weight: wp.float64,
        source_amplitude: wp.float64,
        seed: wp.uint64,
        history: wp.uint64,
        active_material: int,
        active_materials: wp.vec4i,
        active_count: int,
        material_ids: wp.array(dtype=wp.int32),
        voxel_density: wp.array(dtype=wp.float64),
        energies: wp.array(dtype=wp.float64),
        absorption: wp.array(dtype=wp.float64),
        scattering: wp.array(dtype=wp.float64),
        density: wp.array(dtype=wp.float64),
        p: Parameters,
        records: VoxelRecords,
        coefficient_records: CoefficientRecords,
        readout_records: ReadoutRecords,
    ) -> Trace:
        result = Trace()
        result.pixel = MISS_PIXEL
        result.events = 0
        result.real_events = 0
        result.status = ESCAPED
        result.energy = p.source_energy
        result.score = wp.float64(0.0)
        result.density_score = wp.float64(0.0)
        result.density_scores = wp.vec4d(wp.float64(0.0))
        result.absorption_depth = wp.float64(0.0)
        result.coefficient_likelihood = wp.float64(0.0)
        result.coefficient_pathwise = wp.float64(0.0)
        result.spectrum_index = -1
        result.deposited_energy = wp.float64(0.0)
        if p.physics.enabled != 0 and p.physics.spectrum_bins != 0:
            source_draw = transport_random4(
                seed, history, wp.uint32(0x80000002), wp.uint32(0), p.sampling
            )
            result.spectrum_index = spectrum_index(source_draw[0], p.physics)
            result.energy = p.physics.spectrum_energies[result.spectrum_index]
        p.physics.sampling = p.sampling
        result.initial_energy = result.energy
        if wp.static(readout):
            p.physics.random_branch = readout_records.branch
            if readout_records.derivative_kind == 2:
                readout_records.spectrum_score = (
                    wp.float64(int(result.spectrum_index == readout_records.spectrum_bin))
                    - readout_records.spectrum_probability
                )
        position = initial_position
        direction = initial_direction
        if p.detector_general != 0:
            position = p.inverse_rotation * initial_position + p.inverse_translation
            direction = p.inverse_rotation * initial_direction
        entry = enter_grid(position, direction, p)
        position = entry.position
        cell = entry.cell
        alive = entry.alive
        result.status = entry.status
        crossings = int(0)
        branch_weight = wp.float64(1.0)
        allow_direct_woodcock = int(p.physics.enabled != 0 and p.physics.woodcock != 0)
        direct_woodcock = int(0)
        checked_energy = wp.float64(-1.0)
        if wp.static(readout):
            # FP32 interpolation can exceed a bound valid in FP64. The mixed
            # path checks that rounded rate in every traversed cell.
            if readout_records.mixed_rates != 0:
                allow_direct_woodcock = 0
        if wp.static(readout):
            # A photon entering material at the cutoff deposits before flight.
            # A ray missing the grid still carries its detector contribution.
            if readout_records.next_event != 0 and (
                alive == 0 or p.physics.enabled == 0 or result.energy > p.physics.cutoff
            ):
                integral = connection_depth(
                    position,
                    direction,
                    result.energy,
                    material_ids,
                    voxel_density,
                    energies,
                    absorption,
                    scattering,
                    density,
                    p,
                    coefficient_records,
                    readout_records.mixed_rates,
                )
                if int(integral[2]) != ESCAPED:
                    result.status = int(integral[2])
                    alive = 0
                else:
                    pixel = detector_pixel(position, direction, p)
                    if pixel == INVALID_PIXEL:
                        result.status = NUMERICAL_FAILURE
                        alive = 0
                    elif pixel >= 0:
                        energy_score = wp.float64(1.0)
                        if p.energy_score != 0:
                            energy_score = result.energy
                        primary_score = attenuated_product4(
                            integral[0],
                            source_weight,
                            source_amplitude,
                            energy_score,
                            wp.float64(1.0),
                        )
                        record_readout(
                            readout_records,
                            pixel,
                            result.energy,
                            primary_score,
                            0,
                            -primary_score * integral[1],
                        )
        while alive != 0:
            if p.physics.enabled != 0 and result.energy <= p.physics.cutoff:
                cutoff_voxel = (cell[2] * p.shape[1] + cell[1]) * p.shape[0] + cell[0]
                record_deposition(
                    p.physics,
                    cutoff_voxel,
                    result.energy,
                    source_weight,
                    source_amplitude,
                )
                result.deposited_energy += result.energy
                result.energy = wp.float64(0.0)
                result.status = ABSORBED
                break
            if result.events >= p.max_events:
                result.status = EVENT_BUDGET
                break
            if result.energy < energies[0] or result.energy > energies[p.energy_bins - 1]:
                result.status = ENERGY_SUPPORT
                break
            domain = wp.uint32(0)
            if wp.static(readout):
                domain = wp.uint32(readout_records.branch * 65536)
            draws = transport_random4(seed, history, wp.uint32(result.events), domain, p.sampling)
            residual = -wp.log(draws[0])
            coordinate = energy_coordinate(result.energy, energies, p)
            collision = int(0)
            material = int(0)
            interaction = wp.vec2d(wp.float64(0.0))
            physical_extinction = wp.float64(0.0)
            extinction = wp.float64(0.0)
            flat = int(0)
            majorant = wp.float64(0.0)
            if p.physics.enabled != 0 and p.physics.woodcock != 0:
                majorant = energy_table_value(p.physics.majorant, 0, coordinate, p.energy_bins)
            if allow_direct_woodcock != 0 and checked_energy != result.energy:
                # Interpolation can consume one-ULP nodal slack. Keep cellwise
                # checks when the rounded rate at this energy reaches the bound.
                direct_woodcock = direct_majorant_flight_allowed(
                    coordinate, majorant, absorption, scattering, density, p
                )
                checked_energy = result.energy
            if direct_woodcock != 0:
                flight = woodcock_flight(position, direction, residual, majorant, p)
                position = flight.position
                cell = flight.cell
                alive = flight.alive
                if alive == 0:
                    result.status = flight.status
            while alive != 0 and collision == 0:
                flat = (cell[2] * p.shape[1] + cell[1]) * p.shape[0] + cell[0]
                material = material_ids[flat]
                interaction = coefficients_at(material, coordinate, absorption, scattering, p)
                if wp.static(readout):
                    if readout_records.mixed_rates != 0:
                        interaction = coefficients_mixed(
                            material, coordinate, absorption, scattering, p
                        )
                extinction = density[material] * (interaction[0] + interaction[1])
                if p.voxel_density != 0:
                    extinction = product3(
                        density[material],
                        voxel_density[flat],
                        interaction[0] + interaction[1],
                    )
                physical_extinction = extinction
                if p.physics.enabled != 0 and p.physics.woodcock != 0:
                    extinction = majorant
                    if physical_extinction >= extinction:
                        result.status = NUMERICAL_FAILURE
                        alive = 0
                        break
                absorption_rate = wp.float64(0.0)
                if p.continuous_absorption != 0:
                    extinction = density[material] * interaction[1]
                    absorption_rate = density[material] * interaction[0]
                    if p.voxel_density != 0:
                        extinction = product3(
                            density[material], voxel_density[flat], interaction[1]
                        )
                        absorption_rate = product3(
                            density[material], voxel_density[flat], interaction[0]
                        )
                if wp.static(readout):
                    if readout_records.mixed_rates != 0:
                        scale = wp.float32(density[material])
                        if p.voxel_density != 0:
                            scale *= wp.float32(voxel_density[flat])
                        partial_a = wp.float32(interaction[0])
                        partial_s = wp.float32(interaction[1])
                        physical_extinction = wp.float64(scale * (partial_a + partial_s))
                        if p.physics.enabled != 0 and p.physics.woodcock != 0:
                            if physical_extinction >= extinction:
                                result.status = NUMERICAL_FAILURE
                                alive = 0
                                break
                        if p.physics.enabled == 0 or p.physics.woodcock == 0:
                            extinction = physical_extinction
                        if p.continuous_absorption != 0:
                            extinction = wp.float64(scale * partial_s)
                            absorption_rate = wp.float64(scale * partial_a)
                coefficient_rate = wp.float64(0.0)
                if wp.static(coefficient_recording):
                    if (
                        coefficient_records.enabled != 0
                        and material == coefficient_records.material
                    ):
                        basis = wp.float64(0.0)
                        if p.energy_bins == 1:
                            basis = wp.float64(1.0)
                        elif coefficient_records.node == int(coordinate[0]):
                            basis = wp.float64(1.0) - coordinate[1]
                        elif coefficient_records.node == int(coordinate[0]) + 1:
                            basis = coordinate[1]
                        node_value = absorption[material * p.energy_bins + coefficient_records.node]
                        if coefficient_records.partial != 0:
                            node_value = scattering[
                                material * p.energy_bins + coefficient_records.node
                            ]
                        coefficient_rate = density[material] * basis * node_value
                        if p.voxel_density != 0:
                            coefficient_rate *= voxel_density[flat]
                if direct_woodcock != 0:
                    collision = 1
                    break
                face_distance = distances_to_faces(position, direction, cell, p)
                distance = wp.min(face_distance[0], wp.min(face_distance[1], face_distance[2]))
                if not wp.isfinite(distance):
                    result.status = NUMERICAL_FAILURE
                    alive = 0
                    break
                optical_distance = extinction * distance
                if (
                    not wp.isfinite(extinction)
                    or not wp.isfinite(optical_distance)
                    or not wp.isfinite(absorption_rate)
                ):
                    result.status = NUMERICAL_FAILURE
                    alive = 0
                    break
                realised_distance = distance
                if extinction > wp.float64(0.0) and residual < optical_distance:
                    realised_distance = residual / extinction
                if wp.static(coefficient_recording):
                    derivative_depth = coefficient_rate * realised_distance
                    if p.continuous_absorption != 0 and coefficient_records.partial == 0:
                        result.coefficient_pathwise -= derivative_depth
                    elif p.physics.enabled == 0 or p.physics.woodcock == 0:
                        result.coefficient_likelihood -= derivative_depth
                absorption_distance = wp.float64(0.0)
                if p.continuous_absorption != 0:
                    segment_distance = distance
                    if extinction > wp.float64(0.0) and residual < optical_distance:
                        segment_distance = residual / extinction
                    # Use the realised segment, not the entire distance to the face:
                    # scattering can interrupt this segment before the face is reached.
                    absorption_distance = absorption_rate * segment_distance
                    result.absorption_depth += absorption_distance
                    if not wp.isfinite(result.absorption_depth):
                        result.status = NUMERICAL_FAILURE
                        alive = 0
                        break
                if extinction > wp.float64(0.0) and residual < optical_distance:
                    distance = residual / extinction
                    position += distance * direction
                    if wp.static(recording) and (p.physics.enabled == 0 or p.physics.woodcock == 0):
                        voxel_score = wp.float64(1.0) - residual
                        if p.continuous_absorption != 0:
                            voxel_score -= absorption_distance
                        record_voxel_score(records, flat, voxel_score)
                    if material == active_material and (
                        p.physics.enabled == 0 or p.physics.woodcock == 0
                    ):
                        result.density_score += wp.float64(1.0) - residual
                        if p.continuous_absorption != 0:
                            result.density_score -= absorption_distance
                    for column in range(active_count):
                        if material == active_materials[column] and (
                            p.physics.enabled == 0 or p.physics.woodcock == 0
                        ):
                            result.density_scores[column] += wp.float64(1.0) - residual
                            if p.continuous_absorption != 0:
                                result.density_scores[column] -= absorption_distance
                    collision = 1
                else:
                    if wp.static(recording) and (p.physics.enabled == 0 or p.physics.woodcock == 0):
                        voxel_score = -optical_distance
                        if p.continuous_absorption != 0:
                            voxel_score -= absorption_distance
                        record_voxel_score(records, flat, voxel_score)
                    if material == active_material and (
                        p.physics.enabled == 0 or p.physics.woodcock == 0
                    ):
                        result.density_score -= optical_distance
                        if p.continuous_absorption != 0:
                            result.density_score -= absorption_distance
                    for column in range(active_count):
                        if material == active_materials[column] and (
                            p.physics.enabled == 0 or p.physics.woodcock == 0
                        ):
                            result.density_scores[column] -= optical_distance
                            if p.continuous_absorption != 0:
                                result.density_scores[column] -= absorption_distance
                    residual -= optical_distance
                    position += distance * direction
                    crossing = cross_faces(position, direction, cell, face_distance, distance, p)
                    position = crossing.position
                    cell = crossing.cell
                    alive = crossing.inside
                    crossings += 1
                    if alive != 0 and crossings >= p.max_crossings:
                        result.status = CROSSING_BUDGET
                        alive = 0
            if collision != 0:
                result.events += 1
                if p.physics.enabled != 0 and p.physics.woodcock != 0:
                    likelihood = wp.float64(1.0)
                    candidate = transport_random4(
                        seed,
                        history,
                        wp.uint32(result.events - 1),
                        wp.uint32(1) + wp.uint32(p.physics.random_branch) * wp.uint32(65536),
                        p.sampling,
                    )
                    if candidate[0] >= physical_extinction / extinction:
                        likelihood = -physical_extinction / (extinction - physical_extinction)
                        collision = 0
                        if wp.static(coefficient_recording):
                            result.coefficient_likelihood -= coefficient_rate / (
                                extinction - physical_extinction
                            )
                    if wp.static(recording):
                        record_voxel_score(records, flat, likelihood)
                    if material == active_material:
                        result.density_score += likelihood
                    for column in range(active_count):
                        if material == active_materials[column]:
                            result.density_scores[column] += likelihood
            if collision != 0:
                result.real_events += 1
                absorption_probability = wp.float64(0.0)
                if p.continuous_absorption == 0:
                    absorption_probability = interaction[0] / (interaction[0] + interaction[1])
                if p.continuous_absorption == 0 and draws[1] < absorption_probability:
                    if wp.static(coefficient_recording):
                        if coefficient_records.partial == 0 and coefficient_records.enabled != 0:
                            selected_rate = density[material] * interaction[0]
                            if p.voxel_density != 0:
                                selected_rate *= voxel_density[flat]
                            if selected_rate > wp.float64(0.0):
                                result.coefficient_likelihood += coefficient_rate / selected_rate
                    result.status = ABSORBED
                    alive = 0
                    if p.physics.enabled != 0:
                        emitted = fluorescence_event(
                            result.energy,
                            material,
                            coordinate,
                            p.energy_bins,
                            seed,
                            history,
                            wp.uint32(result.events - 1),
                            p.physics,
                        )
                        record_deposition(
                            p.physics,
                            flat,
                            result.energy - emitted[2],
                            source_weight,
                            source_amplitude,
                        )
                        result.deposited_energy += result.energy - emitted[2]
                        result.energy = emitted[2]
                        if int(emitted[3]) == ESCAPED:
                            result.status = ESCAPED
                            alive = 1
                            direction = scatter_direction(direction, emitted[0], emitted[1])
                            if wp.static(readout):
                                if readout_records.next_event != 0:
                                    connection_status = next_event_readout(
                                        position,
                                        direction,
                                        result.energy,
                                        material,
                                        2,
                                        source_weight,
                                        source_amplitude,
                                        result.absorption_depth,
                                        branch_weight,
                                        result.coefficient_likelihood + result.coefficient_pathwise,
                                        coefficient_records,
                                        seed,
                                        history,
                                        wp.uint32(result.events - 1),
                                        material_ids,
                                        voxel_density,
                                        energies,
                                        absorption,
                                        scattering,
                                        density,
                                        p,
                                        readout_records,
                                    )
                                    if connection_status != ESCAPED:
                                        result.status = connection_status
                                        alive = 0
                else:
                    cosine = wp.float64(2.0) * draws[2] - wp.float64(1.0)
                    azimuth = TWO_PI * draws[3]
                    coherent = int(0)
                    rayleigh_rate = wp.float64(0.0)
                    if p.physics.enabled != 0:
                        rayleigh_rate = energy_table_value(
                            p.physics.rayleigh, material, coordinate, p.energy_bins
                        )
                        channel = transport_random4(
                            seed,
                            history,
                            wp.uint32(result.events - 1),
                            wp.uint32(3) + wp.uint32(p.physics.random_branch) * wp.uint32(65536),
                            p.sampling,
                        )
                        if rayleigh_rate > interaction[1]:
                            result.status = NUMERICAL_FAILURE
                            alive = 0
                        elif channel[0] < rayleigh_rate / interaction[1]:
                            coherent = 1
                            sampled = rayleigh_scatter_event(
                                result.energy,
                                material,
                                seed,
                                history,
                                wp.uint32(result.events - 1),
                                p.max_angle_trials,
                                p.physics,
                            )
                            cosine = sampled[0]
                            azimuth = sampled[1]
                            if int(sampled[3]) != ESCAPED:
                                result.status = ANGLE_BUDGET
                                alive = 0
                    if wp.static(coefficient_recording):
                        if (
                            coefficient_records.partial != 0
                            and coefficient_records.enabled != 0
                            and coherent == 0
                        ):
                            selected_rate = density[material] * (interaction[1] - rayleigh_rate)
                            if p.voxel_density != 0:
                                selected_rate *= voxel_density[flat]
                            if selected_rate > wp.float64(0.0):
                                result.coefficient_likelihood += coefficient_rate / selected_rate
                    if wp.static(readout):
                        if alive != 0 and readout_records.next_event != 0:
                            connection_status = next_event_readout(
                                position,
                                direction,
                                result.energy,
                                material,
                                coherent,
                                source_weight,
                                source_amplitude,
                                result.absorption_depth,
                                branch_weight,
                                result.coefficient_likelihood + result.coefficient_pathwise,
                                coefficient_records,
                                seed,
                                history,
                                wp.uint32(result.events - 1),
                                material_ids,
                                voxel_density,
                                energies,
                                absorption,
                                scattering,
                                density,
                                p,
                                readout_records,
                            )
                            if connection_status != ESCAPED:
                                result.status = connection_status
                                alive = 0
                        if alive != 0 and readout_records.roulette_threshold > wp.float64(0.0):
                            current_weight = branch_weight * wp.exp(-result.absorption_depth)
                            if current_weight < readout_records.roulette_threshold:
                                roulette = transport_random4(
                                    seed,
                                    history,
                                    wp.uint32(result.events - 1),
                                    wp.uint32(0x40000001)
                                    + wp.uint32(readout_records.branch * 65536),
                                    p.sampling,
                                )
                                if roulette[0] >= readout_records.roulette_survival:
                                    result.status = ROULETTE_TERMINATED
                                    alive = 0
                                else:
                                    branch_weight /= readout_records.roulette_survival
                    if alive != 0 and p.compton != 0 and coherent == 0:
                        before_scatter = result.energy
                        angular_domain = wp.uint32(RANDOM_NAMESPACE_BIT)
                        if wp.static(readout):
                            angular_domain += wp.uint32(readout_records.branch * 65536)
                        angular = ComptonEvent()
                        if p.compton == 2:
                            bound = bound_scatter_event(
                                result.energy,
                                material,
                                seed,
                                history,
                                wp.uint32(result.events - 1),
                                p.max_angle_trials,
                                p.physics,
                            )
                            angular.cosine = bound[0]
                            angular.azimuth = bound[1]
                            angular.energy = bound[2]
                            angular.status = int(bound[3])
                        elif p.compton_packed != 0:
                            angular = _compton_packed_event_domain(
                                result.energy,
                                seed,
                                history,
                                wp.uint32(result.events - 1),
                                p.max_angle_trials,
                                angular_domain,
                                draws[2],
                                p.sampling,
                            )
                        else:
                            angular = _compton_event_domain(
                                result.energy,
                                seed,
                                history,
                                wp.uint32(result.events - 1),
                                p.max_angle_trials,
                                angular_domain,
                                p.sampling,
                            )
                        cosine = angular.cosine
                        azimuth = angular.azimuth
                        result.energy = angular.energy
                        if p.physics.enabled != 0:
                            record_deposition(
                                p.physics,
                                flat,
                                before_scatter - result.energy,
                                source_weight,
                                source_amplitude,
                            )
                            result.deposited_energy += before_scatter - result.energy
                        if angular.status != ESCAPED:
                            result.status = ANGLE_BUDGET
                            alive = 0
                    if alive != 0:
                        direction = scatter_direction(direction, cosine, azimuth)
        if result.status == ESCAPED:
            result.pixel = detector_pixel(position, direction, p)
            if result.pixel == INVALID_PIXEL:
                result.pixel = MISS_PIXEL
                result.status = NUMERICAL_FAILURE
            if result.pixel >= 0:
                score_energy = wp.float64(1.0)
                if p.energy_score != 0:
                    score_energy = result.energy
                result.score = product3(source_weight, source_amplitude, score_energy)
                if p.continuous_absorption != 0:
                    result.score = attenuated_product4(
                        result.absorption_depth,
                        source_weight,
                        source_amplitude,
                        score_energy,
                        wp.float64(1.0),
                    )
        if wp.static(readout):
            result.score *= branch_weight
            if result.status == ESCAPED and readout_records.next_event == 0:
                record_readout(
                    readout_records,
                    result.pixel,
                    result.energy,
                    result.score,
                    int(result.real_events > 0),
                    result.score * (result.coefficient_likelihood + result.coefficient_pathwise),
                )
        if not wp.isfinite(result.score) or not wp.isfinite(result.density_score):
            result.status = NUMERICAL_FAILURE
        for column in range(active_count):
            if not wp.isfinite(result.density_scores[column]):
                result.status = NUMERICAL_FAILURE
        return result

    return implementation


_walk_unrecorded = _make_walk(False)
_walk_recorded = _make_walk(True)
_walk_coefficient = _make_walk(False, True)
_walk_readout = _make_walk(False, True, True)


@wp.func
def walk(
    initial_position: wp.vec3d,
    initial_direction: wp.vec3d,
    source_weight: wp.float64,
    source_amplitude: wp.float64,
    seed: wp.uint64,
    history: wp.uint64,
    active_material: int,
    active_materials: wp.vec4i,
    active_count: int,
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    p: Parameters,
) -> Trace:
    # Keep the legacy signature and canonical arithmetic/RNG addressing. A
    # disabled record has no array allocation or active per-history storage.
    records = VoxelRecords()
    return _walk_unrecorded(
        initial_position,
        initial_direction,
        source_weight,
        source_amplitude,
        seed,
        history,
        active_material,
        active_materials,
        active_count,
        material_ids,
        voxel_density,
        energies,
        absorption,
        scattering,
        density,
        p,
        records,
        CoefficientRecords(),
        ReadoutRecords(),
    )


@wp.func
def walk_recorded(
    initial_position: wp.vec3d,
    initial_direction: wp.vec3d,
    source_weight: wp.float64,
    source_amplitude: wp.float64,
    seed: wp.uint64,
    history: wp.uint64,
    active_material: int,
    active_materials: wp.vec4i,
    active_count: int,
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    p: Parameters,
    records: VoxelRecords,
) -> Trace:
    return _walk_recorded(
        initial_position,
        initial_direction,
        source_weight,
        source_amplitude,
        seed,
        history,
        active_material,
        active_materials,
        active_count,
        material_ids,
        voxel_density,
        energies,
        absorption,
        scattering,
        density,
        p,
        records,
        CoefficientRecords(),
        ReadoutRecords(),
    )


@wp.func
def walk_coefficient(
    initial_position: wp.vec3d,
    initial_direction: wp.vec3d,
    source_weight: wp.float64,
    source_amplitude: wp.float64,
    seed: wp.uint64,
    history: wp.uint64,
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    p: Parameters,
    records: CoefficientRecords,
) -> Trace:
    return _walk_coefficient(
        initial_position,
        initial_direction,
        source_weight,
        source_amplitude,
        seed,
        history,
        -1,
        wp.vec4i(-1),
        0,
        material_ids,
        voxel_density,
        energies,
        absorption,
        scattering,
        density,
        p,
        VoxelRecords(),
        records,
        ReadoutRecords(),
    )


@wp.func
def walk_readout(
    initial_position: wp.vec3d,
    initial_direction: wp.vec3d,
    source_weight: wp.float64,
    source_amplitude: wp.float64,
    seed: wp.uint64,
    history: wp.uint64,
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    p: Parameters,
    records: ReadoutRecords,
) -> Trace:
    active = CoefficientRecords()
    active.enabled = int(records.derivative_kind == 1)
    active.material = records.coefficient_material
    active.node = records.coefficient_node
    active.partial = records.coefficient_partial
    return _walk_readout(
        initial_position,
        initial_direction,
        source_weight,
        source_amplitude,
        seed,
        history,
        -1,
        wp.vec4i(-1),
        0,
        material_ids,
        voxel_density,
        energies,
        absorption,
        scattering,
        density,
        p,
        VoxelRecords(),
        active,
        records,
    )


@wp.kernel
def prepare_majorant_density_bounds(
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    maximum: wp.array(dtype=wp.float64),
):
    index = wp.tid()
    material = material_ids[index]
    participating = int(material >= 0 and material < maximum.shape[0])
    scale = wp.float64(1.0)
    if voxel_density.shape[0] != 0:
        scale = voxel_density[index]
    group = grouped_tally(scale, material, 1, participating)
    if group[1] > wp.float64(0.0):
        wp.atomic_max(maximum, material, group[0])


@wp.kernel
def validate_model(
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    materials: int,
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    if index < material_ids.shape[0]:
        if material_ids[index] < 0 or material_ids[index] >= materials:
            wp.atomic_or(status, 0, INPUT_MASK)
    if index < voxel_density.shape[0]:
        if not wp.isfinite(voxel_density[index]) or voxel_density[index] < wp.float64(0.0):
            wp.atomic_or(status, 0, INPUT_MASK)
    if index < absorption.shape[0]:
        a = absorption[index]
        s = scattering[index]
        if (
            not wp.isfinite(a)
            or not wp.isfinite(s)
            or a < wp.float64(0.0)
            or s < wp.float64(0.0)
            or not wp.isfinite(a + s)
        ):
            wp.atomic_or(status, 0, INPUT_MASK)


@wp.kernel
def validate_sources(
    positions: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    weights: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    parameters: Parameters,
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    if index < positions.shape[0]:
        position = positions[index]
        direction = directions[index]
        detector_distance = position[2] - parameters.detector_z
        if parameters.detector_general != 0:
            object_position = (
                parameters.inverse_rotation * position + parameters.inverse_translation
            )
            detector_distance = wp.dot(
                object_position - parameters.detector_corner, parameters.detector_normal
            )
        for axis in range(3):
            if not wp.isfinite(position[axis]) or not wp.isfinite(direction[axis]):
                wp.atomic_or(status, 0, INPUT_MASK)
        if (
            wp.abs(wp.dot(direction, direction) - wp.float64(1.0)) > wp.float64(1.0e-12)
            or not wp.isfinite(weights[index])
            or weights[index] < wp.float64(0.0)
            or detector_distance >= wp.float64(0.0)
        ):
            wp.atomic_or(status, 0, INPUT_MASK)
    if index < density.shape[0]:
        if not wp.isfinite(density[index]) or density[index] <= wp.float64(0.0):
            wp.atomic_or(status, 0, INPUT_MASK)


@wp.kernel
def validate_coefficient_parameter(
    scattering: wp.array(dtype=wp.float64),
    p: Parameters,
    material: int,
    node: int,
    partial: int,
    status: wp.array(dtype=wp.int32),
):
    if partial != 0 and p.physics.enabled != 0:
        index = material * p.energy_bins + node
        if scattering[index] > wp.float64(0.0) and p.physics.rayleigh[index] >= scattering[index]:
            # The active aggregate node would open a zero incoherent channel.
            # A score-function estimator needs fixed conditional support.
            wp.atomic_or(status, 0, INPUT_MASK)


@wp.kernel
def validate_physical_coefficients(
    scattering: wp.array(dtype=wp.float64),
    physical: PhysicalParameters,
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    if physical.rayleigh[index] > scattering[index]:
        wp.atomic_or(status, 0, INPUT_MASK)


@wp.kernel
def validate_majorant(
    density: wp.array(dtype=wp.float64),
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    p: Parameters,
    status: wp.array(dtype=wp.int32),
):
    voxel = wp.tid()
    material = material_ids[voxel]
    for node in range(p.energy_bins):
        index = material * p.energy_bins + node
        rate = density[material] * (absorption[index] + scattering[index])
        if p.voxel_density != 0:
            rate = product3(
                density[material],
                voxel_density[voxel],
                absorption[index] + scattering[index],
            )
        if not wp.isfinite(rate) or rate >= p.physics.majorant[node]:
            wp.atomic_or(status, 0, INPUT_MASK)


@wp.kernel
def trace_energy_balance(
    positions: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    weights: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    parameters: Parameters,
    seed: wp.uint64,
    first_history: wp.uint64,
    source_amplitude: wp.float64,
    out_pixel: wp.array(dtype=wp.int32),
    out_score: wp.array(dtype=wp.float64),
    out_energy: wp.array(dtype=wp.float64),
    out_events: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
    out_initial_energy: wp.array(dtype=wp.float64),
    out_deposited_energy: wp.array(dtype=wp.float64),
    out_spectrum_index: wp.array(dtype=wp.int32),
    out_local_deposition: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    p = parameters
    p.physics.record_deposition = 1
    p.physics.deposition = out_local_deposition
    result = walk(
        positions[index],
        directions[index],
        weights[index],
        source_amplitude,
        seed,
        first_history + wp.uint64(index),
        -1,
        wp.vec4i(-1),
        0,
        material_ids,
        voxel_density,
        energies,
        absorption,
        scattering,
        density,
        p,
    )
    out_pixel[index] = result.pixel
    out_score[index] = result.score
    out_energy[index] = result.energy
    out_events[index] = result.events
    out_status[index] = result.status
    out_initial_energy[index] = result.initial_energy
    out_deposited_energy[index] = result.deposited_energy
    out_spectrum_index[index] = result.spectrum_index
    if result.status >= EVENT_BUDGET:
        wp.atomic_or(status, 0, 1 << result.status)
    if not wp.isfinite(result.deposited_energy):
        wp.atomic_or(status, 0, NUMERICAL_MASK)


@wp.kernel
def validate_deposition(values: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32)):
    index = wp.tid()
    if not wp.isfinite(values[index]) or values[index] < wp.float64(0.0):
        wp.atomic_or(status, 0, NUMERICAL_MASK)


@wp.kernel
def trace_histories(
    positions: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    weights: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    parameters: Parameters,
    seed: wp.uint64,
    first_history: wp.uint64,
    source_amplitude: wp.float64,
    out_pixel: wp.array(dtype=wp.int32),
    out_score: wp.array(dtype=wp.float64),
    out_energy: wp.array(dtype=wp.float64),
    out_events: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    result = walk(
        positions[index],
        directions[index],
        weights[index],
        source_amplitude,
        seed,
        first_history + wp.uint64(index),
        SOURCE_AMPLITUDE,
        wp.vec4i(SOURCE_AMPLITUDE),
        0,
        material_ids,
        voxel_density,
        energies,
        absorption,
        scattering,
        density,
        parameters,
    )
    out_pixel[index] = result.pixel
    out_score[index] = result.score
    if not wp.isfinite(out_score[index]):
        wp.atomic_or(status, 0, NUMERICAL_MASK)
    out_energy[index] = result.energy
    out_events[index] = result.events
    out_status[index] = result.status
    if result.status >= EVENT_BUDGET:
        wp.atomic_or(status, 0, 1 << result.status)


@wp.func
def _density_derivative_contribution(
    weight: wp.float64,
    amplitude: wp.float64,
    score_energy: wp.float64,
    factor: wp.float64,
    absorption_depth: wp.float64,
    continuous_absorption: int,
) -> wp.float64:
    derivative = product4(weight, amplitude, score_energy, factor)
    if continuous_absorption != 0:
        derivative = attenuated_product4(absorption_depth, weight, amplitude, score_energy, factor)
    return derivative


@wp.func
def _derivative_factors(
    density_score: wp.float64,
    active_material: int,
    source_amplitude: wp.float64,
) -> tuple[wp.float64, wp.float64]:
    factor = density_score
    amplitude = source_amplitude
    if active_material == SOURCE_AMPLITUDE:
        factor = wp.float64(1.0)
        amplitude = wp.float64(1.0)
    elif active_material == LOG_SOURCE_AMPLITUDE:
        factor = wp.float64(1.0)
    return factor, amplitude


@wp.kernel
def derivative_histories(
    positions: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    weights: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    parameters: Parameters,
    seed: wp.uint64,
    first_history: wp.uint64,
    active_material: int,
    source_amplitude: wp.float64,
    out_pixel: wp.array(dtype=wp.int32),
    out_derivative: wp.array(dtype=wp.float64),
    out_status: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    amplitude = source_amplitude
    base_weight = weights[index]
    if active_material >= 0:
        # A density derivative is formed directly below. An unused primal
        # overflow/underflow must not determine its representability.
        base_weight = wp.float64(0.0)
    if active_material == SOURCE_AMPLITUDE:
        amplitude = wp.float64(1.0)
    result = walk(
        positions[index],
        directions[index],
        base_weight,
        amplitude,
        seed,
        first_history + wp.uint64(index),
        active_material,
        wp.vec4i(SOURCE_AMPLITUDE),
        0,
        material_ids,
        voxel_density,
        energies,
        absorption,
        scattering,
        density,
        parameters,
    )
    out_pixel[index] = result.pixel
    derivative = wp.float64(0.0)
    if active_material >= 0 and result.pixel >= 0:
        score_energy = wp.float64(1.0)
        if parameters.energy_score != 0:
            score_energy = result.energy
        derivative = _density_derivative_contribution(
            weights[index],
            source_amplitude,
            score_energy,
            result.density_score,
            result.absorption_depth,
            parameters.continuous_absorption,
        )
    if active_material == SOURCE_AMPLITUDE:
        # d(a * base_score)/da, including at a=0; never divide by amplitude.
        derivative = result.score
    if active_material == LOG_SOURCE_AMPLITUDE:
        derivative = result.score
    out_derivative[index] = derivative
    out_status[index] = result.status
    if result.status >= EVENT_BUDGET:
        wp.atomic_or(status, 0, 1 << result.status)
    if not wp.isfinite(derivative):
        wp.atomic_or(status, 0, NUMERICAL_MASK)


@wp.kernel
def derivative_columns(
    positions: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    weights: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    parameters: Parameters,
    seed: wp.uint64,
    first_history: wp.uint64,
    active_materials: wp.vec4i,
    active_count: int,
    source_amplitude: wp.float64,
    out_pixel: wp.array(dtype=wp.int32),
    out_derivatives: wp.array(dtype=wp.float64),
    out_status: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    history = wp.tid()
    # Do not form an unused primal. Each column uses original factors, preserving
    # derivatives whose range differs from the forward score.
    result = walk(
        positions[history],
        directions[history],
        wp.float64(0.0),
        wp.float64(1.0),
        seed,
        first_history + wp.uint64(history),
        SOURCE_AMPLITUDE,
        active_materials,
        active_count,
        material_ids,
        voxel_density,
        energies,
        absorption,
        scattering,
        density,
        parameters,
    )
    out_pixel[history] = result.pixel
    out_status[history] = result.status
    if result.status >= EVENT_BUDGET:
        wp.atomic_or(status, 0, 1 << result.status)
    for column in range(active_count):
        derivative = wp.float64(0.0)
        if result.pixel >= 0:
            score_energy = wp.float64(1.0)
            if parameters.energy_score != 0:
                score_energy = result.energy
            factor, amplitude = _derivative_factors(
                result.density_scores[column],
                active_materials[column],
                source_amplitude,
            )
            derivative = _density_derivative_contribution(
                weights[history],
                amplitude,
                score_energy,
                factor,
                result.absorption_depth,
                parameters.continuous_absorption,
            )
        out_derivatives[column * positions.shape[0] + history] = derivative
        if not wp.isfinite(derivative):
            wp.atomic_or(status, 0, NUMERICAL_MASK)


@wp.func
def _centred_residual(
    value: wp.float64,
    average: wp.float64,
    scale: wp.array(dtype=wp.float64),
    target: int,
    residual: wp.float64,
) -> wp.float64:
    if (value >= wp.float64(0.0)) == (average >= wp.float64(0.0)):
        residual = (value - average) / scale[target]
    else:
        residual = value / scale[target] - average / scale[target]
    return residual


@wp.func
def _centred_tally_contribution(
    history: int,
    target: int,
    score: wp.array(dtype=wp.float64),
    mean: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
) -> tuple[wp.float64, int]:
    """Shared guarded centred square before the selected atomic reduction."""
    participating = int(0)
    contribution = wp.float64(0.0)
    if target >= 0 and target < mean.shape[0]:
        residual = wp.float64(0.0)
        if scale[target] > wp.float64(0.0):
            value = score[history]
            average = mean[target]
            residual = _centred_residual(value, average, scale, target, residual)
        contribution = residual * residual
        if not wp.isfinite(contribution):
            wp.atomic_or(status, 0, NUMERICAL_MASK)
        else:
            participating = 1
    return contribution, participating


@wp.kernel
def centred_moments(
    pixel: wp.array(dtype=wp.int32),
    score: wp.array(dtype=wp.float64),
    mean: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64),
    out_variance: wp.array(dtype=wp.float64),
    out_hits: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    history = wp.tid()
    target = pixel[history]
    contribution, participating = _centred_tally_contribution(
        history, target, score, mean, scale, status
    )
    aggregate = grouped_tally(contribution, target, 0, participating)
    if aggregate[1] > wp.float64(0.0):
        wp.atomic_add(out_variance, target, aggregate[0])
        wp.atomic_add(out_hits, target, int(aggregate[1]))


@wp.kernel
def centred_moments_plain(
    pixel: wp.array(dtype=wp.int32),
    score: wp.array(dtype=wp.float64),
    mean: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64),
    out_variance: wp.array(dtype=wp.float64),
    out_hits: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    history = wp.tid()
    target = pixel[history]
    contribution, participating = _centred_tally_contribution(
        history, target, score, mean, scale, status
    )
    if participating != 0:
        wp.atomic_add(out_variance, target, contribution)
        wp.atomic_add(out_hits, target, 1)


@wp.kernel
def finish_variance(
    mean: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64),
    hits: wp.array(dtype=wp.int32),
    count: int,
    out_variance: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    pixel = wp.tid()
    n = wp.float64(count)
    missing = count - hits[pixel]
    centred = out_variance[pixel]
    if missing > 0 and scale[pixel] > wp.float64(0.0):
        residual = mean[pixel] / scale[pixel]
        centred += wp.float64(missing) * residual * residual
    # Accumulate normalised squares before restoring their exponent. Squaring
    # each tiny history first can turn a representable aggregate variance to zero.
    sigma = scale[pixel] * wp.sqrt(centred / n / (n - wp.float64(1.0)))
    variance = sigma * sigma
    out_variance[pixel] = variance
    if missing < 0 or not wp.isfinite(variance):
        wp.atomic_or(status, 0, NUMERICAL_MASK)


@wp.kernel
def independent_product(
    mean_a: wp.array(dtype=wp.float64),
    other_b: wp.array(dtype=wp.float64),
    observed: wp.array(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    loss_mode: int,
    out_components: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    pixel = wp.tid()
    a = mean_a[pixel]
    b = other_b[pixel]
    y = observed[pixel]
    w = weights[pixel]
    if (
        not wp.isfinite(a)
        or not wp.isfinite(b)
        or not wp.isfinite(y)
        or not wp.isfinite(w)
        or w < wp.float64(0.0)
    ):
        wp.atomic_or(status, 0, INPUT_MASK)
    value = w * (a - y) * b
    if loss_mode != 0:
        value = wp.float64(0.5) * w * (a - y) * (b - y)
    out_components[pixel] = value
    if not wp.isfinite(value):
        wp.atomic_or(status, 0, NUMERICAL_MASK)


@wp.kernel
def sample_parallel_source(
    lower: wp.vec3d,
    extent: wp.vec2d,
    direction: wp.vec3d,
    weight: wp.float64,
    seed: wp.uint64,
    first_history: wp.uint64,
    out_position: wp.array(dtype=wp.vec3d),
    out_direction: wp.array(dtype=wp.vec3d),
    out_weight: wp.array(dtype=wp.float64),
):
    index = wp.tid()
    # The high event bit separates source draws from every supported collision.
    draw = random4(
        seed,
        first_history + wp.uint64(index),
        wp.uint32(RANDOM_NAMESPACE_BIT),
        wp.uint32(0),
    )
    out_position[index] = lower + wp.vec3d(
        extent[0] * draw[0], extent[1] * draw[1], wp.float64(0.0)
    )
    out_direction[index] = direction
    out_weight[index] = weight


@wp.kernel
def sample_point_source(
    source: wp.vec3d,
    corner: wp.vec3d,
    u: wp.vec3d,
    v: wp.vec3d,
    extent: wp.vec2d,
    weight: wp.float64,
    seed: wp.uint64,
    first_history: wp.uint64,
    out_position: wp.array(dtype=wp.vec3d),
    out_direction: wp.array(dtype=wp.vec3d),
    out_weight: wp.array(dtype=wp.float64),
):
    index = wp.tid()
    draw = random4(
        seed,
        first_history + wp.uint64(index),
        wp.uint32(RANDOM_NAMESPACE_BIT),
        wp.uint32(0),
    )
    endpoint = corner + extent[0] * draw[0] * u + extent[1] * draw[1] * v
    out_position[index] = source
    delta = endpoint - source
    scale = wp.max(wp.abs(delta[0]), wp.max(wp.abs(delta[1]), wp.abs(delta[2])))
    out_direction[index] = wp.normalize(delta / scale)
    out_weight[index] = weight


@wp.kernel
def update_density_chart(
    parameters: wp.array(dtype=wp.float64),
    material_parameter: wp.array(dtype=wp.int32),
    base_density: wp.array(dtype=wp.float64),
    out_density: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    material = wp.tid()
    active = material_parameter[material]
    value = base_density[material]
    if active >= 0:
        value = wp.exp(parameters[active])
    out_density[material] = value
    if not wp.isfinite(value) or value <= wp.float64(0.0):
        wp.atomic_or(status, 0, NUMERICAL_MASK)


@wp.func
def valid_history(
    target: int,
    value: wp.float64,
    terminal: int,
    pixels: int,
    status: wp.array(dtype=wp.int32),
) -> bool:
    """Validate one original history with the shared terminal-status contract."""
    if terminal >= EVENT_BUDGET and terminal <= ANGLE_BUDGET:
        wp.atomic_or(status, 0, 1 << terminal)
        return False
    if (
        target < MISS_PIXEL
        or target >= pixels
        or not wp.isfinite(value)
        or terminal < ESCAPED
        or terminal > ANGLE_BUDGET
        or (target == MISS_PIXEL and value != wp.float64(0.0))
        or (terminal == ABSORBED and target != MISS_PIXEL)
    ):
        wp.atomic_or(status, 0, INPUT_MASK)
        return False
    return target >= 0


@wp.kernel
def find_tally_scale(
    pixel: wp.array(dtype=wp.int32),
    score: wp.array(dtype=wp.float64),
    history_status: wp.array(dtype=wp.int32),
    pixels: int,
    out_scale: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    history = wp.tid()
    target = pixel[history]
    value = score[history]
    participating = int(0)
    if valid_history(target, value, history_status[history], pixels, status):
        participating = 1
    aggregate = grouped_tally(wp.abs(value), target, 1, participating)
    if aggregate[1] > wp.float64(0.0):
        wp.atomic_max(out_scale, target, aggregate[0])


@wp.kernel
def find_tally_scale_plain(
    pixel: wp.array(dtype=wp.int32),
    score: wp.array(dtype=wp.float64),
    history_status: wp.array(dtype=wp.int32),
    pixels: int,
    out_scale: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    history = wp.tid()
    target = pixel[history]
    value = score[history]
    participating = int(0)
    if valid_history(target, value, history_status[history], pixels, status):
        participating = 1
    if participating != 0:
        wp.atomic_max(out_scale, target, wp.abs(value))


@wp.func
def _normalised_score_contribution(
    history: int,
    target: int,
    score: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
) -> tuple[wp.float64, int]:
    contribution = score[history] / scale[target]
    participating = int(0)
    if not wp.isfinite(contribution):
        wp.atomic_or(status, 0, NUMERICAL_MASK)
    else:
        participating = 1
    return contribution, participating


@wp.func
def _normalised_tally_contribution(
    history: int,
    target: int,
    score: wp.array(dtype=wp.float64),
    history_status: wp.array(dtype=wp.int32),
    pixels: int,
    scale: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
) -> tuple[wp.float64, int]:
    """Prepare a normalised score and preserve the tally status guard."""
    participating = int(0)
    contribution = wp.float64(0.0)
    if (
        history_status[history] <= ABSORBED
        and history_status[history] >= ESCAPED
        and target >= 0
        and target < pixels
        and scale[target] > wp.float64(0.0)
    ):
        contribution, participating = _normalised_score_contribution(
            history, target, score, scale, status
        )
    return contribution, participating


@wp.kernel
def tally_sum(
    pixel: wp.array(dtype=wp.int32),
    score: wp.array(dtype=wp.float64),
    history_status: wp.array(dtype=wp.int32),
    pixels: int,
    scale: wp.array(dtype=wp.float64),
    out_sum: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    history = wp.tid()
    target = pixel[history]
    contribution, participating = _normalised_tally_contribution(
        history, target, score, history_status, pixels, scale, status
    )
    aggregate = grouped_tally(contribution, target, 0, participating)
    if aggregate[1] > wp.float64(0.0):
        wp.atomic_add(out_sum, target, aggregate[0])


@wp.kernel
def tally_sum_plain(
    pixel: wp.array(dtype=wp.int32),
    score: wp.array(dtype=wp.float64),
    history_status: wp.array(dtype=wp.int32),
    pixels: int,
    scale: wp.array(dtype=wp.float64),
    out_sum: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    history = wp.tid()
    target = pixel[history]
    contribution, participating = _normalised_tally_contribution(
        history, target, score, history_status, pixels, scale, status
    )
    if participating != 0:
        wp.atomic_add(out_sum, target, contribution)


@wp.kernel
def finish_mean(
    sums: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64),
    count: int,
    out_mean: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    pixel = wp.tid()
    # Scale after dividing the bounded sum: neither a huge raw sum nor a
    # subnormal score divided prematurely by the history count is required.
    value = scale[pixel] * (sums[pixel] / wp.float64(count))
    out_mean[pixel] = value
    if not wp.isfinite(value):
        wp.atomic_or(status, 0, NUMERICAL_MASK)


@wp.kernel
def validate_measurement(
    observation: wp.array(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    pixel = wp.tid()
    if (
        not wp.isfinite(observation[pixel])
        or not wp.isfinite(weights[pixel])
        or weights[pixel] < wp.float64(0.0)
    ):
        wp.atomic_or(status, 0, INPUT_MASK)


@wp.kernel
def validate_tally_bound(bound: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32)):
    index = wp.tid()
    if not wp.isfinite(bound[index]) or bound[index] <= wp.float64(0.0):
        wp.atomic_or(status, 0, INPUT_MASK)


@wp.kernel
def trace_tally(
    positions: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    weights: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    parameters: Parameters,
    seed: wp.uint64,
    first_history: wp.uint64,
    source_amplitude: wp.float64,
    bound: wp.array(dtype=wp.float64),
    grouped: int,
    sums: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    history = wp.tid()
    result = walk(
        positions[history],
        directions[history],
        weights[history],
        source_amplitude,
        seed,
        first_history + wp.uint64(history),
        SOURCE_AMPLITUDE,
        wp.vec4i(SOURCE_AMPLITUDE),
        0,
        material_ids,
        voxel_density,
        energies,
        absorption,
        scattering,
        density,
        parameters,
    )
    if result.status >= EVENT_BUDGET:
        wp.atomic_or(status, 0, 1 << result.status)
    participating = int(0)
    contribution = wp.float64(0.0)
    if result.status == ESCAPED and result.pixel >= 0:
        contribution = result.score / bound[result.pixel]
        if result.score > bound[result.pixel]:
            wp.atomic_or(status, 0, INPUT_MASK)
        elif not wp.isfinite(contribution) or (
            result.score != wp.float64(0.0) and contribution == wp.float64(0.0)
        ):
            wp.atomic_or(status, 0, NUMERICAL_MASK)
        else:
            participating = 1
    if grouped != 0:
        aggregate = grouped_tally(contribution, result.pixel, 0, participating)
        if aggregate[1] > wp.float64(0.0):
            wp.atomic_add(sums, result.pixel, aggregate[0])
    elif participating != 0:
        wp.atomic_add(sums, result.pixel, contribution)


@wp.kernel
def finish_bounded_mean(
    sums: wp.array(dtype=wp.float64),
    bound: wp.array(dtype=wp.float64),
    count: int,
    mean: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    pixel = wp.tid()
    average = sums[pixel] / wp.float64(count)
    value = bound[pixel] * average
    mean[pixel] = value
    if not wp.isfinite(value) or (sums[pixel] != wp.float64(0.0) and average == wp.float64(0.0)):
        wp.atomic_or(status, 0, NUMERICAL_MASK)
