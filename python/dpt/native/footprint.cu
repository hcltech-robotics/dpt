// Voxel gathering of the realised FP32 midpoint matrix, without filtered weights.
#include "common.cuh"

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cmath>

using namespace dpt_native;

namespace {

struct FloatInterval { float lower, upper; };
struct FloatSupport { float3 lower, upper; };

struct PreparedRay {
    Ray32 ray;
    float lower, step, span, factor;
    float3 point_error;
    FloatInterval inverse_direction[3], inverse_step;
    int fast_bounds;
};

struct Culling {
    double3 origin, column, row, denominator;
    double point_error[3];
    double column_error, row_error;
    int fallback;
    FloatInterval coefficients[3][3];
    int fast_bounds;
};

struct Interval { double lower, upper; };
struct Support { double3 lower, upper; };
struct Bounds { int first_column, last_column, first_row, last_row; };

constexpr size_t ray_offset = (sizeof(Culling) + 255) & ~size_t(255);
constexpr double position_error = 8.0 * FLT_EPSILON;

__device__ double component(double3 value, int axis) {
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

__device__ float component(float3 value, int axis) {
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

__device__ int component(int3 value, int axis) {
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

__device__ void set_component(double3& value, int axis, double scalar) {
    if (axis == 0) value.x = scalar;
    else if (axis == 1) value.y = scalar;
    else value.z = scalar;
}

__device__ void set_component(float3& value, int axis, float scalar) {
    if (axis == 0) value.x = scalar;
    else if (axis == 1) value.y = scalar;
    else value.z = scalar;
}

__device__ double3 widen(float3 value) {
    return make_double3(value.x, value.y, value.z);
}

__device__ double3 cross_product(double3 a, double3 b) {
    return make_double3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}

__device__ double3 grid_direction(const double* pose, const double* value, Config config) {
    const double object[3] = {
        pose[0]*value[0] + pose[3]*value[1] + pose[6]*value[2],
        pose[1]*value[0] + pose[4]*value[1] + pose[7]*value[2],
        pose[2]*value[0] + pose[5]*value[1] + pose[8]*value[2]};
    return make_double3(
        config.object_to_grid[0]*object[0] + config.object_to_grid[1]*object[1] + config.object_to_grid[2]*object[2],
        config.object_to_grid[3]*object[0] + config.object_to_grid[4]*object[1] + config.object_to_grid[5]*object[2],
        config.object_to_grid[6]*object[0] + config.object_to_grid[7]*object[1] + config.object_to_grid[8]*object[2]);
}

// Only nonnegative finite values are stored, so unsigned bit ordering applies.
__device__ void maximum(double* address, double value) {
    atomicMax(reinterpret_cast<unsigned long long*>(address), __double_as_longlong(value));
}

__device__ Interval enclosed_dot(double3 a, double3 b) {
    Interval result{0.0, 0.0};
    for (int axis = 0; axis < 3; ++axis) {
        const double x = component(a, axis), y = component(b, axis);
        result.lower = __dadd_rd(result.lower, __dmul_rd(x, y));
        result.upper = __dadd_ru(result.upper, __dmul_ru(x, y));
    }
    return result;
}

__device__ Interval enclosed_box_dot(double3 coefficient, Support box, double3 origin) {
    Interval result{0.0, 0.0};
    for (int axis = 0; axis < 3; ++axis) {
        const double c = component(coefficient, axis);
        const double a = __dadd_rd(component(box.lower, axis), -component(origin, axis));
        const double b = __dadd_ru(component(box.upper, axis), -component(origin, axis));
        result.lower = __dadd_rd(result.lower, __dmul_rd(c, c >= 0.0 ? a : b));
        result.upper = __dadd_ru(result.upper, __dmul_ru(c, c >= 0.0 ? b : a));
    }
    return result;
}

__device__ Interval enclosed_quotient(Interval a, Interval b) {
    if (!isfinite(a.lower) || !isfinite(a.upper) || !isfinite(b.lower)
        || !isfinite(b.upper) || !(b.lower > 0.0 || b.upper < 0.0)) {
        return {NAN, NAN};
    }
    if (b.upper < 0.0) {
        a = {-a.upper, -a.lower};
        b = {-b.upper, -b.lower};
    }
    return {__ddiv_rd(a.lower, a.lower >= 0.0 ? b.upper : b.lower),
            __ddiv_ru(a.upper, a.upper >= 0.0 ? b.lower : b.upper)};
}

__device__ bool finite_interval(Interval value) {
    return isfinite(value.lower) && isfinite(value.upper);
}

__device__ bool finite_interval(FloatInterval value) {
    return isfinite(value.lower) && isfinite(value.upper);
}

// Directed operations enclose the real product of two closed intervals. Most
// projective coefficients and source-relative voxel boxes have fixed signs.
__device__ FloatInterval enclosed_product(FloatInterval a, FloatInterval b) {
    if (a.lower >= 0.0f) {
        if (b.lower >= 0.0f) return {__fmul_rd(a.lower, b.lower), __fmul_ru(a.upper, b.upper)};
        if (b.upper <= 0.0f) return {__fmul_rd(a.upper, b.lower), __fmul_ru(a.lower, b.upper)};
        return {__fmul_rd(a.upper, b.lower), __fmul_ru(a.upper, b.upper)};
    }
    if (a.upper <= 0.0f) {
        if (b.lower >= 0.0f) return {__fmul_rd(a.lower, b.upper), __fmul_ru(a.upper, b.lower)};
        if (b.upper <= 0.0f) return {__fmul_rd(a.upper, b.upper), __fmul_ru(a.lower, b.lower)};
        return {__fmul_rd(a.lower, b.upper), __fmul_ru(a.lower, b.lower)};
    }
    if (b.lower >= 0.0f) return {__fmul_rd(a.lower, b.upper), __fmul_ru(a.upper, b.upper)};
    if (b.upper <= 0.0f) return {__fmul_rd(a.upper, b.lower), __fmul_ru(a.lower, b.lower)};
    return {fminf(__fmul_rd(a.lower, b.upper), __fmul_rd(a.upper, b.lower)),
            fmaxf(__fmul_ru(a.lower, b.lower), __fmul_ru(a.upper, b.upper))};
}

__device__ FloatInterval enclosed_quotient(FloatInterval a, FloatInterval b) {
    if (!finite_interval(a) || !finite_interval(b)
        || !(b.lower > 0.0f || b.upper < 0.0f)) return {NAN, NAN};
    if (b.upper < 0.0f) {
        a = {-a.upper, -a.lower};
        b = {-b.upper, -b.lower};
    }
    return {__fdiv_rd(a.lower, a.lower >= 0.0f ? b.upper : b.lower),
            __fdiv_ru(a.upper, a.upper >= 0.0f ? b.lower : b.upper)};
}

__device__ FloatInterval enclosed_box_dot(const FloatInterval* coefficients, FloatSupport box) {
    FloatInterval result{0.0f, 0.0f};
    for (int axis = 0; axis < 3; ++axis) {
        const FloatInterval term = enclosed_product(coefficients[axis],
            {component(box.lower, axis), component(box.upper, axis)});
        result.lower = __fadd_rd(result.lower, term.lower);
        result.upper = __fadd_ru(result.upper, term.upper);
    }
    return result;
}

__global__ void initialise_culling(Config config, const double* pose,
                                  const double* sources, const double* endpoints,
                                  Culling* culling, int* status) {
    *culling = {};
    if (config.ray_bundle || config.width <= 0 || config.pixels % config.width != 0) {
        culling->fallback = 1;
        return;
    }
    const Ray32 first = make_ray(config, pose, 0, sources, endpoints, status);
    culling->origin = widen(first.origin);
    const double3 base = widen(first.direction);
    const double3 column = grid_direction(pose, config.column_step, config);
    const double3 row = grid_direction(pose, config.row_step, config);
    culling->column = cross_product(row, base);
    culling->row = cross_product(base, column);
    culling->denominator = cross_product(column, row);
    culling->fast_bounds = 1;
    const double3 coefficients[3] = {culling->column, culling->row, culling->denominator};
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        for (int axis = 0; axis < 3; ++axis) {
            const double value = component(coefficients[coordinate], axis);
            const FloatInterval enclosed{__double2float_rd(value), __double2float_ru(value)};
            culling->coefficients[coordinate][axis] = enclosed;
            if (!finite_interval(enclosed)) culling->fast_bounds = 0;
        }
    }
    // These stored coefficients need not reproduce ideal geometry. The next
    // kernel encloses every actual ray's projection and measures its discrepancy.
}

template <bool DoubleSeed>
__device__ void prepare_ray(Config config, const double* pose, const double* sources,
                           const double* endpoints, const void* seeds, int pixel,
                           PreparedRay* prepared, Culling* culling, int* status,
                           double* maxima) {
    PreparedRay result{};
    result.ray = make_ray(config, pose, pixel, sources, endpoints, status);
    if (!result.ray.valid) {
        atomicOr(status, 2);
        prepared[pixel] = result;
        return;
    }
    const Interval32 interval = clip_ray(result.ray, config);
    result.lower = interval.lower;
    result.span = __fsub_rn(interval.upper, interval.lower);
    const float seed = DoubleSeed ? float(static_cast<const double*>(seeds)[pixel])
                                  : static_cast<const float*>(seeds)[pixel];
    if (!isfinite(seed)) atomicOr(status, 2);
    if (result.span > 0.0f) {
        result.step = __fdiv_rn(result.span, float(config.samples));
        if (seed != 0.0f) {
            result.factor = __fmul_rn(seed, __fmul_rn(result.ray.length, result.step));
        }
        const float first = fmaf(0.5f, result.step, result.lower);
        const float last = fmaf(__fadd_rn(float(config.samples - 1), 0.5f),
                                result.step, result.lower);
        // Rounded sample positions and coordinates are monotone in sample
        // index. Finite endpoints certify every interior node without a scan.
        if (!finite_point(sample_point(result.ray, first))
            || !finite_point(sample_point(result.ray, last))) atomicOr(status, 2);
    }
    result.point_error = make_float3(
        4.0f * FLT_EPSILON * (fabsf(result.ray.origin.x) + fabsf(result.ray.direction.x) + 1.0f),
        4.0f * FLT_EPSILON * (fabsf(result.ray.origin.y) + fabsf(result.ray.direction.y) + 1.0f),
        4.0f * FLT_EPSILON * (fabsf(result.ray.origin.z) + fabsf(result.ray.direction.z) + 1.0f));
    // Float(samples - 1) must be exact before the final ceil/floor conversions.
    result.fast_bounds = config.samples < (1 << 24);
    for (int axis = 0; axis < 3; ++axis) {
        const float direction = component(result.ray.direction, axis);
        if (direction != 0.0f) {
            result.inverse_direction[axis] = {__fdiv_rd(1.0f, direction), __fdiv_ru(1.0f, direction)};
            if (!finite_interval(result.inverse_direction[axis])) result.fast_bounds = 0;
        }
    }
    if (result.step > 0.0f) {
        result.inverse_step = {__fdiv_rd(1.0f, result.step), __fdiv_ru(1.0f, result.step)};
        if (!finite_interval(result.inverse_step)) result.fast_bounds = 0;
    }
    prepared[pixel] = result;
    if (!config.ray_bundle && config.width > 0 && config.pixels % config.width == 0) {
        for (int axis = 0; axis < 3; ++axis) {
            const double radius = component(result.point_error, axis);
            if (!isfinite(radius) || component(result.ray.origin, axis)
                != component(culling->origin, axis)) atomicExch(&culling->fallback, 1);
            else maxima[axis] = radius;
        }
        const double3 direction = widen(result.ray.direction);
        const Interval denominator = enclosed_dot(culling->denominator, direction);
        const Interval column = enclosed_quotient(enclosed_dot(culling->column, direction), denominator);
        const Interval row = enclosed_quotient(enclosed_dot(culling->row, direction), denominator);
        if (!finite_interval(column) || !finite_interval(row)) {
            atomicExch(&culling->fallback, 1);
        } else {
            const double c = pixel % config.width, r = pixel / config.width;
            maxima[3] = fmax(fabs(__dadd_rd(column.lower, -c)), fabs(__dadd_ru(column.upper, -c)));
            maxima[4] = fmax(fabs(__dadd_rd(row.lower, -r)), fabs(__dadd_ru(row.upper, -r)));
        }
    }
}

template <bool DoubleSeed>
__global__ void prepare_rays(Config config, const double* pose, const double* sources,
                             const double* endpoints, const void* seeds,
                             PreparedRay* prepared, Culling* culling, int* status) {
    const unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
    double maxima[5] = {};
    if (thread < unsigned(config.pixels)) {
        prepare_ray<DoubleSeed>(config, pose, sources, endpoints, seeds, int(thread),
                                prepared, culling, status, maxima);
    }
    // Every lane participates, including partial blocks and invalid rays. Five
    // global maxima per block replace five contended atomics per detector pixel.
    __shared__ double warp_maxima[5][8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int metric = 0; metric < 5; ++metric) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            maxima[metric] = fmax(maxima[metric], __shfl_down_sync(0xffffffff, maxima[metric], offset));
        }
        if (lane == 0) warp_maxima[metric][warp] = maxima[metric];
    }
    __syncthreads();
    if (warp == 0) {
        for (int metric = 0; metric < 5; ++metric) {
            double value = lane < 8 ? warp_maxima[metric][lane] : 0.0;
            for (int offset = 16; offset > 0; offset >>= 1) {
                value = fmax(value, __shfl_down_sync(0xffffffff, value, offset));
            }
            if (lane == 0) {
                double* destination = metric < 3 ? culling->point_error + metric
                    : (metric == 3 ? &culling->column_error : &culling->row_error);
                maximum(destination, value);
            }
        }
    }
}

__device__ FloatSupport voxel_support_float(int3 voxel, Config config) {
    FloatSupport result;
    const int sizes[3] = {config.nx, config.ny, config.nz};
    for (int axis = 0; axis < 3; ++axis) {
        const int index = component(voxel, axis), size = sizes[axis];
        // Keep selected zero-weight corners too: an infinite factor times zero
        // must retain the same numerical diagnostic as the existing transpose.
        set_component(result.lower, axis, index <= 1 ? -0.5f : __int2float_rd(index - 1));
        set_component(result.upper, axis, index >= size - 2
            ? __fsub_ru(__int2float_ru(size), 0.5f) : __int2float_ru(index + 1));
    }
    return result;
}

__device__ Bounds pixel_bounds(FloatSupport fast_support, Config config, const Culling& culling) {
    if (config.ray_bundle || culling.fallback) return {0, config.pixels, 0, 1};
    Bounds result{0, config.width, 0, config.pixels / config.width};
    if (culling.fast_bounds && config.width <= (1 << 24) && result.last_row <= (1 << 24)) {
        FloatSupport relative = fast_support;
        bool finite = true;
        for (int axis = 0; axis < 3; ++axis) {
            const float error = __double2float_ru(culling.point_error[axis]);
            // The source was stored from Ray32, so its float conversion is exact.
            const float origin = float(component(culling.origin, axis));
            const float lower = __fsub_rd(__fsub_rd(component(relative.lower, axis), error), origin);
            const float upper = __fsub_ru(__fadd_ru(component(relative.upper, axis), error), origin);
            set_component(relative.lower, axis, lower);
            set_component(relative.upper, axis, upper);
            finite = finite && isfinite(lower) && isfinite(upper);
        }
        if (finite) {
            const FloatInterval denominator = enclosed_box_dot(culling.coefficients[2], relative);
            FloatInterval column = enclosed_quotient(enclosed_box_dot(culling.coefficients[0], relative), denominator);
            FloatInterval row = enclosed_quotient(enclosed_box_dot(culling.coefficients[1], relative), denominator);
            const float column_error = __double2float_ru(culling.column_error);
            const float row_error = __double2float_ru(culling.row_error);
            column = {__fsub_rd(column.lower, column_error), __fadd_ru(column.upper, column_error)};
            row = {__fsub_rd(row.lower, row_error), __fadd_ru(row.upper, row_error)};
            if (finite_interval(column) && finite_interval(row)) {
                if (column.upper < 0.0f || row.upper < 0.0f || column.lower > config.width - 1
                    || row.lower > result.last_row - 1) return {0, 0, 0, 0};
                result.first_column = int(ceilf(fmaxf(column.lower, 0.0f)));
                result.last_column = int(floorf(fminf(column.upper, float(config.width - 1)))) + 1;
                result.first_row = int(ceilf(fmaxf(row.lower, 0.0f)));
                result.last_row = int(floorf(fminf(row.upper, float(result.last_row - 1)))) + 1;
                return result;
            }
        }
        // Extreme coordinates, coefficient overflow or a denominator interval
        // crossing zero keep the original double enclosure below.
    }
    Support support{widen(fast_support.lower), widen(fast_support.upper)};
    for (int axis = 0; axis < 3; ++axis) {
        set_component(support.lower, axis,
                      __dadd_rd(component(support.lower, axis), -culling.point_error[axis]));
        set_component(support.upper, axis,
                      __dadd_ru(component(support.upper, axis), culling.point_error[axis]));
    }
    const Interval denominator = enclosed_box_dot(culling.denominator, support, culling.origin);
    Interval column = enclosed_quotient(enclosed_box_dot(culling.column, support, culling.origin), denominator);
    Interval row = enclosed_quotient(enclosed_box_dot(culling.row, support, culling.origin), denominator);
    if (!finite_interval(column) || !finite_interval(row)) return result;
    column.lower = __dadd_rd(column.lower, -culling.column_error);
    column.upper = __dadd_ru(column.upper, culling.column_error);
    row.lower = __dadd_rd(row.lower, -culling.row_error);
    row.upper = __dadd_ru(row.upper, culling.row_error);
    if (column.upper < 0.0 || row.upper < 0.0 || column.lower > config.width - 1
        || row.lower > result.last_row - 1) return {0, 0, 0, 0};
    result.first_column = int(ceil(fmax(column.lower, 0.0)));
    result.last_column = int(floor(fmin(column.upper, double(config.width - 1)))) + 1;
    result.first_row = int(ceil(fmax(row.lower, 0.0)));
    result.last_row = int(floor(fmin(row.upper, double(result.last_row - 1)))) + 1;
    return result;
}

__device__ int2 sample_bounds_double(const PreparedRay& ray, Support support, int samples) {
    if (!(ray.span > 0.0f)) return make_int2(0, 0);
    if (!(ray.step > 0.0f)) return make_int2(0, samples);
    Interval interval{-position_error, 1.0 + position_error};
    for (int axis = 0; axis < 3; ++axis) {
        const double origin = component(ray.ray.origin, axis);
        const double direction = component(ray.ray.direction, axis);
        const double first = __dadd_rd(component(support.lower, axis), -component(ray.point_error, axis));
        const double last = __dadd_ru(component(support.upper, axis), component(ray.point_error, axis));
        if (direction == 0.0) {
            if (origin < first || origin > last) return make_int2(0, 0);
        } else {
            const double a = __dadd_rd(first, -origin), b = __dadd_ru(last, -origin);
            const double low = __ddiv_rd(direction > 0.0 ? a : b, direction);
            const double high = __ddiv_ru(direction > 0.0 ? b : a, direction);
            if (!isfinite(low) || !isfinite(high)) return make_int2(0, samples);
            interval.lower = fmax(interval.lower, low);
            interval.upper = fmin(interval.upper, high);
        }
    }
    if (interval.lower > interval.upper) return make_int2(0, 0);
    // Cover float(index), index+0.5 and the sample-position FMA separately
    // from the spatial FMA envelope above, including counts larger than 2^24.
    const double low = __dadd_rd(__ddiv_rd(__dadd_rd(__dadd_rd(interval.lower,
        -position_error), -double(ray.lower)), double(ray.step)), -0.5);
    const double high = __dadd_ru(__ddiv_ru(__dadd_ru(__dadd_ru(interval.upper,
        position_error), -double(ray.lower)), double(ray.step)), -0.5);
    if (!isfinite(low) || !isfinite(high)) return make_int2(0, samples);
    if (high < 0.0 || low > samples - 1) return make_int2(0, 0);
    return make_int2(int(ceil(fmax(low, 0.0))), int(floor(fmin(high, double(samples - 1)))) + 1);
}

__device__ int2 sample_bounds(const PreparedRay& ray, FloatSupport fast_support,
                              Config config) {
    const int samples = config.samples;
    if (!(ray.span > 0.0f)) return make_int2(0, 0);
    if (!(ray.step > 0.0f)) return make_int2(0, samples);
    const Support support{widen(fast_support.lower), widen(fast_support.upper)};
    if (!ray.fast_bounds) return sample_bounds_double(ray, support, samples);
    FloatInterval interval{-float(position_error), 1.0f + float(position_error)};
    #pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        const float origin = component(ray.ray.origin, axis);
        const float direction = component(ray.ray.direction, axis);
        const float first = __fsub_rd(component(fast_support.lower, axis), component(ray.point_error, axis));
        const float last = __fadd_ru(component(fast_support.upper, axis), component(ray.point_error, axis));
        if (!isfinite(first) || !isfinite(last)) return sample_bounds_double(ray, support, samples);
        if (direction == 0.0f) {
            if (origin < first || origin > last) return make_int2(0, 0);
        } else {
            const FloatInterval relative{__fsub_rd(first, origin), __fsub_ru(last, origin)};
            if (!finite_interval(relative)) return sample_bounds_double(ray, support, samples);
            const FloatInterval times = enclosed_product(relative, ray.inverse_direction[axis]);
            if (!finite_interval(times)) return sample_bounds_double(ray, support, samples);
            interval.lower = fmaxf(interval.lower, times.lower);
            interval.upper = fminf(interval.upper, times.upper);
        }
    }
    if (interval.lower > interval.upper) return make_int2(0, 0);
    const FloatInterval relative{
        __fsub_rd(__fsub_rd(interval.lower, float(position_error)), ray.lower),
        __fsub_ru(__fadd_ru(interval.upper, float(position_error)), ray.lower)};
    if (!finite_interval(relative)) return sample_bounds_double(ray, support, samples);
    const FloatInterval indices = enclosed_product(relative, ray.inverse_step);
    const float low = __fsub_rd(indices.lower, 0.5f), high = __fsub_ru(indices.upper, 0.5f);
    if (!isfinite(low) || !isfinite(high)) return sample_bounds_double(ray, support, samples);
    if (high < 0.0f || low > samples - 1) return make_int2(0, 0);
    return make_int2(int(ceilf(fmaxf(low, 0.0f))), int(floorf(fminf(high, float(samples - 1)))) + 1);
}

__device__ float flush_subnormal(float value) {
    return fabsf(value) < FLT_MIN ? copysignf(0.0f, value) : value;
}

template <bool DoubleGradient>
__device__ void add_contribution(float factor, float weight, double& wide, float& narrow,
                                 int* status) {
    float increment = __fmul_rn(factor, weight);
    if (!isfinite(increment)) atomicOr(status, 2);
    if constexpr (DoubleGradient) {
        wide += double(increment);
        if (!isfinite(wide)) atomicOr(status, 2);
    } else {
        // CUDA FP32 global atomics flush both inputs and results. Do not let
        // many individually subnormal terms reappear as a normal local sum.
        increment = flush_subnormal(increment);
        narrow = flush_subnormal(__fadd_rn(flush_subnormal(narrow), increment));
        if (!isfinite(narrow)) atomicOr(status, 2);
    }
}

template <bool DoubleGradient>
__global__ void gather_voxels(Config config, const PreparedRay* rays, const Culling* culling,
                              void* output, int accumulate, int* status) {
    const unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = config.nx * config.ny * config.nz;
    if (thread >= unsigned(count)) return;
    const int voxel = int(thread);
    const int3 index = make_int3(voxel % config.nx, (voxel / config.nx) % config.ny,
                                 voxel / (config.nx * config.ny));
    const FloatSupport fast_support = voxel_support_float(index, config);
    const Bounds bounds = pixel_bounds(fast_support, config, *culling);
    const bool all_rays = config.ray_bundle || culling->fallback;
    double wide = DoubleGradient && accumulate ? static_cast<double*>(output)[voxel] : 0.0;
    float narrow = !DoubleGradient && accumulate ? static_cast<float*>(output)[voxel] : 0.0f;
    for (int row = bounds.first_row; row < bounds.last_row; ++row) {
        for (int column = bounds.first_column; column < bounds.last_column; ++column) {
            const int pixel = all_rays ? column : row * config.width + column;
            const PreparedRay ray = rays[pixel];
            if (!ray.ray.valid || ray.factor == 0.0f) continue;
            const int2 nodes = sample_bounds(ray, fast_support, config);
            for (int sample = nodes.x; sample < nodes.y; ++sample) {
                const float position = fmaf(__fadd_rn(float(sample), 0.5f), ray.step, ray.lower);
                const float3 point = sample_point(ray.ray, position);
                if (!inside_support(point, config)) continue;
                const Cell32 cell = cell_for_point(point, config);
                if (config.nx > 1 && config.ny > 1 && config.nz > 1) {
                    const int x = index.x - cell.lower.x;
                    const int y = index.y - cell.lower.y;
                    const int z = index.z - cell.lower.z;
                    if (unsigned(x) <= 1 && unsigned(y) <= 1 && unsigned(z) <= 1) {
                        const int corner = x | (y << 1) | (z << 2);
                        add_contribution<DoubleGradient>(ray.factor, corner_weight(cell, corner),
                                                          wide, narrow, status);
                    }
                } else {
                    for (int corner = 0; corner < 8; ++corner) {
                        if (corner_index(cell, config, corner) == voxel) {
                            add_contribution<DoubleGradient>(ray.factor, corner_weight(cell, corner),
                                                              wide, narrow, status);
                        }
                    }
                }
            }
        }
    }
    if constexpr (DoubleGradient) {
        static_cast<double*>(output)[voxel] = wide;
        if (!isfinite(wide)) atomicOr(status, 2);
    } else {
        static_cast<float*>(output)[voxel] = narrow;
        if (!isfinite(narrow)) atomicOr(status, 2);
    }
}

template <bool DoubleGradient>
__device__ void shared_increment(float increment, void* gradient, int index,
                                  int* status, bool fast, unsigned int* small_values) {
    if (!isfinite(increment)) atomicOr(status, 2);
    if constexpr (DoubleGradient) {
        const double previous = atomicAdd(static_cast<double*>(gradient) + index, double(increment));
        if (!isfinite(__dadd_rn(previous, double(increment)))) atomicOr(status, 2);
    } else {
        // Shared FP32 atomicAdd retains subnormals, unlike the existing global
        // atomics. CAS applies the global input/result flush at each update,
        // including cancellation that would otherwise leave a subnormal sum.
        increment = flush_subnormal(increment);
        if (fast) {
            if (increment != 0.0f && fabsf(increment) < 0x1p-103f) atomicExch(small_values, 1u);
            atomicAdd(static_cast<float*>(gradient) + index, increment);
            return;
        }
        auto* address = reinterpret_cast<unsigned int*>(static_cast<float*>(gradient) + index);
        unsigned int previous = atomicCAS(address, 0u, 0u);
        for (;;) {
            const float sum = flush_subnormal(__fadd_rn(flush_subnormal(__uint_as_float(previous)), increment));
            const unsigned int observed = atomicCAS(address, previous, __float_as_uint(sum));
            if (observed == previous) {
                if (!isfinite(sum)) atomicOr(status, 2);
                break;
            }
            previous = observed;
        }
    }
}

template <int Side>
__device__ void flush_tile_cell(int3 lower, int3 first, const float* sums, void* storage,
                                int* status, bool fast, unsigned int* small_values) {
    const int x = lower.x - first.x, y = lower.y - first.y, z = lower.z - first.z;
    #pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        const int a = x + (corner & 1), b = y + ((corner >> 1) & 1), c = z + ((corner >> 2) & 1);
        if (unsigned(a) < Side && unsigned(b) < Side && unsigned(c) < Side) {
            shared_increment<false>(sums[corner], storage, (c * Side + b) * Side + a,
                                     status, fast, small_values);
        }
    }
}

template <bool DoubleGradient, int Side = 8>
__global__ void gather_tiles(Config config, const PreparedRay* rays, const Culling* culling,
                             void* output, int accumulate, int* status) {
    constexpr int side = Side, tile_voxels = side * side * side;
    const int tiles_x = (config.nx + side - 1) / side;
    const int tiles_y = (config.ny + side - 1) / side;
    const int3 first = make_int3((blockIdx.x % tiles_x) * side,
        ((blockIdx.x / tiles_x) % tiles_y) * side, (blockIdx.x / (tiles_x * tiles_y)) * side);
    __shared__ double storage[DoubleGradient ? tile_voxels : tile_voxels / 2];
    __shared__ Bounds bounds;
    __shared__ FloatSupport support;
    __shared__ unsigned int small_values;
    if (threadIdx.x == 0) {
        small_values = 0;
        const int3 last = make_int3(min(first.x + side - 1, config.nx - 1),
            min(first.y + side - 1, config.ny - 1), min(first.z + side - 1, config.nz - 1));
        support = {voxel_support_float(first, config).lower, voxel_support_float(last, config).upper};
        bounds = pixel_bounds(support, config, *culling);
    }
    // A nonzero FP32 value of magnitude >= 2^-103 is an integer multiple of
    // 2^-126. Sums of such values, rounded after every addition, stay on that
    // lattice and therefore cannot be subnormal. If every initial value and
    // flushed increment meets this condition, shared atomicAdd has the same
    // input/result FTZ semantics as global atomicAdd, including cancellation.
    // Otherwise discard the provisional tile and recompute with exact CAS FTZ.
    for (int pass = 0; pass < (DoubleGradient ? 1 : 2); ++pass) {
        if (pass == 1 && !small_values) break;
        bool initial_small = false;
        for (int local = threadIdx.x; local < tile_voxels; local += blockDim.x) {
            const int x = first.x + local % side;
            const int y = first.y + (local / side) % side;
            const int z = first.z + local / (side * side);
            const bool valid = x < config.nx && y < config.ny && z < config.nz;
            const int voxel = valid ? (z * config.ny + y) * config.nx + x : 0;
            if constexpr (DoubleGradient) {
                storage[local] = valid && accumulate ? static_cast<double*>(output)[voxel] : 0.0;
            } else {
                const float value = valid && accumulate ? static_cast<float*>(output)[voxel] : 0.0f;
                reinterpret_cast<float*>(storage)[local] = value;
                initial_small = initial_small || (value != 0.0f && fabsf(value) < 0x1p-103f);
            }
        }
        __syncthreads();
        if (initial_small) atomicExch(&small_values, 1u);
        const int columns = bounds.last_column - bounds.first_column;
        const int pixels = columns * (bounds.last_row - bounds.first_row);
        const bool all_rays = config.ray_bundle || culling->fallback;
        for (unsigned int candidate = threadIdx.x; candidate < unsigned(pixels); candidate += blockDim.x) {
            const int column = bounds.first_column + int(candidate % unsigned(columns));
            const int row = bounds.first_row + int(candidate / unsigned(columns));
            const int pixel = all_rays ? column : row * config.width + column;
            const PreparedRay ray = rays[pixel];
            if (!ray.ray.valid || ray.factor == 0.0f) continue;
            const int2 nodes = sample_bounds(ray, support, config);
            const bool aggregate = !DoubleGradient && fabsf(ray.factor) <= 1.0f;
            int3 pending = make_int3(-1, -1, -1);
            float sums[8] = {};
            for (int sample = nodes.x; sample < nodes.y; ++sample) {
                const float position = fmaf(__fadd_rn(float(sample), 0.5f), ray.step, ray.lower);
                const float3 point = sample_point(ray.ray, position);
                if (!inside_support(point, config)) continue;
                const Cell32 cell = cell_for_point(point, config);
                if (aggregate) {
                    if (cell.lower.x != pending.x || cell.lower.y != pending.y || cell.lower.z != pending.z) {
                        if (pending.x >= 0) {
                            flush_tile_cell<Side>(pending, first, sums, storage, status, pass == 0, &small_values);
                        }
                        #pragma unroll
                        for (int corner = 0; corner < 8; ++corner) sums[corner] = 0.0f;
                        pending = cell.lower;
                    }
                    #pragma unroll
                    for (int corner = 0; corner < 8; ++corner) {
                        const float increment = flush_subnormal(__fmul_rn(ray.factor, corner_weight(cell, corner)));
                        // All terms for one ray have one sign; unit-bounded terms
                        // cannot overflow a sum over at most INT_MAX samples.
                        sums[corner] = __fadd_rn(sums[corner], increment);
                    }
                    continue;
                }
                const int x = cell.lower.x - first.x, y = cell.lower.y - first.y, z = cell.lower.z - first.z;
                #pragma unroll
                for (int corner = 0; corner < 8; ++corner) {
                    const int a = x + (corner & 1), b = y + ((corner >> 1) & 1), c = z + ((corner >> 2) & 1);
                    if (unsigned(a) < side && unsigned(b) < side && unsigned(c) < side) {
                        shared_increment<DoubleGradient>(__fmul_rn(ray.factor, corner_weight(cell, corner)),
                            storage, (c * side + b) * side + a, status, pass == 0, &small_values);
                    }
                }
            }
            if (aggregate && pending.x >= 0) {
                flush_tile_cell<Side>(pending, first, sums, storage, status, pass == 0, &small_values);
            }
        }
        __syncthreads();
    }
    for (int local = threadIdx.x; local < tile_voxels; local += blockDim.x) {
        const int x = first.x + local % side;
        const int y = first.y + (local / side) % side;
        const int z = first.z + local / (side * side);
        if (x < config.nx && y < config.ny && z < config.nz) {
            const int voxel = (z * config.ny + y) * config.nx + x;
            if constexpr (DoubleGradient) {
                static_cast<double*>(output)[voxel] = storage[local];
                if (!isfinite(storage[local])) atomicOr(status, 2);
            } else {
                const float value = reinterpret_cast<float*>(storage)[local];
                static_cast<float*>(output)[voxel] = value;
                if (!isfinite(value)) atomicOr(status, 2);
            }
        }
    }
}

}  // namespace

extern "C" size_t dpt_adjoint_scratch_bytes(Config config) {
    return ray_offset + size_t(config.pixels) * sizeof(PreparedRay);
}

extern "C" cudaError_t dpt_launch_adjoint(Config config, const double* pose,
                                          const double* sources, const double* endpoints,
                                          const void* seeds, int seed_double, void* gradient,
                                          int gradient_double, int accumulate, int* status,
                                          void* scratch, cudaStream_t stream) {
    auto* culling = static_cast<Culling*>(scratch);
    auto* rays = reinterpret_cast<PreparedRay*>(static_cast<char*>(scratch) + ray_offset);
    initialise_culling<<<1, 1, 0, stream>>>(config, pose, sources, endpoints, culling, status);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    if (seed_double) {
        prepare_rays<true><<<(unsigned(config.pixels) + 255u) / 256u, 256, 0, stream>>>(
            config, pose, sources, endpoints, seeds, rays, culling, status);
    } else {
        prepare_rays<false><<<(unsigned(config.pixels) + 255u) / 256u, 256, 0, stream>>>(
            config, pose, sources, endpoints, seeds, rays, culling, status);
    }
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    const int voxels = config.nx * config.ny * config.nz;
    if (!config.ray_bundle && config.nx >= 8 && config.ny >= 8 && config.nz >= 8) {
        const int tiles = ((config.nx + 7) / 8) * ((config.ny + 7) / 8) * ((config.nz + 7) / 8);
        if (gradient_double) {
            gather_tiles<true><<<tiles, 128, 0, stream>>>(config, rays, culling, gradient, accumulate, status);
        } else {
            gather_tiles<false><<<tiles, 128, 0, stream>>>(config, rays, culling, gradient, accumulate, status);
        }
    } else if (gradient_double) {
        gather_voxels<true><<<(unsigned(voxels) + 127u) / 128u, 128, 0, stream>>>(
            config, rays, culling, gradient, accumulate, status);
    } else {
        gather_voxels<false><<<(unsigned(voxels) + 127u) / 128u, 128, 0, stream>>>(
            config, rays, culling, gradient, accumulate, status);
    }
    return cudaGetLastError();
}
