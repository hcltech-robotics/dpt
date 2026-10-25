#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

namespace dpt_native {

struct Config {
    int nx, ny, nz, width, pixels, samples;
    double object_to_grid[9], grid_origin[3], source[3], detector_origin[3];
    double column_step[3], row_step[3];
    int ray_bundle;
};

struct Ray32 {
    float3 origin, direction;
    float length;
    bool valid;
};

struct Interval32 { float lower, upper; };
struct Cell32 { int3 lower; float3 fraction; };

__device__ __forceinline__ bool finite_point(float3 p) {
    return isfinite(p.x) && isfinite(p.y) && isfinite(p.z);
}

__device__ __forceinline__ Ray32 make_ray(
    Config c, const double* pose, int pixel, const double* sources,
    const double* endpoints, int* status
) {
    double s[3], d[3], relative[3], os[3], od[3], origin[3], direction[3];
    bool valid = true;
    const int row = pixel / c.width, column = pixel % c.width;
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        const size_t component = size_t(pixel) * 3 + size_t(a);
        s[a] = c.ray_bundle ? sources[component] : c.source[a];
        const double end = c.ray_bundle ? endpoints[component] :
            c.detector_origin[a] + double(column) * c.column_step[a] +
            double(row) * c.row_step[a];
        d[a] = end - s[a];
        relative[a] = s[a] - pose[9 + a];
        valid = valid && isfinite(d[a]) && isfinite(relative[a]);
    }
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        os[a] = pose[a] * relative[0] + pose[3+a] * relative[1] + pose[6+a] * relative[2];
        od[a] = pose[a] * d[0] + pose[3+a] * d[1] + pose[6+a] * d[2];
        valid = valid && isfinite(os[a]) && isfinite(od[a]);
        os[a] -= c.grid_origin[a];
    }
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        origin[a] = c.object_to_grid[3*a] * os[0] +
            c.object_to_grid[3*a+1] * os[1] + c.object_to_grid[3*a+2] * os[2];
        direction[a] = c.object_to_grid[3*a] * od[0] +
            c.object_to_grid[3*a+1] * od[1] + c.object_to_grid[3*a+2] * od[2];
        valid = valid && isfinite(origin[a]) && isfinite(direction[a]);
    }
    const double length = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    Ray32 ray{make_float3(float(origin[0]), float(origin[1]), float(origin[2])),
              make_float3(float(direction[0]), float(direction[1]), float(direction[2])),
              float(length), false};
    ray.valid = valid && isfinite(length) && length > 0.0 &&
        finite_point(ray.origin) && finite_point(ray.direction) &&
        isfinite(ray.length) && ray.length > 0.0f;
    if (!ray.valid) atomicOr(status, 2);
    return ray;
}

__device__ __forceinline__ Interval32 clip_ray(Ray32 ray, Config c) {
    Interval32 interval{0.0f, 1.0f};
    const float origin[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
    const float direction[3] = {ray.direction.x, ray.direction.y, ray.direction.z};
    const int shape[3] = {c.nx, c.ny, c.nz};
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        const float high = float(shape[a]) - 0.5f;
        if (direction[a] == 0.0f) {
            if (origin[a] < -0.5f || origin[a] > high) interval.upper = -1.0f;
        } else {
            float first = (-0.5f - origin[a]) / direction[a];
            float last = (high - origin[a]) / direction[a];
            if (first > last) { const float swap = first; first = last; last = swap; }
            if (first > interval.lower) interval.lower = first;
            if (last < interval.upper) interval.upper = last;
        }
    }
    return interval;
}

__device__ __forceinline__ float3 sample_point(Ray32 ray, float position) {
    return make_float3(fmaf(position, ray.direction.x, ray.origin.x),
                       fmaf(position, ray.direction.y, ray.origin.y),
                       fmaf(position, ray.direction.z, ray.origin.z));
}

__device__ __forceinline__ bool inside_support(float3 p, Config c) {
    return p.x >= -0.5f && p.y >= -0.5f && p.z >= -0.5f &&
        p.x <= float(c.nx)-0.5f && p.y <= float(c.ny)-0.5f && p.z <= float(c.nz)-0.5f;
}

__device__ __forceinline__ Cell32 cell_for_point(float3 p, Config c) {
    p.x = fminf(fmaxf(p.x, 0.0f), float(c.nx-1));
    p.y = fminf(fmaxf(p.y, 0.0f), float(c.ny-1));
    p.z = fminf(fmaxf(p.z, 0.0f), float(c.nz-1));
    const int3 lower = make_int3(min(int(floorf(p.x)), max(c.nx-2, 0)),
                                min(int(floorf(p.y)), max(c.ny-2, 0)),
                                min(int(floorf(p.z)), max(c.nz-2, 0)));
    return {lower, make_float3(p.x-float(lower.x), p.y-float(lower.y), p.z-float(lower.z))};
}

__device__ __forceinline__ int corner_index(Cell32 cell, Config c, int corner) {
    const int x = min(cell.lower.x + (corner & 1), c.nx-1);
    const int y = min(cell.lower.y + ((corner >> 1) & 1), c.ny-1);
    const int z = min(cell.lower.z + ((corner >> 2) & 1), c.nz-1);
    return (z*c.ny + y)*c.nx + x;
}

__device__ __forceinline__ float corner_weight(Cell32 cell, int corner) {
    const float x = (corner & 1) ? cell.fraction.x : 1.0f-cell.fraction.x;
    const float y = (corner & 2) ? cell.fraction.y : 1.0f-cell.fraction.y;
    const float z = (corner & 4) ? cell.fraction.z : 1.0f-cell.fraction.z;
    return __fmul_rn(__fmul_rn(x, y), z);
}

}  // namespace dpt_native

extern "C" size_t dpt_adjoint_scratch_bytes(dpt_native::Config config);
extern "C" cudaError_t dpt_launch_adjoint(
    dpt_native::Config config, const double* pose, const double* sources,
    const double* endpoints, const void* seeds, int seed_double, void* gradient,
    int gradient_double, int accumulate, int* status, void* scratch, cudaStream_t stream
);
