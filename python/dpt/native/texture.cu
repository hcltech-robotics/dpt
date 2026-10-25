#include "common.cuh"
#include <cstdint>
#include <new>

using namespace dpt_native;

struct Atlas {
    cudaArray_t array{};
    cudaTextureObject_t texture{};
    cudaSurfaceObject_t surface{};
    int tiles;
    size_t bytes;
};

namespace {

class DeviceScope {
    int previous = -1;
public:
    cudaError_t error;
    explicit DeviceScope(int device) : error(cudaGetDevice(&previous)) {
        if (error == cudaSuccess && previous != device) error = cudaSetDevice(device);
    }
    ~DeviceScope() { if (previous >= 0) cudaSetDevice(previous); }
};

__global__ void pack_atlas(const float* field, cudaSurfaceObject_t surface,
                           int nx, int ny, int nz, int tiles) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= unsigned(nx * ny * nz)) return;
    const int x = i % nx, y = (i / nx) % ny, z = i / (nx * ny);
    surf2Dwrite(__float_as_uint(field[i]), surface, (x + (z & (tiles-1)) * nx) * sizeof(float),
                y + (z >> (__ffs(tiles)-1)) * ny);
}

__device__ __forceinline__ float4 gather_bits(cudaTextureObject_t texture, float x, float y) {
    // Integer texels preserve subnormals that float gather would flush to zero.
    const uint4 bits = tex2Dgather<uint4>(texture, x, y, 0);
    return make_float4(__uint_as_float(bits.x), __uint_as_float(bits.y),
                       __uint_as_float(bits.z), __uint_as_float(bits.w));
}

template <bool Gather>
__device__ __forceinline__ float sample_value(Config config, Cell32 cell,
    const float* field, cudaTextureObject_t texture, cudaTextureObject_t point_texture,
    int tiles) {
    float value = 0.0f;
    if (Gather && config.nx > 1 && config.ny > 1 && config.nz > 1) {
        const int tile_shift = __ffs(tiles)-1;
        const int z0 = cell.lower.z, z1 = cell.lower.z + 1;
        const float4 a = gather_bits(texture,
            float(cell.lower.x + (z0 & (tiles-1)) * config.nx) + 1.0f,
            float(cell.lower.y + (z0 >> tile_shift) * config.ny) + 1.0f);
        const float4 b = gather_bits(texture,
            float(cell.lower.x + (z1 & (tiles-1)) * config.nx) + 1.0f,
            float(cell.lower.y + (z1 >> tile_shift) * config.ny) + 1.0f);
        value = fmaf(corner_weight(cell, 0), a.w, value);
        value = fmaf(corner_weight(cell, 1), a.z, value);
        value = fmaf(corner_weight(cell, 2), a.x, value);
        value = fmaf(corner_weight(cell, 3), a.y, value);
        value = fmaf(corner_weight(cell, 4), b.w, value);
        value = fmaf(corner_weight(cell, 5), b.z, value);
        value = fmaf(corner_weight(cell, 6), b.x, value);
        value = fmaf(corner_weight(cell, 7), b.y, value);
    } else {
        #pragma unroll
        for (int corner = 0; corner < 8; ++corner) {
            const int index = corner_index(cell, config, corner);
            const float mu = point_texture ? tex1Dfetch<float>(point_texture, index) : field[index];
            value = fmaf(corner_weight(cell, corner), mu, value);
        }
    }
    return value;
}

template <bool Guaranteed, bool Gather>
__device__ __forceinline__ float canonical_value(unsigned i, float step, float lower,
    Ray32 ray, Config config, const float* field, cudaTextureObject_t texture,
    cudaTextureObject_t point_texture, int tiles, int* status) {
    const float position = fmaf(float(i) + 0.5f, step, lower);
    const float3 point = sample_point(ray, position);
    if constexpr (!Guaranteed) {
        if (!finite_point(point)) atomicOr(status, 2);
        if (!inside_support(point, config)) return 0.0f;
    }
    return sample_value<Gather>(config, cell_for_point(point, config), field,
                                texture, point_texture, tiles);
}

template <int Lanes, bool Gather, bool Guaranteed>
__device__ __forceinline__ float serial_chunks(unsigned count, int lane, unsigned mask,
    float step, float lower, Ray32 ray, Config config, const float* field,
    cudaTextureObject_t texture, cudaTextureObject_t point_texture, int tiles, int* status) {
    float total = 0.0f;
    const unsigned full = (count / Lanes) * Lanes;
    for (unsigned base = 0; base < full; base += Lanes) {
        const float value = canonical_value<Guaranteed, Gather>(base + unsigned(lane), step,
            lower, ray, config, field, texture, point_texture, tiles, status);
        #pragma unroll
        for (int j = 0; j < Lanes; ++j)
            total = __fadd_rn(total, __shfl_sync(mask, value, j, Lanes));
    }
    if (full < count) {
        float value = 0.0f;
        if (full + unsigned(lane) < count)
            value = canonical_value<Guaranteed, Gather>(full + unsigned(lane), step,
                lower, ray, config, field, texture, point_texture, tiles, status);
        #pragma unroll
        for (int j = 0; j < Lanes; ++j) {
            const float serial_value = __shfl_sync(mask, value, j, Lanes);
            if (full + unsigned(j) < count) total = __fadd_rn(total, serial_value);
        }
    }
    return total;
}

template <int Lanes, bool Gather>
__device__ __forceinline__ float integrate_ray(Config config, unsigned pixel,
    const float* field, cudaTextureObject_t texture, cudaTextureObject_t point_texture,
    int tiles, const double* pose, const double* sources, const double* endpoints, int* status) {
    const int lane = threadIdx.x % Lanes;
    const Ray32 ray = make_ray(config, pose, pixel, sources, endpoints, status);
    const Interval32 interval = clip_ray(ray, config);
    const bool valid = ray.valid && interval.upper > interval.lower;
    const float step = (interval.upper - interval.lower) / float(config.samples);
    bool guaranteed = false;
    if (valid) {
        // Each FMA coordinate is monotone in the sample index. Finite endpoints
        // inside the box certify all intervening nodes without changing them.
        const float3 first = sample_point(ray, fmaf(0.5f, step, interval.lower));
        const float3 last = sample_point(ray,
            fmaf(float(unsigned(config.samples)-1u) + 0.5f, step, interval.lower));
        guaranteed = finite_point(first) && finite_point(last)
            && inside_support(first, config) && inside_support(last, config);
    }
    const unsigned active = __activemask();
    const unsigned fast_mask = __ballot_sync(active, valid && guaranteed);
    const unsigned slow_mask = __ballot_sync(active, valid && !guaranteed);
    float total = 0.0f;
    if (valid) {
        if (guaranteed)
            total = serial_chunks<Lanes,Gather,true>(unsigned(config.samples),lane,
                fast_mask,step,interval.lower,ray,config,field,texture,point_texture,tiles,status);
        else
            total = serial_chunks<Lanes,Gather,false>(unsigned(config.samples),lane,
                slow_mask,step,interval.lower,ray,config,field,texture,point_texture,tiles,status);
        total = __fmul_rn(total, __fmul_rn(ray.length, step));
    }
    return total;
}

template <int Lanes, bool Gather, typename Output>
__global__ void forward_kernel(Config config, const float* field, cudaTextureObject_t texture,
    cudaTextureObject_t point_texture, int tiles, const double* pose,
    const double* sources, const double* endpoints, Output* output, int* status) {
    const unsigned pixel = blockIdx.x * (blockDim.x / Lanes) + threadIdx.x / Lanes;
    if (pixel >= unsigned(config.pixels)) return;
    const float total = integrate_ray<Lanes, Gather>(config, pixel, field, texture,
        point_texture, tiles, pose, sources, endpoints, status);
    if (threadIdx.x % Lanes == 0) {
        output[pixel] = Output(total);
        if (!isfinite(total)) atomicOr(status, 2);
    }
}

template <int Lanes, bool Gather>
__global__ void forward_many_kernel(const Config* configurations, int stride,
    const float* field, cudaTextureObject_t texture, cudaTextureObject_t point_texture,
    int tiles, const double* poses, float* output, int* statuses) {
    const Config config = configurations[blockIdx.y];
    const unsigned pixel = blockIdx.x * (blockDim.x / Lanes) + threadIdx.x / Lanes;
    if (pixel >= unsigned(config.pixels)) return;
    int* status = statuses + blockIdx.y;
    const float total = integrate_ray<Lanes, Gather>(config, pixel, field, texture,
        point_texture, tiles, poses + size_t(blockIdx.y) * 12, nullptr, nullptr, status);
    if (threadIdx.x % Lanes == 0) {
        output[size_t(blockIdx.y) * stride + pixel] = total;
        if (!isfinite(total)) atomicOr(status, 2);
    }
}

template <int Lanes, bool Gather>
void launch_forward(Atlas* atlas, const Config* config, const float* field,
    uint64_t point_texture, const double* pose, const double* sources,
    const double* endpoints, void* output, int output_double, int* status,
    cudaStream_t stream) {
    constexpr unsigned rays_per_block = 256 / Lanes;
    const unsigned blocks = (unsigned(config->pixels) + rays_per_block-1) / rays_per_block;
    if (output_double)
        forward_kernel<Lanes,Gather><<<blocks,256,0,stream>>>(*config,field,
            atlas ? atlas->texture : 0,point_texture,atlas ? atlas->tiles : 1,pose,sources,endpoints,
            static_cast<double*>(output),status);
    else
        forward_kernel<Lanes,Gather><<<blocks,256,0,stream>>>(*config,field,
            atlas ? atlas->texture : 0,point_texture,atlas ? atlas->tiles : 1,pose,sources,endpoints,
            static_cast<float*>(output),status);
}

}  // namespace

extern "C" const char* dpt_cuda_error(int code) {
    return cudaGetErrorString(cudaError_t(code));
}

extern "C" int dpt_atlas_create(const Config* config, int device, Atlas** result) {
    DeviceScope scope(device);
    if (scope.error != cudaSuccess) return scope.error;
    *result = nullptr;
    if (config->nx == 1 || config->ny == 1 || config->nz == 1) return cudaSuccess;
    cudaDeviceProp properties;
    cudaError_t error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) return error;
    int tiles = 0;
    size_t side = SIZE_MAX;
    for (size_t candidate = 1; candidate <= size_t(config->nz) * 2; candidate *= 2) {
        const size_t width = size_t(config->nx) * candidate;
        const size_t height = size_t(config->ny) *
            ((size_t(config->nz) + candidate - 1) / candidate);
        // Gather coordinates must retain every integer centre exactly in FP32.
        if (width > size_t(properties.maxTexture2DGather[0]) ||
            height > size_t(properties.maxTexture2DGather[1]) ||
            width >= (1u << 24) || height >= (1u << 24)) continue;
        const size_t maximum = width > height ? width : height;
        if (maximum < side) { side = maximum; tiles = int(candidate); }
    }
    if (!tiles) return cudaSuccess;  // Unsupported atlas shapes use point fetches.
    Atlas* atlas = new (std::nothrow) Atlas{};
    if (!atlas) return cudaErrorMemoryAllocation;
    atlas->tiles = tiles;
    const size_t width = size_t(config->nx) * tiles;
    const size_t height = size_t(config->ny) * ((config->nz + size_t(tiles) - 1) / tiles);
    atlas->bytes = width * height * sizeof(float);
    const cudaChannelFormatDesc channel = cudaCreateChannelDesc<unsigned int>();
    error = cudaMallocArray(&atlas->array, &channel, width, height,
        cudaArrayTextureGather | cudaArraySurfaceLoadStore);
    if (error == cudaSuccess) {
        cudaResourceDesc resource{};
        resource.resType = cudaResourceTypeArray;
        resource.res.array.array = atlas->array;
        error = cudaCreateSurfaceObject(&atlas->surface, &resource);
        if (error == cudaSuccess) {
            cudaTextureDesc texture{};
            texture.readMode = cudaReadModeElementType;
            texture.filterMode = cudaFilterModePoint;
            texture.addressMode[0] = cudaAddressModeClamp;
            texture.addressMode[1] = cudaAddressModeClamp;
            error = cudaCreateTextureObject(&atlas->texture, &resource, &texture, nullptr);
        }
    }
    if (error != cudaSuccess) {
        if (atlas->texture) cudaDestroyTextureObject(atlas->texture);
        if (atlas->surface) cudaDestroySurfaceObject(atlas->surface);
        if (atlas->array) cudaFreeArray(atlas->array);
        delete atlas;
        return error;
    }
    *result = atlas;
    return cudaSuccess;
}

extern "C" size_t dpt_atlas_bytes(const Atlas* atlas) {
    return atlas ? atlas->bytes : 0;
}

extern "C" int dpt_atlas_destroy(Atlas* atlas, int device, cudaStream_t stream) {
    DeviceScope scope(device);
    if (scope.error != cudaSuccess) return scope.error;
    if (!atlas) return cudaSuccess;
    cudaError_t result = cudaStreamSynchronize(stream);
    const cudaError_t texture = cudaDestroyTextureObject(atlas->texture);
    const cudaError_t surface = cudaDestroySurfaceObject(atlas->surface);
    const cudaError_t allocation = cudaFreeArray(atlas->array);
    if (result == cudaSuccess) result = texture;
    if (result == cudaSuccess) result = surface;
    if (result == cudaSuccess) result = allocation;
    delete atlas;
    return result;
}

extern "C" int dpt_texture_create(const float* field, size_t count, int device, uint64_t* handle) {
    DeviceScope scope(device);
    if (scope.error != cudaSuccess) return scope.error;
    *handle = 0;
    cudaDeviceProp properties;
    cudaError_t error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) return error;
    int maximum_width = 0;
    error = cudaDeviceGetAttribute(&maximum_width, cudaDevAttrMaxTexture1DLinearWidth, device);
    if (error != cudaSuccess) return error;
    // Contiguous subviews need not satisfy the texture resource alignment.
    if (reinterpret_cast<uintptr_t>(field) % properties.textureAlignment ||
        count > size_t(maximum_width)) return cudaSuccess;
    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeLinear;
    resource.res.linear.devPtr = const_cast<float*>(field);
    resource.res.linear.desc = cudaCreateChannelDesc<float>();
    resource.res.linear.sizeInBytes = count * sizeof(float);
    cudaTextureDesc texture{};
    texture.readMode = cudaReadModeElementType;
    texture.filterMode = cudaFilterModePoint;
    return cudaCreateTextureObject(reinterpret_cast<cudaTextureObject_t*>(handle),
                                   &resource, &texture, nullptr);
}

extern "C" int dpt_texture_destroy(uint64_t texture, int device, cudaStream_t stream) {
    DeviceScope scope(device);
    if (scope.error != cudaSuccess) return scope.error;
    const cudaError_t completion = cudaStreamSynchronize(stream);
    const cudaError_t destruction = texture ? cudaDestroyTextureObject(texture) : cudaSuccess;
    return completion == cudaSuccess ? destruction : completion;
}

extern "C" int dpt_forward(
    const Config* config, const float* field, uint64_t texture, Atlas* atlas,
    const double* pose,
    const double* sources, const double* endpoints, void* output, int output_double,
    int* status, int device, cudaStream_t stream
) {
    DeviceScope scope(device);
    if (scope.error != cudaSuccess) return scope.error;
    if (atlas) {
        const unsigned count = unsigned(config->nx * config->ny * config->nz);
        pack_atlas<<<(count + 255u) / 256u, 256, 0, stream>>>(
            field, atlas->surface, config->nx, config->ny, config->nz, atlas->tiles);
        const cudaError_t packed = cudaGetLastError();
        if (packed != cudaSuccess) return packed;
        launch_forward<4, true>(atlas, config, field, texture, pose, sources,
            endpoints, output, output_double, status, stream);
    } else {
        launch_forward<8, false>(nullptr, config, field, texture, pose, sources,
            endpoints, output, output_double, status, stream);
    }
    return cudaGetLastError();
}

extern "C" size_t dpt_scratch_bytes(const Config* config) {
    return dpt_adjoint_scratch_bytes(*config);
}

extern "C" int dpt_forward_many(
    const Config* first, const Config* configurations, int views, const float* field,
    uint64_t point_texture, Atlas* atlas, const double* poses, float* output,
    int* statuses, int device, cudaStream_t stream
) {
    DeviceScope scope(device);
    if (scope.error != cudaSuccess) return scope.error;
    if (views < 1 || views > 65535 || first->ray_bundle) return cudaErrorInvalidValue;
    if (atlas) {
        const unsigned count = unsigned(first->nx * first->ny * first->nz);
        pack_atlas<<<(count + 255u) / 256u, 256, 0, stream>>>(
            field, atlas->surface, first->nx, first->ny, first->nz, atlas->tiles);
        const cudaError_t packed = cudaGetLastError();
        if (packed != cudaSuccess) return packed;
        const dim3 blocks((unsigned(first->pixels) + 63u) / 64u, unsigned(views));
        forward_many_kernel<4, true><<<blocks, 256, 0, stream>>>(
            configurations, first->pixels, field, atlas->texture, point_texture,
            atlas->tiles, poses, output, statuses);
    } else {
        const dim3 blocks((unsigned(first->pixels) + 31u) / 32u, unsigned(views));
        forward_many_kernel<8, false><<<blocks, 256, 0, stream>>>(
            configurations, first->pixels, field, 0, point_texture, 1, poses, output, statuses);
    }
    return cudaGetLastError();
}

extern "C" int dpt_adjoint(
    const Config* config, const double* pose, const double* sources, const double* endpoints,
    const void* seeds, int seed_double, void* gradient, int gradient_double, int accumulate,
    int* status, void* scratch, int device, cudaStream_t stream
) {
    DeviceScope scope(device);
    if (scope.error != cudaSuccess) return scope.error;
    return dpt_launch_adjoint(*config, pose, sources, endpoints, seeds, seed_double,
        gradient, gradient_double, accumulate, status, scratch, stream);
}
