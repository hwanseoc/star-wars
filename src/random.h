#pragma once

#include <random>
#include <curand_kernel.h>
#include <vec.h>

__global__ void initCurandStates(curandState *states, unsigned long seed, int32_t x_dim) {
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t id = y * x_dim + x;
    curand_init(seed, id, 0, &states[id]);
}

__device__ inline float cuda_random_float(curandState *state) {
    return curand_uniform(state);
}

__host__ inline float random_float() {
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    static std::mt19937 generator;
    return distribution(generator);
}

__device__ inline vec3 cuda_random_disk(curandState *state) {
    return vec3(cuda_random_float(state) - 0.5f, cuda_random_float(state) - 0.5f, 0.0f);
}

__host__ inline vec3 random_disk() {
    return vec3(random_float() - 0.5f, random_float() - 0.5f, 0.0f);
}

__device__ inline vec3 cuda_random_sphere(curandState *state) {
    float phi = curand_uniform(state) * 2.0f * M_PI;
    float cosTheta = curand_uniform(state) * 2.0f - 1.0f;
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

    return vec3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
}


__host__ inline vec3 random_sphere() {
    while (true) {
        vec3 p = vec3(random_float()*2.0f - 1.0f, random_float()*2.0f - 1.0f, random_float()*2.0f - 1.0f);
        float lensq = p.length_squared();
        if (1e-6f < lensq/* && lensq <= 1.0f*/) {
            return p / sqrt(lensq);
        }
    }
}

__device__ inline vec3 cuda_random_hemisphere(curandState *state, const vec3 &normal) {
    vec3 ret = cuda_random_sphere(state);
    if (dot(ret, normal) > 0.0f){
        return ret;
    } else {
        return -ret;
    }
}

__host__ inline vec3 random_hemisphere(const vec3& normal) {
    vec3 ret = random_sphere();
    if (dot(ret, normal) > 0.0f){
        return ret;
    } else {
        return -ret;
    }
}
