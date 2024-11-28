#pragma once

#include <algorithm>
#include <memory>

// #include <glm/glm.hpp>
#include <vec.h>

class cuda_Texture {
public:
    __host__ __device__ virtual ~cuda_Texture() = default;

    __device__ virtual vec3 value(float u, float v, const vec3 &p) const = 0;
};



class Texture {
public:
    __host__ __device__ virtual ~Texture() = default;

    __host__ virtual vec3 value(float u, float v, const vec3 &p) const = 0;

    __host__ virtual cuda_Texture *convertToDevice() = 0;
};

class cuda_SolidTexture : public cuda_Texture {
    vec3 albedo;
public:
    __host__ cuda_SolidTexture(const vec3 &albedo) : albedo(albedo) {}

    __device__ vec3 value(float u, float v, const vec3& p) const override {
        return albedo;
    }
};


class SolidTexture : public Texture {
    vec3 albedo;
    cuda_SolidTexture *host_cuda_texture;
public:
    __host__ SolidTexture(const vec3 &albedo) : albedo(albedo) {}
    __host__ ~SolidTexture() {
        delete host_cuda_texture;
    }

    __host__ vec3 value(float u, float v, const vec3& p) const override {
        return albedo;
    }

    __host__ virtual cuda_Texture *convertToDevice() override {
        host_cuda_texture = new cuda_SolidTexture(albedo);

        cuda_SolidTexture *dev_cuda_texture;

        cudaMalloc(&dev_cuda_texture, sizeof(cuda_Texture));
        cudaMemcpy(dev_cuda_texture, host_cuda_texture, sizeof(cuda_Texture), cudaMemcpyHostToDevice);

        return dev_cuda_texture;
    }
};


class cuda_CheckerTexture : public cuda_Texture {
    float inv_scale;
    cuda_Texture *even;
    cuda_Texture *odd;

public:
    cuda_CheckerTexture(float inv_scale, cuda_Texture *even, cuda_Texture *odd) : inv_scale(inv_scale), even(even), odd(odd) {}
    __host__ ~cuda_CheckerTexture(){
        cudaFree(even);
        cudaFree(odd);
    }

    __device__ vec3 value(float u, float v, const vec3 &p) const override {
        int32_t xInt = static_cast<int32_t>(std::floor(inv_scale * p.x));
        int32_t yInt = static_cast<int32_t>(std::floor(inv_scale * p.y));
        int32_t zInt = static_cast<int32_t>(std::floor(inv_scale * p.z));

        bool isEven = (xInt + yInt + zInt) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }
};



class CheckerTexture : public Texture {
    float inv_scale;
    Texture *even;
    Texture *odd;
    cuda_Texture *host_even;
    cuda_Texture *host_odd;
    cuda_CheckerTexture *host_cuda_texture;

public:
    CheckerTexture(float scale, Texture *even, Texture *odd) : inv_scale(1.0f / scale), even(even), odd(odd) {}
    __host__ CheckerTexture(float scale, const vec3 &c1, const vec3 &c2) {
        inv_scale = 1.0f / scale;
        even = new SolidTexture(c1);
        odd = new SolidTexture(c2);
    }

    __host__ ~CheckerTexture(){
        delete even;
        delete odd;
        delete host_even;
        delete host_odd;
        delete host_cuda_texture;
    }

    __host__ vec3 value(float u, float v, const vec3 &p) const override {
        int32_t xInt = static_cast<int32_t>(std::floor(inv_scale * p.x));
        int32_t yInt = static_cast<int32_t>(std::floor(inv_scale * p.y));
        int32_t zInt = static_cast<int32_t>(std::floor(inv_scale * p.z));

        bool isEven = (xInt + yInt + zInt) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

    __host__ virtual cuda_Texture *convertToDevice() override {
        host_even = even->convertToDevice();
        host_odd = odd->convertToDevice();

        cuda_Texture *dev_even;
        cuda_Texture *dev_odd;

        cudaMalloc(&dev_even, sizeof(cuda_Texture));
        cudaMalloc(&dev_odd, sizeof(cuda_Texture));

        cudaMemcpy(dev_even, host_even, sizeof(cuda_Texture), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_odd, host_odd, sizeof(cuda_Texture), cudaMemcpyHostToDevice);


        host_cuda_texture = new cuda_CheckerTexture(inv_scale, dev_even, dev_odd);

        cuda_CheckerTexture *dev_cuda_texture;

        cudaMalloc(&dev_cuda_texture, sizeof(cuda_Texture));
        cudaMemcpy(dev_cuda_texture, host_cuda_texture, sizeof(cuda_Texture), cudaMemcpyHostToDevice);

        return dev_cuda_texture;
    }
};
