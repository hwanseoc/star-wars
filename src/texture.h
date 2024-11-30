#pragma once

#include <algorithm>
#include <memory>

#include <vec.h>

#define TEX_TYPE_DEFAULT 0
#define TEX_TYPE_SOLIDTEXTURE 1
#define TEX_TYPE_CHECKERTEXTURE 2

class cuda_Texture {};

class Texture {
public:
    virtual ~Texture() = default;

    virtual vec3 value(float u, float v, const vec3 &p) const = 0;

    virtual cuda_Texture *convertToDevice() = 0;

    virtual int32_t type() const {
        return TEX_TYPE_DEFAULT;
    }

};










class cuda_SolidTexture : public cuda_Texture {
    vec3 albedo;
public:
    __host__ cuda_SolidTexture(const vec3 &albedo) : albedo(albedo) {}

    __device__ vec3 value(float u, float v, const vec3& p) const {
        return albedo;
    }
};

class SolidTexture : public Texture {
    vec3 albedo;
    cuda_SolidTexture *host_cuda_texture;
public:
    SolidTexture(const vec3 &albedo) : albedo(albedo) {}
    ~SolidTexture() {
        delete host_cuda_texture;
    }

    vec3 value(float u, float v, const vec3& p) const override {
        return albedo;
    }

    cuda_Texture *convertToDevice() override {
        host_cuda_texture = new cuda_SolidTexture(albedo);

        cuda_SolidTexture *dev_cuda_texture;

        cudaMalloc(&dev_cuda_texture, sizeof(cuda_Texture));
        cudaMemcpy(dev_cuda_texture, host_cuda_texture, sizeof(cuda_Texture), cudaMemcpyHostToDevice);

        return dev_cuda_texture;
    }

    int32_t type() const override {
        return TEX_TYPE_SOLIDTEXTURE;
    }
};










class cuda_CheckerTexture : public cuda_Texture {
    float inv_scale;
    cuda_Texture *even;
    cuda_Texture *odd;
    int32_t even_tex_type;
    int32_t odd_tex_type;

public:
    cuda_CheckerTexture(
        float inv_scale,
        cuda_Texture *even,
        int32_t even_tex_type,
        cuda_Texture *odd,
        int32_t odd_tex_type
    ) : inv_scale(inv_scale), even(even), even_tex_type(even_tex_type), odd(odd), odd_tex_type(odd_tex_type) {}
    __host__ ~cuda_CheckerTexture(){
        cudaFree(even);
        cudaFree(odd);
    }

    __device__ vec3 value(float u, float v, const vec3 &p) const {
        int32_t xInt = static_cast<int32_t>(std::floor(inv_scale * p.x));
        int32_t yInt = static_cast<int32_t>(std::floor(inv_scale * p.y));
        int32_t zInt = static_cast<int32_t>(std::floor(inv_scale * p.z));

        bool isEven = (xInt + yInt + zInt) % 2 == 0;

        vec3 ret_vec = vec3();

        if (isEven) {
            switch (even_tex_type)
            {
            case TEX_TYPE_SOLIDTEXTURE:
                ret_vec = ((cuda_SolidTexture *)even)->value(u, v, p);
                break;
            
            default:
                break;
            }
        } else {
            switch (odd_tex_type)
            {
            case TEX_TYPE_SOLIDTEXTURE:
                ret_vec = ((cuda_SolidTexture *)odd)->value(u, v, p);
                break;
            
            default:
                break;
            }
        }

        return ret_vec;
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
    CheckerTexture(float scale, const vec3 &c1, const vec3 &c2) {
        inv_scale = 1.0f / scale;
        even = new SolidTexture(c1);
        odd = new SolidTexture(c2);
    }

    ~CheckerTexture(){
        delete even;
        delete odd;
        delete host_even;
        delete host_odd;
        delete host_cuda_texture;
    }

    vec3 value(float u, float v, const vec3 &p) const override {
        int32_t xInt = static_cast<int32_t>(std::floor(inv_scale * p.x));
        int32_t yInt = static_cast<int32_t>(std::floor(inv_scale * p.y));
        int32_t zInt = static_cast<int32_t>(std::floor(inv_scale * p.z));

        bool isEven = (xInt + yInt + zInt) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

    cuda_Texture *convertToDevice() override {
        host_even = even->convertToDevice();
        host_odd = odd->convertToDevice();

        cuda_Texture *dev_even;
        cuda_Texture *dev_odd;

        cudaMalloc(&dev_even, sizeof(cuda_Texture));
        cudaMalloc(&dev_odd, sizeof(cuda_Texture));

        cudaMemcpy(dev_even, host_even, sizeof(cuda_Texture), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_odd, host_odd, sizeof(cuda_Texture), cudaMemcpyHostToDevice);


        host_cuda_texture = new cuda_CheckerTexture(inv_scale, dev_even, even->type(), dev_odd, odd->type());

        cuda_CheckerTexture *dev_cuda_texture;

        cudaMalloc(&dev_cuda_texture, sizeof(cuda_Texture));
        cudaMemcpy(dev_cuda_texture, host_cuda_texture, sizeof(cuda_Texture), cudaMemcpyHostToDevice);

        return dev_cuda_texture;
    }

    int32_t type() const override {
        return TEX_TYPE_CHECKERTEXTURE;
    }
};
