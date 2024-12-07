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

    virtual int8_t type() const {
        return TEX_TYPE_DEFAULT;
    }

};










class cuda_SolidTexture : public cuda_Texture {
    vec3 albedo;
public:
    __host__ cuda_SolidTexture(const vec3 albedo) : albedo(albedo) {}

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
        if (host_cuda_texture) delete host_cuda_texture;
    }

    vec3 value(float u, float v, const vec3& p) const override {
        return albedo;
    }

    cuda_SolidTexture *convertToDevice() override {
        host_cuda_texture = new cuda_SolidTexture(albedo);
        cuda_SolidTexture *dev_cuda_texture;

        cudaMalloc(&dev_cuda_texture, sizeof(cuda_SolidTexture));
        cudaMemcpy(dev_cuda_texture, host_cuda_texture, sizeof(cuda_SolidTexture), cudaMemcpyHostToDevice);

        return dev_cuda_texture;
    }

    int8_t type() const override {
        return TEX_TYPE_SOLIDTEXTURE;
    }
};










class cuda_CheckerTexture : public cuda_Texture {
    float inv_scale;
    cuda_Texture *even;
    cuda_Texture *odd;
    int8_t even_tex_type;
    int8_t odd_tex_type;

public:
    cuda_CheckerTexture(
        float inv_scale,
        cuda_Texture *even,
        int8_t even_tex_type,
        cuda_Texture *odd,
        int8_t odd_tex_type
    ) : inv_scale(inv_scale), even(even), even_tex_type(even_tex_type), odd(odd), odd_tex_type(odd_tex_type) {}
    __host__ ~cuda_CheckerTexture() {
        if (even) {
            cudaFree(even);
            even = nullptr;
        }
        if (odd) {
            cudaFree(odd);
            odd = nullptr;
        }
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
    std::shared_ptr<Texture> even;
    std::shared_ptr<Texture> odd;
    cuda_CheckerTexture *host_cuda_texture;

public:
    CheckerTexture(float scale, Texture *even, Texture *odd) : inv_scale(1.0f / scale), even(even), odd(odd) {}
    CheckerTexture(float scale, const vec3 &c1, const vec3 &c2) {
        inv_scale = 1.0f / scale;
        even = std::make_shared<SolidTexture>(c1);
        odd = std::make_shared<SolidTexture>(c2);
    }

    ~CheckerTexture(){
        if (host_cuda_texture) delete host_cuda_texture;
    }

    vec3 value(float u, float v, const vec3 &p) const override {
        int32_t xInt = static_cast<int32_t>(std::floor(inv_scale * p.x));
        int32_t yInt = static_cast<int32_t>(std::floor(inv_scale * p.y));
        int32_t zInt = static_cast<int32_t>(std::floor(inv_scale * p.z));

        bool isEven = (xInt + yInt + zInt) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

    cuda_CheckerTexture *convertToDevice() override {
        host_cuda_texture = new cuda_CheckerTexture(inv_scale, even->convertToDevice(), even->type(), odd->convertToDevice(), odd->type());

        cuda_CheckerTexture *dev_cuda_texture;

        cudaMalloc(&dev_cuda_texture, sizeof(cuda_CheckerTexture));
        cudaMemcpy(dev_cuda_texture, host_cuda_texture, sizeof(cuda_CheckerTexture), cudaMemcpyHostToDevice);

        return dev_cuda_texture;
    }

    int8_t type() const override {
        return TEX_TYPE_CHECKERTEXTURE;
    }
};
