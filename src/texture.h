#pragma once

#include <algorithm>
#include <memory>

// #include <glm/glm.hpp>
#include <vec.h>

class Texture {
public:
    __host__ __device__ virtual ~Texture() = default;

    __host__ virtual vec3 value(float u, float v, const vec3 &p) const = 0;
};


class SolidTexture : public Texture {
    vec3 albedo;
public:
    __host__ SolidTexture(const vec3 &albedo) : albedo(albedo) {}

    __host__ vec3 value(float u, float v, const vec3& p) const override {
        return albedo;
    }
};


class CheckerTexture : public Texture {
    float inv_scale;
    Texture *even;
    Texture *odd;

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
    }

    __host__ vec3 value(float u, float v, const vec3 &p) const override {
        int32_t xInt = static_cast<int32_t>(std::floor(inv_scale * p.x));
        int32_t yInt = static_cast<int32_t>(std::floor(inv_scale * p.y));
        int32_t zInt = static_cast<int32_t>(std::floor(inv_scale * p.z));

        bool isEven = (xInt + yInt + zInt) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }
};
