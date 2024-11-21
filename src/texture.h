#pragma once

#include <algorithm>
#include <memory>

#include <lodepng.h>
// #include <glm/glm.hpp>
#include <vec.h>

class Texture {
public:
    virtual ~Texture() = default;

    virtual vec3 value(float u, float v, const vec3 &p) const = 0;
};


class SolidTexture : public Texture {
    vec3 albedo;
public:
    SolidTexture(const vec3 &albedo) : albedo(albedo) {}

    vec3 value(float u, float v, const vec3& p) const override {
        return albedo;
    }
};


class CheckerTexture : public Texture {
    float inv_scale;
    std::shared_ptr<Texture> even;
    std::shared_ptr<Texture> odd;

public:
    CheckerTexture(float scale, std::shared_ptr<Texture> even, std::shared_ptr<Texture> odd) : inv_scale(1.0f / scale), even(even), odd(odd) {}
    CheckerTexture(float scale, const vec3 &c1, const vec3 &c2) : CheckerTexture(scale, std::make_shared<SolidTexture>(c1), std::make_shared<SolidTexture>(c2)) {}

    vec3 value(float u, float v, const vec3 &p) const override {
        int32_t xInt = static_cast<int32_t>(std::floor(inv_scale * p.x));
        int32_t yInt = static_cast<int32_t>(std::floor(inv_scale * p.y));
        int32_t zInt = static_cast<int32_t>(std::floor(inv_scale * p.z));

        bool isEven = (xInt + yInt + zInt) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }
};

class ImageTexture : public Texture {
    std::vector<uint8_t> image;
    uint32_t height, width;

public:
    ImageTexture(const std::string &filename) {
        uint32_t error = lodepng::decode(image, width, height, filename);
        if (error) {
            std::cout << "decoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
        }
    }

    vec3 value(float u, float v, const vec3 &p) const override {
        int32_t i = static_cast<int32_t>(u * static_cast<float>(width));
        int32_t j = static_cast<int32_t>((1.0f - v) * static_cast<float>(height));

        i = std::clamp(i, 0, static_cast<int32_t>(width) - 1);
        j = std::clamp(j, 0, static_cast<int32_t>(height) - 1);

        float r = static_cast<float>(image[j * width * 4 + i * 4 + 0]) / 256.0f;
        float g = static_cast<float>(image[j * width * 4 + i * 4 + 1]) / 256.0f;
        float b = static_cast<float>(image[j * width * 4 + i * 4 + 2]) / 256.0f;

        return vec3(r, g, b);
    }

};
