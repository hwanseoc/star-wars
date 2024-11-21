#pragma once

#include <random.h>
#include <object.h>
#include <texture.h>

class Material {
public:
    virtual std::tuple<bool, vec3, Ray> scatter(const Ray &r, const ColorHit &hit) const {
        return std::make_tuple(false, vec3(0, 0, 0), Ray());
    }

    virtual vec3 emitted(const ColorHit &hit) const {
        return vec3(0, 0, 0);
    }
};


class Lambertian : public Material {
    std::shared_ptr<Texture> texture;

public:
    Lambertian(const vec3 albedo) :  texture(std::make_shared<SolidTexture>(albedo)) {}
    Lambertian(const std::shared_ptr<Texture> &texture) :  texture(texture) {}

    std::tuple<bool, vec3, Ray> scatter(
        const Ray &r,
        const ColorHit &hit
    ) const override {
        vec3 scattered_direction = hit.normal + random_sphere();

        // Catch bad scatter direction
        if (std::fabs(scattered_direction.x) < 1e-8f &&
            std::fabs(scattered_direction.y) < 1e-8f &&
            std::fabs(scattered_direction.z) < 1e-8f) {
            scattered_direction = hit.normal;
        }

        Ray scattered = Ray(hit.point, normalize(scattered_direction));
        vec3 attenuation = texture->value(hit.u, hit.v, hit.point);
        return std::make_tuple(true, attenuation, scattered);
    }
};

class Metal : public Material {
    vec3 albedo;
    float fuzz;

public:
    Metal(const vec3 albedo, float fuzz) : albedo(albedo), fuzz(fuzz) {}

    std::tuple<bool, vec3, Ray> scatter(
        const Ray &r,
        const ColorHit &hit
    ) const override {
        vec3 reflected = reflect(r.direction, hit.normal);
        reflected = normalize(reflected) + random_sphere() * fuzz;
        Ray scattered = Ray(hit.point, normalize(reflected));
        vec3 attenuation = albedo;
        bool is_scattered = dot(scattered.direction, hit.normal) > 0;
        return std::make_tuple(is_scattered, attenuation, scattered);
    }

private:
    inline vec3 reflect(const vec3 &direction, const vec3 &normal) const {
        return direction - 2 * dot(direction, normal) * normal;
    }
};

class Dielectric : public Material {
    float refractive_index;

public:
    Dielectric(float refractive_index) : refractive_index(refractive_index) {}

    std::tuple<bool, vec3, Ray> scatter(
        const Ray &r,
        const ColorHit &hit
    ) const override {
        float refractive_index_face = hit.is_front ? (1.0f / refractive_index) : refractive_index;

        float cos_theta = dot(-r.direction, hit.normal);
        float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);

        vec3 direction;
        if (refractive_index_face * sin_theta > 1.0f || schlick_reflectance(cos_theta, refractive_index_face) > random_float()) {
            direction = reflect(r.direction, hit.normal);
        } else {
            direction = refract(r.direction, hit.normal, refractive_index_face);
        }

        return std::make_tuple(true, vec3(1.0, 1.0, 1.0), Ray(hit.point, direction));
    }
private:
    inline vec3 refract(const vec3 &direction, const vec3 &normal, float refractive_index_face) const {
        vec3 r_vertical = refractive_index_face * (direction + dot(-direction, normal) * normal);
        float r_vertical_sq = r_vertical.x * r_vertical.x + r_vertical.y * r_vertical.y + r_vertical.z * r_vertical.z;
        vec3 r_horizontal = -std::sqrt(std::fabs(1.0f - r_vertical_sq)) * normal;
        return r_vertical + r_horizontal;
    }

    inline vec3 reflect(const vec3 &direction, const vec3 &normal) const {
        return direction - 2 * dot(direction, normal) * normal;
    }

    inline float schlick_reflectance(float cos_theta, float refractive_index) const {
        // https://en.wikipedia.org/wiki/Schlick%27s_approximation
        float r0 = std::pow((1.0f - refractive_index) / (1.0f + refractive_index), 2.0f);
        return r0 + (1.0f-r0) * std::pow((1-cos_theta) , 5.0f);
    }
};

class DiffuseLight : public Material {
    std::shared_ptr<Texture> texture;

public:
    DiffuseLight(std::shared_ptr<Texture> texture) : texture(texture) {}
    DiffuseLight(const vec3& emit) : texture(std::make_shared<SolidTexture>(emit)) {}

    vec3 emitted(const ColorHit &hit) const override {
        return texture->value(hit.u, hit.v, hit.point);
    }
};
