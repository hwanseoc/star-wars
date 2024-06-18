#pragma once

#include <random.h>
#include <object.h>

class Material {
public:
    virtual std::tuple<bool, glm::vec3, Ray> scatter(const Ray &r, const Hit &hit) const = 0;
};


class Lambertian : public Material {
    glm::vec3 albedo_;

public:
    Lambertian(const glm::vec3 albedo) :  albedo_(albedo) {}

    std::tuple<bool, glm::vec3, Ray> scatter(
        const Ray &r,
        const Hit &hit
    ) const override {
        glm::vec3 scattered_direction = hit.normal + random_sphere();

        // Catch bad scatter direction
        if (std::fabs(scattered_direction.x) < 1e-8 &&
            std::fabs(scattered_direction.y) < 1e-8 &&
            std::fabs(scattered_direction.z) < 1e-8) {
            scattered_direction = hit.normal;
        }

        Ray scattered = Ray(hit.point, glm::normalize(scattered_direction));
        glm::vec3 attenuation = albedo_;
        return std::make_tuple(true, attenuation, scattered);
    }
};

class Metal : public Material {
    glm::vec3 albedo_;
    float fuzz_;

public:
    Metal(const glm::vec3 albedo, float fuzz) : albedo_(albedo), fuzz_(fuzz) {}

    std::tuple<bool, glm::vec3, Ray> scatter(
        const Ray &r,
        const Hit &hit
    ) const override {
        glm::vec3 reflected = reflect(r.direction(), hit.normal);
        reflected = glm::normalize(reflected) + random_sphere(fuzz_);
        Ray scattered = Ray(hit.point, glm::normalize(reflected));
        glm::vec3 attenuation = albedo_;
        bool is_scattered = glm::dot(scattered.direction(), hit.normal) > 0;
        return std::make_tuple(is_scattered, attenuation, scattered);
    }

private:
    inline glm::vec3 reflect(const glm::vec3 &direction, const glm::vec3 &normal) const {
        return direction - 2 * glm::dot(direction, normal) * normal;
    }
};

class Dielectric : public Material {
    float refractive_index_;

public:
    Dielectric(float refractive_index) : refractive_index_(refractive_index) {}

    std::tuple<bool, glm::vec3, Ray> scatter(
        const Ray &r,
        const Hit &hit
    ) const override {
        float refractive_index_face = hit.is_front ? (1.0f / refractive_index_) : refractive_index_;
        glm::vec3 refracted = refract(r.direction(), hit.normal, refractive_index_face);
        return std::make_tuple(true, glm::vec3(1.0, 1.0, 1.0), Ray(hit.point, refracted));
    }
private:
    inline glm::vec3 refract(const glm::vec3 &direction, const glm::vec3 &normal, float refractive_index_face) const {
        glm::vec3 r_vertical = refractive_index_face * (direction + glm::dot(-direction, normal) * normal);
        float r_vertical_sq = r_vertical.x * r_vertical.x + r_vertical.y * r_vertical.y + r_vertical.z * r_vertical.z;
        glm::vec3 r_horizontal = -std::sqrt(std::fabs(1.0f - r_vertical_sq)) * normal;
        return r_vertical + r_horizontal;
    }
};

