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

public:
    Metal(const glm::vec3 albedo) : albedo_(albedo) {}

    std::tuple<bool, glm::vec3, Ray> scatter(
        const Ray &r,
        const Hit &hit
    ) const override {
        glm::vec3 reflected = reflect(r.direction(), hit.normal);
        Ray scattered = Ray(hit.point, glm::normalize(reflected));
        glm::vec3 attenuation = albedo_;
        return std::make_tuple(true, attenuation, scattered);
    }

private:
    inline glm::vec3 reflect(const glm::vec3 &direction, const glm::vec3 &normal) const {
        return direction - 2 * glm::dot(direction, normal) * normal;
    }
};
