#pragma once

#include <random.h>
#include <object.h>
#include <texture.h>

class Material {
public:
    virtual std::tuple<bool, glm::vec3, Ray> scatter(const Ray &r, const ColorHit &hit) const = 0;
};


class Lambertian : public Material {
    std::shared_ptr<Texture> texture;

public:
    Lambertian(const glm::vec3 albedo) :  texture(std::make_shared<SolidTexture>(albedo)) {}
    Lambertian(const std::shared_ptr<Texture> &texture) :  texture(texture) {}

    std::tuple<bool, glm::vec3, Ray> scatter(
        const Ray &r,
        const ColorHit &hit
    ) const override {
        glm::vec3 scattered_direction = hit.normal + random_sphere();

        // Catch bad scatter direction
        if (std::fabs(scattered_direction.x) < 1e-8f &&
            std::fabs(scattered_direction.y) < 1e-8f &&
            std::fabs(scattered_direction.z) < 1e-8f) {
            scattered_direction = hit.normal;
        }

        Ray scattered = Ray(hit.point, glm::normalize(scattered_direction));
        glm::vec3 attenuation = texture->value(hit.u, hit.v, hit.point);
        return std::make_tuple(true, attenuation, scattered);
    }
};

class Metal : public Material {
    glm::vec3 albedo;
    float fuzz;

public:
    Metal(const glm::vec3 albedo, float fuzz) : albedo(albedo), fuzz(fuzz) {}

    std::tuple<bool, glm::vec3, Ray> scatter(
        const Ray &r,
        const ColorHit &hit
    ) const override {
        glm::vec3 reflected = reflect(r.direction(), hit.normal);
        reflected = glm::normalize(reflected) + random_sphere() * fuzz;
        Ray scattered = Ray(hit.point, glm::normalize(reflected));
        glm::vec3 attenuation = albedo;
        bool is_scattered = glm::dot(scattered.direction(), hit.normal) > 0;
        return std::make_tuple(is_scattered, attenuation, scattered);
    }

private:
    inline glm::vec3 reflect(const glm::vec3 &direction, const glm::vec3 &normal) const {
        return direction - 2 * glm::dot(direction, normal) * normal;
    }
};

class Dielectric : public Material {
    float refractive_index;

public:
    Dielectric(float refractive_index) : refractive_index(refractive_index) {}

    std::tuple<bool, glm::vec3, Ray> scatter(
        const Ray &r,
        const ColorHit &hit
    ) const override {
        float refractive_index_face = hit.is_front ? (1.0f / refractive_index) : refractive_index;

        float cos_theta = glm::dot(-r.direction(), hit.normal);
        float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);

        glm::vec3 direction;
        if (refractive_index_face * sin_theta > 1.0f || schlick_reflectance(cos_theta, refractive_index_face) > random_float()) {
            direction = reflect(r.direction(), hit.normal);
        } else {
            direction = refract(r.direction(), hit.normal, refractive_index_face);
        }

        return std::make_tuple(true, glm::vec3(1.0, 1.0, 1.0), Ray(hit.point, direction));
    }
private:
    inline glm::vec3 refract(const glm::vec3 &direction, const glm::vec3 &normal, float refractive_index_face) const {
        glm::vec3 r_vertical = refractive_index_face * (direction + glm::dot(-direction, normal) * normal);
        float r_vertical_sq = r_vertical.x * r_vertical.x + r_vertical.y * r_vertical.y + r_vertical.z * r_vertical.z;
        glm::vec3 r_horizontal = -std::sqrt(std::fabs(1.0f - r_vertical_sq)) * normal;
        return r_vertical + r_horizontal;
    }

    inline glm::vec3 reflect(const glm::vec3 &direction, const glm::vec3 &normal) const {
        return direction - 2 * glm::dot(direction, normal) * normal;
    }

    inline float schlick_reflectance(float cos_theta, float refractive_index) const {
        // https://en.wikipedia.org/wiki/Schlick%27s_approximation
        float r0 = std::pow((1.0f - refractive_index) / (1.0f + refractive_index), 2.0f);
        return r0 + (1.0f-r0) * std::pow((1-cos_theta) , 5.0f);
    }
};

