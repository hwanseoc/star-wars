#pragma once

#include <random.h>
#include <object.h>
#include <texture.h>

class Material {
public:
    __device__ virtual vec3 emitted(const ColorHit &hit) const {
        return vec3(0, 0, 0);
    }

    __device__ virtual void scatter(const Ray &r, const ColorHit &hit, bool &is_scattered, vec3 &attenuation, Ray &scattered) const {
        is_scattered = false;
        attenuation = vec3(0, 0, 0);
        scattered = Ray();
    }

};


class Lambertian : public Material {
    Texture *texture;

public:
    __host__ Lambertian(const vec3 albedo) {
        Texture *host_tex = new SolidTexture(albedo);
        cudaMalloc(&texture, sizeof(Texture));
        cudaMemcpy(texture, host_tex, sizeof(Texture), cudaMemcpyHostToDevice);

        delete host_tex;
    }
    __host__ Lambertian(Texture *texture) :  texture(texture) {}

    __host__ ~Lambertian(){
        cudaFree(texture);
    }

    __device__ virtual void scatter(const Ray &r, const ColorHit &hit, bool &is_scattered, vec3 &attenuation, Ray &scattered) const override{
        vec3 scattered_direction = hit.normal + random_sphere();

        // Catch bad scatter direction
        if (std::fabs(scattered_direction.x) < 1e-8f &&
            std::fabs(scattered_direction.y) < 1e-8f &&
            std::fabs(scattered_direction.z) < 1e-8f) {
            scattered_direction = hit.normal;
        }
        is_scattered = true;
        scattered = Ray(hit.point, normalize(scattered_direction));
        attenuation = texture->value(hit.u, hit.v, hit.point);
    }

};

class Metal : public Material {
    vec3 albedo;
    float fuzz;

public:
    __host__ __device__  Metal(const vec3 albedo, float fuzz) : albedo(albedo), fuzz(fuzz) {}

    __device__ virtual void scatter(const Ray &r, const ColorHit &hit, bool &is_scattered, vec3 &attenuation, Ray &scattered) const override {
        vec3 reflected = reflect(r.direction, hit.normal);
        reflected = normalize(reflected) + random_sphere() * fuzz;

        scattered = Ray(hit.point, normalize(reflected));
        attenuation = albedo;
        is_scattered = dot(scattered.direction, hit.normal) > 0;
    }
    

private:
    __device__ inline vec3 reflect(const vec3 &direction, const vec3 &normal) const {
        return direction - 2 * dot(direction, normal) * normal;
    }
};

class Dielectric : public Material {
    float refractive_index;

public:
    __host__ __device__ Dielectric(float refractive_index) : refractive_index(refractive_index) {}

    __device__ virtual void scatter(const Ray &r, const ColorHit &hit, bool &is_scattered, vec3 &attenuation, Ray &scattered) const override {
        float refractive_index_face = hit.is_front ? (1.0f / refractive_index) : refractive_index;

        float cos_theta = dot(-r.direction, hit.normal);
        float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);

        vec3 direction;
        if (refractive_index_face * sin_theta > 1.0f || schlick_reflectance(cos_theta, refractive_index_face) > random_float()) {
            direction = reflect(r.direction, hit.normal);
        } else {
            direction = refract(r.direction, hit.normal, refractive_index_face);
        }

        is_scattered = true;
        attenuation = vec3(1.0, 1.0, 1.0);
        scattered = Ray(hit.point, direction);
    }
private:
    __device__ inline vec3 refract(const vec3 &direction, const vec3 &normal, float refractive_index_face) const {
        vec3 r_vertical = refractive_index_face * (direction + dot(-direction, normal) * normal);
        float r_vertical_sq = r_vertical.x * r_vertical.x + r_vertical.y * r_vertical.y + r_vertical.z * r_vertical.z;
        vec3 r_horizontal = -std::sqrt(std::fabs(1.0f - r_vertical_sq)) * normal;
        return r_vertical + r_horizontal;
    }

    __device__ inline vec3 reflect(const vec3 &direction, const vec3 &normal) const {
        return direction - 2 * dot(direction, normal) * normal;
    }

    __device__ inline float schlick_reflectance(float cos_theta, float refractive_index) const {
        // https://en.wikipedia.org/wiki/Schlick%27s_approximation
        float r0 = std::pow((1.0f - refractive_index) / (1.0f + refractive_index), 2.0f);
        return r0 + (1.0f-r0) * std::pow((1-cos_theta) , 5.0f);
    }
};

class DiffuseLight : public Material {
    Texture *texture;

public:
    __host__ __device__ DiffuseLight(Texture *texture) : texture(texture) {}
    __host__ __device__ DiffuseLight(const vec3& emit) {
        Texture *host_tex = new SolidTexture(emit);
        cudaMalloc(&texture, sizeof(Texture));
        cudaMemcpy(texture, host_tex, sizeof(Texture), cudaMemcpyHostToDevice);
        delete host_tex;
    }
    __host__ __device__ ~DiffuseLight(){
        cudaFree(texture);
    }

    __device__ vec3 emitted(const ColorHit &hit) const override {
        return texture->value(hit.u, hit.v, hit.point);
    }
};
