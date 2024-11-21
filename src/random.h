#pragma once

#include <random>
#include <vec.h>
// #include <glm/gtc/random.hpp>

inline float random_float() {
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    static std::mt19937 generator;
    return distribution(generator);
}

inline vec3 random_disk() {
    return vec3(random_float() - 0.5f, random_float() - 0.5f, 0.0f);
}

inline vec3 random_sphere() {
    while (true) {
        vec3 p = vec3(random_float()*2.0f - 1.0f, random_float()*2.0f - 1.0f, random_float()*2.0f - 1.0f);
        float lensq = p.length_squared();
        if (1e-6f < lensq && lensq <= 1.0f) {
            return p / sqrt(lensq);
        }
    }
}

inline vec3 random_hemisphere(const vec3& normal) {
    vec3 ret = random_sphere();
    if (dot(ret, normal) > 0.0f){
        return ret;
    } else {
        return -ret;
    }
}
