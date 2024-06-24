#pragma once

#include <random>

#include <glm/gtc/random.hpp>

inline float random_float() {
    static std::uniform_real_distribution<float> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline glm::vec2 random_disk() {
    return glm::diskRand(1.0f);
}

inline glm::vec3 random_sphere() {
    return glm::sphericalRand(1.0f);
}

inline glm::vec3 random_hemisphere(const glm::vec3& normal) {
    glm::vec3 ret = random_sphere();
    if (glm::dot(ret, normal) > 0.0f){
        return ret;
    } else {
        return -ret;
    }
}
