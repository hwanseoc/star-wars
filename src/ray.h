#pragma once

#include <glm/glm.hpp>

class Ray {
public:
    glm::vec3 origin;
    glm::vec3 direction;

    Ray() {}
    Ray(const glm::vec3 &origin, const glm::vec3 &direction) : origin(origin), direction(direction) {}

    glm::vec3 at(float t) const { return origin + direction * t; }
};
