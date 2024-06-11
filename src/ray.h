#pragma once

#include <glm/glm.hpp>

class Ray {
    glm::vec3 origin_;
    glm::vec3 direction_;

public:
    Ray() {}
    Ray(const glm::vec3 &origin, const glm::vec3 &direction) : origin_(origin), direction_(direction) {}

    const glm::vec3 &origin() const { return origin_; }
    const glm::vec3 &direction() const { return direction_; }

    glm::vec3 at(float t) const { return origin_ + direction_ * t; }
};
