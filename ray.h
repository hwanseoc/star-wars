#pragma once

#include <glm/glm.hpp>

class Ray {
    glm::vec3 origin_;
    glm::vec3 direction_;

public:
    Ray() {}
    Ray(const glm::vec3 &origin, const glm::vec3 &direction) : origin_(origin), direction_(direction) {}

    const glm::vec3& origin() const { return origin_; }
    const glm::vec3& direction() const { return direction_; }

    glm::vec3 at(float t) const { return origin_ + direction_ * t; }

    void print() const {
        std::cout << " origin.x:" << origin_.x << " origin.y:" << origin_.y << " origin.z:" << origin_.z;
        std::cout << " direction.x:" << direction_.x << " direction.y:" << direction_.y << " direction.z:" << direction_.z << std::endl;
    }
};
