#pragma once

// #include <glm/glm.hpp>
#include <vec.h>

class Ray {
public:
    vec3 origin;
    vec3 direction;

    Ray() {}
    Ray(const vec3 &origin, const vec3 &direction) : origin(origin), direction(direction) {}

    vec3 at(float t) const { return origin + direction * t; }
};
