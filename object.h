#pragma once

#include <glm/glm.hpp>

#include <ray.h>

class Hit {
public:
    bool is_hit;
    glm::vec3 point;
    glm::vec3 normal;
    float t;
};

class Object {
public:
    virtual Hit hit(const Ray& r, float tmin, float tmax) const = 0;
};
