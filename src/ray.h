#pragma once

#include <vec.h>

class Ray {
public:
    vec3 origin;
    vec3 direction;

    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const vec3 &origin, const vec3 &direction) : origin(origin), direction(direction) {}

    __host__ __device__ vec3 at(float t) const { return origin + direction * t; }

    __host__ __device__ void printRay() const {
        printf("origin(%f %f %f) direciton(%f %f %f)\n",origin.x, origin.y, origin.z, direction.x, direction.y, direction.z);
    }
};
