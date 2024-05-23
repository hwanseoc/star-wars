#pragma once

#include <vec3.h>

class Ray {
        Vec3 origin_;
        Vec3 direction_;

    public:
        Ray() {}
        Ray(const Vec3 &origin, const Vec3 &direction) : origin_(origin), direction_(direction) {}

        const Vec3& origin() const { return origin_; }
        const Vec3& direction() const { return direction_; }

        
        const Vec3 at(float t) const {
            return origin_ + direction_ * t;
        }
};

