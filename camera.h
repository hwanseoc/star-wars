#pragma once

#include <cmath>
#include <cassert>

#include <glm/glm.hpp>
#include <ray.h>

class PerspectiveCamera {
    int32_t height, width;
    glm::vec3 center, pixel00, dv, du;

public:
    PerspectiveCamera(
        glm::vec3 &center,
        glm::vec3& direction,
        glm::vec3& up,
        int32_t height,
        int32_t width,
        float fov
    ) : center(center), height(height), width(width) {
        float widthf = static_cast<float>(width);
        float heightf = static_cast<float>(height);

        float magnitude = 2.0f * std::tan(fov / 2.0f) / widthf;
        dv = glm::normalize(-up) * magnitude;
        du = glm::normalize(glm::cross(direction, up)) * magnitude;

        pixel00 = center + glm::normalize(direction)
                    - du * (widthf / 2.0f)
                    - dv * (heightf / 2.0f)
                    + du / (2.0f)
                    + dv / (2.0f);
    }

    Ray get_ray(int32_t h, int32_t w) {
        glm::vec3 direction = glm::normalize(
            pixel00 + dv * static_cast<float>(h) + du * static_cast<float>(w) - center
        );
        return Ray(center, direction);
    }
};
