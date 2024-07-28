// #pragma once

// #include <glm/glm.hpp>

// class Texture {
// public:
//     virtual ~Texture() = default;

//     virtual glm::vec3 value(float u, float v, const glm::vec3 &p) const = 0;
// };


// class SolidTexture : public Texture {
// private:
//     glm::vec3 albedo;
// public:
//     SolidTexture(const glm::vec3 &albedo) : albedo(albedo) {}

//     glm::vec3 value(float u, float v, const glm::vec3& p) const override {
//         return albedo;
//     }
// };


// class CheckerTexture : public Texture {
// private:
//     float inv_scale;
//     sh
// }