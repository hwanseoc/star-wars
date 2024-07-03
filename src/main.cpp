#include <iostream>
#include <memory>
#include <cmath>
#include <numbers>

#include <lodepng.h>
#include <glm/glm.hpp>

#include <camera.h>
#include <object.h>
#include <sphere.h>
#include <material.h>
#include <ray.h>

int main(int argc, char *argv[]) {
    std::string filename = "output.png";

    // image
    int32_t width = 1200;
    int32_t height = 675;
    std::vector<uint8_t> image(height * width * 4); // rgba

    // camera
    glm::vec3 center(13.0, 2.0, 3.0);
    glm::vec3 direction(-13.0, -2.0, -3.0);
    direction = glm::normalize(direction);
    glm::vec3 up(0.0, 1.0, 0.0);
    //float fov = 90.00f / 360.0f * 2.0f * std::numbers::pi_v<float>;
    float fov = 0.607537f;
    int32_t samples = 500;
    int32_t max_depth = 50;
    float focal_distance = 10.0f;
    float defocus_angle = 0.6f / 180.0f * std::numbers::pi_v<float>;

    PerspectiveCamera perspectiveCamera(
        center,
        direction,
        up,
        height,
        width,
        fov,
        focal_distance,
        defocus_angle,
        samples,
        max_depth
    );

    // materials
    BVHLeaf world;

    std::shared_ptr<Material> material_ground = std::make_shared<Lambertian>(glm::vec3(0.5, 0.5, 0.5));
    world.add(std::make_shared<Sphere>(glm::vec3(0.0, -1000.0, 0.0), 1000.0, material_ground));

    std::shared_ptr<Material> material1 = std::make_shared<Dielectric>(1.5f);
    std::shared_ptr<Material> material2 = std::make_shared<Lambertian>(glm::vec3(0.4, 0.2, 0.1));
    std::shared_ptr<Material> material3 = std::make_shared<Metal>(glm::vec3(0.7, 0.6, 0.5), 0.0);
    world.add(std::make_shared<Sphere>(glm::vec3( 0.0, 1.0, 0.0), 1.0, material1));
    world.add(std::make_shared<Sphere>(glm::vec3( -4.0, 1.0, 0.0), 1.0, material2));
    world.add(std::make_shared<Sphere>(glm::vec3( 4.0, 1.0, 0.0), 1.0, material3));

    for (float a = -11.0f; a < 11.0f; a = a + 1.0f) {
        for (float b = -11.0f; b < 11.0f; b = b + 1.0f) {
            float material_choice = random_float();

            glm::vec3 center(a + 0.9f * random_float(), 0.2f, b + 0.9f * random_float());

            if (glm::length(center - glm::vec3(4, 0.2, 0.0)) > 0.9f) {
                std::shared_ptr<Material> material;

                if (material_choice < 0.8f) {
                    // diffuse
                    glm::vec3 albedo = glm::vec3(random_float() * random_float(), random_float() * random_float(), random_float() * random_float());
                    material = std::make_shared<Lambertian>(albedo);
                } else if (material_choice < 0.95f) {
                    // metal
                    glm::vec3 albedo = glm::vec3(0.5f + 0.5f * random_float(), 0.5f + 0.5f * random_float(), 0.5f + 0.5f * random_float());
                    float fuzz = random_float() * 0.5f;
                    material = std::make_shared<Metal>(albedo, fuzz);
                } else {
                    // dielectric
                    material = std::make_shared<Dielectric>(1.5f);
                }

                world.add(std::make_shared<Sphere>(center, 0.2, material));
            }
        }
    }
    BVHNode world_bvh(world);

    // render
    perspectiveCamera.render(image, world_bvh);

    unsigned int error = lodepng::encode(filename, image, width, height);
    if (error) {
        std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
    }
}

