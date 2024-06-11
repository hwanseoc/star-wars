#include <iostream>
#include <memory>

#include <lodepng.h>
#include <glm/glm.hpp>


#include <ray.h>
#include <camera.h>
#include <object.h>
#include <sphere.h>
#include <cmath>

int main(int argc, char *argv[]) {
    std::string filename = "output.png";

    // image
    int32_t width = 400;
    int32_t height = 225;
    std::vector<uint8_t> image(height * width * 4); // rgba

    // camera
    glm::vec3 center(0.0, 0.0, 0.0);
    glm::vec3 direction(0.0, 0.0, -1.0);
    glm::vec3 up(0.0, 1.0, 0.0);
    float fov = 103.0f / 360.0f * 2.0f * M_PI;
    int32_t samples = 100;
    int32_t max_depth = 50;
    PerspectiveCamera perspectiveCamera(center, direction, up, height, width, fov, samples, max_depth);

    // objects
    glm::vec3 red(1.0, 0.0, 0.0);
    glm::vec3 green(0.0, 1.0, 0.0);
    glm::vec3 blue(0.0, 0.0, 1.0);
    ObjectList world;
    world.add(std::make_shared<Sphere>(Sphere(glm::vec3(0.0, 0.0, -1.0), 0.5, red)));
    world.add(std::make_shared<Sphere>(Sphere(glm::vec3(0.0, -100.5, -1.0), 100, green)));

    // render
    perspectiveCamera.render(image, world);

    unsigned int error = lodepng::encode(filename, image, width, height);
    if (error) {
        std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
    }
}

