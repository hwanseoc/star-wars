#include <iostream>
#include <lodepng.h>

#include <glm/glm.hpp>


#include <ray.h>
#include <camera.h>
#include <sphere.h>

int main(int argc, char *argv[]) {
    std::string filename = "output.png";

    // image
    int32_t width = 256;
    int32_t height = 256;
    std::vector<uint8_t> image(height * width * 4); // rgba

    // camera
    glm::vec3 center(0.0, 0.0, 2.0);
    glm::vec3 direction(0.0, 0.0, -1.0);
    glm::vec3 up(0.0, 1.0, 0.0);
    PerspectiveCamera perspectiveCamera(center, direction, up, 256, 256, 3.14 / 2);

    // objects
    Sphere sphere(
        glm::vec3(0.0, 0.0, 0.0),
        0.7
    );

    for (int32_t h = 0; h < height; h++) {
        for (int32_t w = 0; w < width; w++) {
            // std::clog << "\rPixels remaining: " << h * width + w << " out of " << height * width << std::flush;

            float alpha = (static_cast<float>(height) - static_cast<float>(h)) / static_cast<float>(height);
            glm::vec3 pixel = (1.0f - alpha) * glm::vec3(1.0, 1.0, 1.0) + alpha * glm::vec3(0.5, 0.7, 1.0);

            Ray r = perspectiveCamera.get_ray(h, w);


            Hit hit = sphere.hit(r, 0.0, 1000.0);
            if (hit.is_hit) {
                pixel = glm::vec3(1.0, 0.0, 0.0);
            }

            uint8_t ir = static_cast<uint8_t>(255.999 * pixel.x);
            uint8_t ig = static_cast<uint8_t>(255.999 * pixel.y);
            uint8_t ib = static_cast<uint8_t>(255.999 * pixel.z);

            image[h * width * 4 + w * 4 + 0] = ir;
            image[h * width * 4 + w * 4 + 1] = ig;
            image[h * width * 4 + w * 4 + 2] = ib;
            image[h * width * 4 + w * 4 + 3] = 255;
        }
    }

    unsigned int error = lodepng::encode(filename, image, width, height);
    if (error) {
        std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
    }
}

