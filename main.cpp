#include <iostream>
#include <lodepng.h>

#include <vec3.h>
#include <ray.h>

int main(int argc, char *argv[]) {
    std::string filename = "output.png";

    int32_t width = 256;
    int32_t height = 256;
    std::vector<uint8_t> image(height * width * 4); // rgba

    int32_t pixels = height * width;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            std::clog << "\rPixels remaining: " << pixels << " out of " << height * width << std::flush;

            Vec3 pixel(float(i) / (width-1), float(j) / (height-1), 0.0);

            uint8_t ir = static_cast<uint8_t>(255.999 * pixel.r());
            uint8_t ig = static_cast<uint8_t>(255.999 * pixel.g());
            uint8_t ib = static_cast<uint8_t>(255.999 * pixel.b());

            image[j * width * 4 + i * 4 + 0] = ir;
            image[j * width * 4 + i * 4 + 1] = ig;
            image[j * width * 4 + i * 4 + 2] = ib;
            image[j * width * 4 + i * 4 + 3] = 255;
        }
    }

    unsigned int error = lodepng::encode(filename, image, width, height);
    if (error) {
        std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
    }
}

