#include <iostream>
#include <fstream>
#include <memory>
#include <cmath>
#include <numbers>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <typeinfo>

#include <vec.h>
#include <ray.h>
#include <camera.h>
#include <object.h>
#include <material.h>
#include <sphere.h>
#include <triangle.h>
#include <texture.h>

// void add_object(
//     World &world,
//     const std::string &filename,
//     const vec3 &translate,
//     const vec3 &rotate_axis,
//     const float rotate_angle, //degree
//     const vec3 &scale,
//     std::shared_ptr<Material> &material
// ) {
//     std::vector<vec3> vertices;

//     std::ifstream ifs(filename);

//     std::string line;
//     while (getline(ifs, line)) {
//         std::istringstream iss(line);
//         std::string type;
//         iss >> type;
//         if (type == "v") {
//             vec3 v;
//             iss >> v.x >> v.y >> v.z;
//             vertices.push_back(v);
//         } else if (type == "f") {
//             int32_t f[3];
//             iss >> f[0] >> f[1] >> f[2];

//             glm::mat4 transform_matrix = glm::mat4(1.0f);
//             transform_matrix = glm::translate(transform_matrix, translate);
//             transform_matrix = glm::rotate(transform_matrix, glm::radians(rotate_angle), rotate_axis);
//             transform_matrix = glm::scale(transform_matrix, scale);

//             auto apply_transform = [&](const vec3& vertex) -> vec3 {
//                 glm::vec4 transformed_vertex = transform_matrix * glm::vec4(vertex, 1.0f);
//                 return vec3(transformed_vertex); // Convert back to 3D vector
//             };

//             Triangle triangle(
//                 apply_transform(vertices[f[0]-1]),
//                 apply_transform(vertices[f[1]-1]),
//                 apply_transform(vertices[f[2]-1]),
//                 material
//             );
//             world.add(triangle);
//         } else {
//             std::cout << "object parser error" << std::endl;
//         }
//     }
// }

void scene1(World &world, PerspectiveCamera &perspectiveCamera, int32_t height, int32_t width){
    // Camera
    vec3 center(13.0, 2.0, 3.0);
    vec3 direction(-13.0, -2.0, -3.0);
    direction = normalize(direction);
    vec3 up(0.0, 1.0, 0.0);
    //float fov = 80.00f / 360.0f * 2.0f * std::numbers::pi_v<float>;
    float fov = 0.607537f;
    int32_t samples = 50;
    int32_t max_depth = 25;
    float focal_distance = 10.0f;
    float defocus_angle = 0.6f / 180.0f * std::numbers::pi_v<float>;

    perspectiveCamera.setCamera(
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

    // World
    std::shared_ptr<Texture> checker = std::make_shared<CheckerTexture>(0.32, vec3(0.2, 0.3, 0.1), vec3(0.9, 0.9, 0.9));

    std::shared_ptr<Material> material_ground = std::make_shared<Lambertian>(checker);

    std::shared_ptr<Object> sphere_ground = std::make_shared<Sphere>(vec3(0.0, -1000.0, 0.0), 1000.0, material_ground);
    world.add(sphere_ground);

    std::shared_ptr<Material> material1 = std::make_shared<Dielectric>(1.5f);
    std::shared_ptr<Material> material2 = std::make_shared<Metal>(vec3(0.7, 0.6, 0.5), 0.0);
    std::shared_ptr<Material> material_light_yellow = std::make_shared<DiffuseLight>(vec3(0.7, 0.7, 0.0));
    std::shared_ptr<Material> material_light_purple = std::make_shared<DiffuseLight>(vec3(0.7, 0.0, 0.7));
    
    std::shared_ptr<Object> sphere1 = std::make_shared<Sphere>(vec3(-4.0, 1.0, 0.0), 1.0, material_light_yellow);
    std::shared_ptr<Object> sphere2 = std::make_shared<Sphere>(vec3(0.0, 1.0, 0.0), 1.0, material_light_purple);
    std::shared_ptr<Object> sphere3 = std::make_shared<Sphere>(vec3(4.0, 1.0, 0.0), 1.0, material_light_yellow);
    world.add(sphere1);
    world.add(sphere2);
    world.add(sphere3);

    for (float a = -11.0f; a < 11.0f; a = a + 1.0f) {
        for (float b = -11.0f; b < 11.0f; b = b + 1.0f) {
            float material_choice = random_float();

            vec3 sphere_center(a + 0.9f * random_float(), 0.2f, b + 0.9f * random_float());

            if (length(sphere_center - vec3(4, 0.2, 0.0)) > 0.9f) {
                std::shared_ptr<Material> material;

                if (material_choice < 0.8f) {
                    // diffuse
                    vec3 albedo = vec3(random_float() * random_float(), random_float() * random_float(), random_float() * random_float());
                    material = std::make_shared<Lambertian>(albedo);
                } else if (material_choice < 0.95f) {
                    // metal
                    vec3 albedo = vec3(0.5f + 0.5f * random_float(), 0.5f + 0.5f * random_float(), 0.5f + 0.5f * random_float());
                    float fuzz = random_float() * 0.5f;
                    material = std::make_shared<Metal>(albedo, fuzz);
                } else {
                    // dielectric
                    material = std::make_shared<Dielectric>(1.5f);
                }
                std::shared_ptr<Object> sphere = std::make_shared<Sphere>(sphere_center, 0.2, material);
                world.add(sphere);
            }
        }
    }

}

// void scene2(World &world, PerspectiveCamera &perspectiveCamera, int32_t height, int32_t width){
//     // Camera
//     vec3 center(0.0, 0.0, 10.0);
//     vec3 direction(0.0, 0.0, -10.0);
//     direction = glm::normalize(direction);
//     vec3 up(0.0, 1.0, 0.0);
//     float fov = 100.00f / 360.0f * 2.0f * std::numbers::pi_v<float>;
//     // float fov = 0.607537f;
//     int32_t samples = 1000;
//     int32_t max_depth = 50;
//     float focal_distance = 10.0f;
//     float defocus_angle = 0.6f / 180.0f * std::numbers::pi_v<float>;

//     perspectiveCamera.setCamera(
//         center,
//         direction,
//         up,
//         height,
//         width,
//         fov,
//         focal_distance,
//         defocus_angle,
//         samples,
//         max_depth
//     );

//     // World

//     std::shared_ptr<Material> material_left_wall = std::make_shared<Lambertian>(vec3(0.0, 0.8, 0.0)); // g
//     std::shared_ptr<Material> material_right_wall = std::make_shared<Lambertian>(vec3(0.8, 0.0, 0.0)); // r
//     std::shared_ptr<Material> material_wall = std::make_shared<Lambertian>(vec3(1.0, 1.0, 1.0));

//     std::shared_ptr<Material> material_light = std::make_shared<DiffuseLight>(vec3(7.0, 7.0, 7.0));

//     std::shared_ptr<Material> material_glass = std::make_shared<Dielectric>(1.5f);

//     add_object(world, "data/plate.obj", vec3(0.0, -5.0, 0.0), vec3(0.0, 0.0, 1.0), 0.0f, vec3(10.0, 10.0, 10.0), material_wall); // down
//     add_object(world, "data/plate.obj", vec3(0.0, 5.0, 0.0), vec3(0.0, 0.0, 1.0), 180.0f, vec3(10.0, 10.0, 10.0), material_wall); // up
//     add_object(world, "data/plate.obj", vec3(5.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), 90.0f, vec3(10.0, 10.0, 10.0), material_left_wall); // left
//     add_object(world, "data/plate.obj", vec3(-5.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), 270.0f, vec3(10.0, 10.0, 10.0), material_right_wall); // right
//     add_object(world, "data/plate.obj", vec3(0.0, 0.0, -5.0), vec3(1.0, 0.0, 0.0), 90.0f, vec3(10.0, 10.0, 10.0), material_wall); // back

//     add_object(world, "data/plate.obj", vec3(0.0, 5.0, 0.0), vec3(0.0, 0.0, 1.0), 180.0f, vec3(3.0, 3.0, 3.0), material_light); // up

//     add_object(world, "data/dragon.obj", vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), 90.0f, vec3(2.5, 2.5, 2.5), material_glass);

// }

void ppm(const std::vector<uint8_t> &image, int32_t height, int32_t width) {
    std::ofstream writeFile("output.ppm");
    if (writeFile.is_open()) {
        writeFile << "P3\n" << width << ' ' << height << "\n255\n";

        for(int32_t h = 0; h < height; ++h) {
            for(int32_t w = 0; w < width; ++ w) {
                writeFile << static_cast<int32_t>(image[h * width * 4 + w * 4 + 0])
                 << ' ' << static_cast<int32_t>(image[h * width * 4 + w * 4 + 1]) 
                 << ' ' << static_cast<int32_t>(image[h * width * 4 + w * 4 + 2]) << '\n';
            }
        }
        writeFile.close();
    }
}

int32_t main(int32_t argc, char *argv[]) {
    auto start = std::chrono::high_resolution_clock::now();

    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "outputs/output-" << std::put_time(std::localtime(&now_c), "%FT%T") << ".png";
    std::string filename = ss.str();

    // image resolution
    int32_t width = 2560;
    int32_t height = 1440;
    // int32_t width = 30;
    // int32_t height = 30;
    std::vector<uint8_t> image(height * width * 4); // rgba

    // materials
    World world;
    PerspectiveCamera perspectiveCamera;

    scene1(world, perspectiveCamera, height, width); // lots of balls

    // render
    perspectiveCamera.render_gpu(image, world);


    ppm(image, height, width);

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << std::endl << "Execution time: " << elapsed.count() << " seconds" << std::endl;
}


