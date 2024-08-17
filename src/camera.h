#pragma once

#include <cmath>
#include <thread>
#include <algorithm>

#include <glm/glm.hpp>

#include <random.h>

#include <ray.h>
#include <object.h>
#include <bvh.h>
#include <material.h>

class PerspectiveCamera {
    int32_t height, width, samples, max_depth;
    float focal_distance, defocus_angle;
    glm::vec3 center, pixel00, du, dv, disk_u, disk_v;
public:
    PerspectiveCamera(
        glm::vec3 &center,
        glm::vec3& direction,
        glm::vec3& up,
        int32_t height,
        int32_t width,
        float fov,
        float focal_distance,
        float defocus_angle,
        int32_t samples,
        int32_t max_depth
    ) : height(height), width(width), samples(samples), max_depth(max_depth), focal_distance(focal_distance), defocus_angle(defocus_angle), center(center) {
        float widthf = static_cast<float>(width);
        float heightf = static_cast<float>(height);

        float magnitude = 2.0f * focal_distance * std::tan(fov / 2.0f) / widthf;
        du = glm::normalize(glm::cross(direction, up)) * magnitude;
        dv = glm::normalize(glm::cross(direction, du)) * magnitude;

        float disk_radius = focal_distance * std::tan(defocus_angle / 2.0f);
        disk_u = glm::normalize(glm::cross(direction, up)) * disk_radius;
        disk_v = glm::normalize(glm::cross(direction, du)) * disk_radius;

        pixel00 = center + focal_distance * glm::normalize(direction)
                    - du * (widthf / 2.0f)
                    - dv * (heightf / 2.0f);
    }

    void render(std::vector<uint8_t> &image, const BVH& bvh, const World& world) {
        for (int32_t h = 0; h < height; ++h) {
            for (int32_t w = 0; w < width; ++w) {
                std::clog << "\rPixels processed: " << h * width + w + 1 << " out of " << height * width << std::flush;

                glm::vec3 pixel(0.0, 0.0, 0.0);

                if (true) {
                // if (h == 50 && w == 200) {
                //if (h >= 500 && h < 501 && w >= 200 && w < 210) {
                    for (int32_t s = 0; s < samples; ++s) {
                        Ray r = this->get_ray(h, w);
                        glm::vec3 sampled = get_color(bvh, world, r, 50);
                        pixel += sampled;
                    }

                    pixel /= samples;
                } else {
                    pixel.x = 0.0f;
                    pixel.y = 1.0f;
                    pixel.z = 0.0f;
                }

                // linear to gamma
                pixel.x = pixel.x > 0.0f ? std::sqrt(pixel.x) : 0.0f;
                pixel.y = pixel.y > 0.0f ? std::sqrt(pixel.y) : 0.0f;
                pixel.z = pixel.z > 0.0f ? std::sqrt(pixel.z) : 0.0f;

                // clamp
                pixel.x = std::clamp(pixel.x, 0.0f, 1.0f);
                pixel.y = std::clamp(pixel.y, 0.0f, 1.0f);
                pixel.z = std::clamp(pixel.z, 0.0f, 1.0f);

                uint8_t ir = static_cast<uint8_t>(255.999f * pixel.x);
                uint8_t ig = static_cast<uint8_t>(255.999f * pixel.y);
                uint8_t ib = static_cast<uint8_t>(255.999f * pixel.z);

                image[h * width * 4 + w * 4 + 0] = ir;
                image[h * width * 4 + w * 4 + 1] = ig;
                image[h * width * 4 + w * 4 + 2] = ib;
                image[h * width * 4 + w * 4 + 3] = 255;
            }
        }

        std::cout << std::endl;
    }
    void multi_thread_render_sub(const BVH& bvh, const World& world, const int32_t num_process, const int32_t worker_id, std::vector<glm::vec3> &ret) {
        // std::cout << "[Sub]\t\tworker_id:" << worker_id << std::endl;
        for(int32_t i = worker_id; i < height * width; i += num_process) {
            // std::cout << "[Sub_for]\t\ti:" << i << " i/num:" << i/num_process <<std::endl;

            int32_t h = i / width;
            int32_t w = i % width;

            if(worker_id == 0){
                std::clog << "\r[WorkerId]:" << worker_id << " Pixels processed: " << i/num_process + 1 << " out of " << (height * width + num_process - 1) / num_process << std::flush;
            }

            glm::vec3 pixel(0.0, 0.0, 0.0);

            for (int32_t s = 0; s < samples; ++s) {
                Ray r = this->get_ray(h, w);
                glm::vec3 sampled = get_color(bvh, world, r, 50);
                pixel += sampled;
            }

            pixel /= samples;

            // linear to gamma
            pixel.x = pixel.x > 0.0f ? std::sqrt(pixel.x) : 0.0f;
            pixel.y = pixel.y > 0.0f ? std::sqrt(pixel.y) : 0.0f;
            pixel.z = pixel.z > 0.0f ? std::sqrt(pixel.z) : 0.0f;

            // clamp
            pixel.x = std::clamp(pixel.x, 0.0f, 1.0f);
            pixel.y = std::clamp(pixel.y, 0.0f, 1.0f);
            pixel.z = std::clamp(pixel.z, 0.0f, 1.0f);

            ret[i/num_process] = pixel;
        }
    }


    //TODO:
    // delete inter variable ret and replace with image
    void multi_thread_render(std::vector<uint8_t> &image, const BVH& bvh, const World& world, const int32_t num_process) {
        std::vector<std::vector<glm::vec3>> ret;
        std::vector<std::thread> process;

        int32_t ret_size = (height * width + num_process - 1) / num_process;

        ret.resize(num_process);
        for(int32_t p = 0; p < num_process; ++p) {
            ret[p].resize(ret_size);
        }
        process.resize(num_process);

        // std::cout << "started multi-thread-render" << std::endl;
        // std::cout << "num_process " << num_process << std::endl;
        // std::cout << "height " << height << std::endl;
        // std::cout << "width " << width << std::endl;
        // std::cout << "ret_size " << ret_size << std::endl;

        for(int32_t p = 0; p < num_process; ++p) {
            // std::cout << "[Thread]\tworker_id:"<< p2 << std::endl;
            process[p] = std::thread(
                &PerspectiveCamera::multi_thread_render_sub,
                this,
                bvh,
                world,
                num_process,
                p,
                std::ref(ret[p])
            );
            // std::cout << "[Thread end]\tworker_id:"<< p2 << std::endl;
        }

        for(int32_t p = 0; p < num_process; ++p) {
            // std::cout << "[Join]\tworker_id:"<< p3 << std::endl;
            process[p].join();
        }

        // std::cout << "ended join" << std::endl;

        for(int32_t h = 0; h < height; ++h) {
            for(int32_t w = 0; w < width; ++w) {
                int32_t worker_id = (h * width + w) % num_process;

                glm::vec3 pixel = ret[worker_id][(h * width + w)/num_process];

                uint8_t ir = static_cast<uint8_t>(255.999f * pixel.x);
                uint8_t ig = static_cast<uint8_t>(255.999f * pixel.y);
                uint8_t ib = static_cast<uint8_t>(255.999f * pixel.z);

                image[h * width * 4 + w * 4 + 0] = ir;
                image[h * width * 4 + w * 4 + 1] = ig;
                image[h * width * 4 + w * 4 + 2] = ib;
                image[h * width * 4 + w * 4 + 3] = 255;
            }
        }

        std::cout << std::endl << "ended multi-thread-render" << std::endl;
    }




    Ray get_ray(int32_t h, int32_t w) {
        float random_h = static_cast<float>(h) + random_float();
        float random_w = static_cast<float>(w) + random_float();
        glm::vec3 origin;

        if (defocus_angle <= 0) {
            origin = center;
        } else {
            glm::vec2 p = random_disk();
            origin = center + p.x * disk_u + p.y * disk_v;
        }

        glm::vec3 direction = glm::normalize(
            pixel00 + dv * random_h + du * random_w - origin
        );
        return Ray(origin, direction);
    }

    glm::vec3 get_color(const BVH &bvh, const World &world, const Ray &r, int32_t depth) const {
        if (depth <= 0) {
            return glm::vec3(0.0, 0.0, 0.0);
        }

        BVHHit bvh_hit = bvh.hit(world, r, 0.001f, std::numeric_limits<float>::max());

        if (bvh_hit.is_hit) {
            Object *obj = bvh_hit.obj;
            ColorHit hit = obj->hit(bvh_hit, r, 0.001f, std::numeric_limits<float>::max());

            // bool is_scatter, glm::vec3 attenuation, Ray ray_scatter
            const auto& [is_scatter, attenuation, ray_scatter] = hit.mat->scatter(r, hit);

            if (is_scatter) {
                return attenuation * get_color(bvh, world, ray_scatter, depth - 1);
            }
            return glm::vec3(0.0, 0.0, 0.0);
        }

        // background
        float alpha = 0.5f * (r.direction().y + 1.0f);
        return (1.0f - alpha) * glm::vec3(1.0, 1.0, 1.0) + alpha * glm::vec3(0.5, 0.7, 1.0);
    }
};


