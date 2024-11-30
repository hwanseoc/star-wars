#pragma once

#include <cmath>
#include <thread>
#include <algorithm>

#include <vec.h>
#include <random.h>
#include <ray.h>

#include <object.h>
#include <bvh.h>
#include <material.h>

class PerspectiveCamera;
__global__  void render_kernel(curandState *states, cuda_BVH *bvh, PerspectiveCamera *camera, int32_t height, int32_t width, vec3* image);

class PerspectiveCamera {
    int32_t height, width, samples, max_depth;
    float focal_distance, defocus_angle;
    vec3 center, pixel00, du, dv, disk_u, disk_v;

public:
    PerspectiveCamera() {}
    void setCamera(
        const vec3 &center,
        const vec3& direction,
        const vec3& up,
        int32_t height,
        int32_t width,
        float fov,
        float focal_distance,
        float defocus_angle,
        int32_t samples,
        int32_t max_depth
    ) {
        this->height = height;
        this->width = width;
        this->samples = samples;
        this->max_depth = max_depth;
        this->focal_distance = focal_distance;
        this->defocus_angle = defocus_angle;
        this->center = center;
        float widthf = static_cast<float>(width);
        float heightf = static_cast<float>(height);

        float magnitude = 2.0f * focal_distance * std::tan(fov / 2.0f) / widthf;
        du = normalize(cross(direction, up)) * magnitude;
        dv = normalize(cross(direction, du)) * magnitude;

        float disk_radius = focal_distance * std::tan(defocus_angle / 2.0f);
        disk_u = normalize(cross(direction, up)) * disk_radius;
        disk_v = normalize(cross(direction, du)) * disk_radius;

        pixel00 = center + focal_distance * normalize(direction)
                    - du * (widthf / 2.0f)
                    - dv * (heightf / 2.0f);
    }

    void render_gpu(std::vector<uint8_t> &image, const World& world) {
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width - 1) / threadsPerBlock.x + 1, (height - 1) / threadsPerBlock.y + 1);

        printf("render_gpu\tstarted\n");
        curandState *states;
        cudaMalloc(&states, sizeof(curandState) * height * width);
        
        initCurandStates<<<numBlocks, threadsPerBlock>>>(states, 1234, height, width);
        printf("render_gpu\tcurand init\n");

        BVH bvh(world);
        cuda_BVH *host_cuda_bvh = bvh.convertToDevice();
        cuda_BVH *dev_cuda_bvh;

        PerspectiveCamera *host_camera = this;
        PerspectiveCamera *dev_camera;

        cudaMalloc(&dev_camera, sizeof(PerspectiveCamera));
        cudaMemcpy(dev_camera, host_camera, sizeof(PerspectiveCamera), cudaMemcpyHostToDevice);

        cudaMalloc(&dev_cuda_bvh, sizeof(cuda_BVH));
        cudaMemcpy(dev_cuda_bvh, host_cuda_bvh, sizeof(cuda_BVH), cudaMemcpyHostToDevice);

        vec3 *host_image = (vec3 *)malloc(sizeof(vec3 ) * height * width);
        vec3 *dev_image;
        cudaMalloc(&dev_image, sizeof(vec3) * height * width);
        printf("render_gpu\tkernel started\n");
        render_kernel<<<numBlocks, threadsPerBlock>>>(states, dev_cuda_bvh, dev_camera, height, width, dev_image);

        printf("render_gpu\tkernel ended\n");

        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
    	if (err != cudaSuccess) {
		    printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
		    return;
    	}

    	// Ensure kernel has completed before copying data back to host
    	err = cudaDeviceSynchronize();
    	if (err != cudaSuccess) {
		    printf("CUDA synchronize failed: %s\n", cudaGetErrorString(err));
		    return;
    	}


        cudaMemcpy(host_image, dev_image, sizeof(vec3) * height * width, cudaMemcpyDeviceToHost);


        for (int32_t h = 0; h < height; ++h) {
            for (int32_t w = 0; w < width; ++ w) {
                vec3 pixel = host_image[h*width + w];
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
        cudaFree(dev_image);
        free(host_image);
        // for (int32_t h = 0; h < height; ++h) {
        //     for (int32_t w = 0; w < width; ++w) {
        //         std::clog << "\rPixels processed: " << h * width + w + 1 << " out of " << height * width << std::flush;

        //         vec3 pixel(0.0, 0.0, 0.0);

        //         for (int32_t s = 0; s < samples; ++s) {
        //             Ray r = this->get_ray(h, w);
        //             vec3 sampled = get_color(bvh, world, r, 50);
        //             pixel += sampled;
        //         }

        //         pixel /= samples;

        //         // linear to gamma
        //         pixel.x = pixel.x > 0.0f ? std::sqrt(pixel.x) : 0.0f;
        //         pixel.y = pixel.y > 0.0f ? std::sqrt(pixel.y) : 0.0f;
        //         pixel.z = pixel.z > 0.0f ? std::sqrt(pixel.z) : 0.0f;

        //         // clamp
        //         pixel.x = std::clamp(pixel.x, 0.0f, 1.0f);
        //         pixel.y = std::clamp(pixel.y, 0.0f, 1.0f);
        //         pixel.z = std::clamp(pixel.z, 0.0f, 1.0f);

        //         uint8_t ir = static_cast<uint8_t>(255.999f * pixel.x);
        //         uint8_t ig = static_cast<uint8_t>(255.999f * pixel.y);
        //         uint8_t ib = static_cast<uint8_t>(255.999f * pixel.z);

        //         image[h * width * 4 + w * 4 + 0] = ir;
        //         image[h * width * 4 + w * 4 + 1] = ig;
        //         image[h * width * 4 + w * 4 + 2] = ib;
        //         image[h * width * 4 + w * 4 + 3] = 255;
        //     }
        // }

        std::cout << std::endl;
    }




    __device__ Ray get_ray(curandState *state, int32_t h, int32_t w) {
        float random_h = static_cast<float>(h) + cuda_random_float(state);
        float random_w = static_cast<float>(w) + cuda_random_float(state);
        vec3 origin;

        if (defocus_angle <= 0) {
            origin = center;
        } else {
            vec3 p = cuda_random_disk(state);
            origin = center + p.x * disk_u + p.y * disk_v;
        }
        vec3 direction = normalize(
            pixel00 + dv * random_h + du * random_w - origin
        );
        return Ray(origin, direction);
    }

    __device__ vec3 get_color(curandState *state, cuda_BVH *bvh, const Ray &r, int32_t depth) const {
        if (depth <= 0) {
            return vec3(0.0, 0.0, 0.0);
        }
        cuda_BVHHit bvh_hit = bvh->hit(r, 0.001f, std::numeric_limits<float>::max());
        if (!bvh_hit.is_hit) {
            return vec3(0.0, 0.0, 0.0);
        }
        cuda_Object* obj = bvh_hit.obj;
        cuda_ColorHit hit = ((cuda_Sphere *)obj)->hit(state, bvh_hit, r, 0.001f, std::numeric_limits<float>::max());

        // bool is_scatter, vec3 attenuation, Ray ray_scatter
        bool is_scatter = false;
        vec3 attenuation = vec3();
        Ray ray_scatter = Ray();
        vec3 color_emitted = vec3();

        switch (hit.mat_type)
        {
        case MAT_TYPE_CUDA_DIELECTRIC:
            ((cuda_Dielectric *)hit.mat)->scatter(state, r, hit, is_scatter, attenuation, ray_scatter);
            break;
        case MAT_TYPE_CUDA_DIFFUSELIGHT:
            color_emitted = ((cuda_DiffuseLight *)hit.mat)->emitted(hit);
            break;
        case MAT_TYPE_CUDA_LAMBERTIAN:
            ((cuda_Lambertian *)hit.mat)->scatter(state, r, hit, is_scatter, attenuation, ray_scatter);
            break;
        case MAT_TYPE_CUDA_METAL:
            ((cuda_Metal *)hit.mat)->scatter(state, r, hit, is_scatter, attenuation, ray_scatter);
            break;
        
        default:
            printf("found wrong mat type\n");
            // hit.mat->scatter(r, hit, is_scatter, attenuation, ray_scatter);
            // color_emitted = hit.mat->emitted(hit);
            break;
        }
        

        if (!is_scatter) {
            return color_emitted;
        }

        return color_emitted + attenuation * get_color(state, bvh, ray_scatter, depth - 1);
    }
};


__global__  void render_kernel(curandState *state, cuda_BVH *bvh, PerspectiveCamera *camera, int32_t height, int32_t width, vec3* image) {
    int32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t h = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("%d %d\n",w,h);
    if ( h < height && w < width) {
        image[h*width + w] = vec3(static_cast<float>(w) / static_cast<float>(width), static_cast<float>(h) / static_cast<float>(height), 0.0f);
    }
    if ( h < height && w < width) {
        Ray r = camera->get_ray(&state[h * width + w],h, w);
        vec3 sampled = camera->get_color(&state[h * width + w], bvh, r, 50);
    }

    __syncthreads();
}

