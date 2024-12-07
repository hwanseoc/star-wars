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
__global__  void render_kernel(curandState *states, int32_t x_dim, int32_t z_dim, cuda_BVH *bvh, PerspectiveCamera *camera, int32_t height, int32_t width, int32_t samples, vec3* image);
__global__  void render_kernel_sub_image(
    curandState *state,
    int32_t x_dim,
    int32_t z_dim,
    cuda_BVH *bvh,
    PerspectiveCamera *camera,
    int32_t sub_height,
    int32_t sub_width,
    int32_t samples,
    int32_t h_start_from,
    int32_t w_start_from,
    vec3* sub_image
);

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


    void render_gpu_with_sub_image(std::vector<uint8_t> &image, const World& world) {
        int32_t sub_image_width = 256;
        int32_t sub_image_height = 256;

        int32_t num_sub_image_width = (width - 1) / sub_image_width + 1;
        int32_t num_sub_image_height = (height - 1) / sub_image_height + 1;


        dim3 threadsPerBlock(8, 8, 10);
        dim3 numBlocks((sub_image_width - 1) / threadsPerBlock.x + 1, (sub_image_height - 1) / threadsPerBlock.y + 1, (samples - 1) / threadsPerBlock.z + 1);

        size_t newStackSize = 16 * 1024; //KB
        cudaDeviceSetLimit(cudaLimitStackSize, newStackSize);

        printf("render_gpu_sub_image\tstarted\n");
        curandState *states;
        cudaMalloc(&states, sizeof(curandState) * (threadsPerBlock.x * numBlocks.x) * (threadsPerBlock.y * numBlocks.y) * (threadsPerBlock.z * numBlocks.z));
        
        initCurandStates<<<numBlocks, threadsPerBlock>>>(states, 1234, (threadsPerBlock.x * numBlocks.x), (threadsPerBlock.z * numBlocks.z));
        printf("render_gpu_sub_image\tcurand init\n");

        BVH bvh(world);
        cuda_BVH *host_cuda_bvh = bvh.convertToDevice();
        cuda_BVH *dev_cuda_bvh;

        PerspectiveCamera *host_camera = this;
        PerspectiveCamera *dev_camera;

        cudaMalloc(&dev_camera, sizeof(PerspectiveCamera));
        cudaMemcpy(dev_camera, host_camera, sizeof(PerspectiveCamera), cudaMemcpyHostToDevice);

        cudaMalloc(&dev_cuda_bvh, sizeof(cuda_BVH));
        cudaMemcpy(dev_cuda_bvh, host_cuda_bvh, sizeof(cuda_BVH), cudaMemcpyHostToDevice);

        vec3 *host_sub_image = (vec3 *)malloc(sizeof(vec3) * sub_image_height * sub_image_width * samples);
        vec3 *dev_sub_image;
        cudaMalloc(&dev_sub_image, sizeof(vec3) * sub_image_height * sub_image_width * samples);


        printf("render_gpu_sub_image\tkernel started\n");
        std::clog << "\rsub block processed: 0 out of " << (num_sub_image_height * num_sub_image_width) << std::flush;
        for (int32_t sub_h = 0; sub_h < num_sub_image_height; ++sub_h) {
            for (int32_t sub_w = 0; sub_w < num_sub_image_width; ++sub_w) {
                render_kernel_sub_image<<<numBlocks, threadsPerBlock>>>(
                    states,
                    (threadsPerBlock.x * numBlocks.x),
                    (threadsPerBlock.z * numBlocks.z),
                    dev_cuda_bvh,
                    dev_camera,
                    sub_image_height,
                    sub_image_width,
                    samples,
                    (sub_h * sub_image_height),
                    (sub_w * sub_image_width),
                    dev_sub_image
                );

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
                cudaMemcpy(host_sub_image, dev_sub_image, sizeof(vec3) * sub_image_height * sub_image_width * samples, cudaMemcpyDeviceToHost);

                for (int32_t h = 0; h < sub_image_height; ++h) {
                    for (int32_t w = 0; w < sub_image_width; ++w) {
                        vec3 pixel(0.0f, 0.0, 0.0f);
                        for (int32_t s = 0; s < samples; ++s) {
                            pixel += host_sub_image[(h * sub_image_width + w) * samples + s];
                        }
                        pixel /= samples;
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
                        
                        int32_t global_h = sub_h * sub_image_height + h;
                        int32_t global_w = sub_w * sub_image_width + w;

                        if (global_h < height && global_w < width) {
                            image[global_h * width * 4 + global_w * 4 + 0] = ir;
                            image[global_h * width * 4 + global_w * 4 + 1] = ig;
                            image[global_h * width * 4 + global_w * 4 + 2] = ib;
                            image[global_h * width * 4 + global_w * 4 + 3] = 255;
                        }
                    }
                }
                //printf("%d out of %d sub blocks done\n", (sub_h * num_sub_image_width + sub_w) + 1, (num_sub_image_height * num_sub_image_width));
                std::clog << "\rsub block processed: " << (sub_h * num_sub_image_width + sub_w) + 1 << " out of " << (num_sub_image_height * num_sub_image_width) << std::flush;
            }
        }
        std::cout << std::endl;

        printf("render_gpu_sub_image\tkernel ended\n");
        
        printf("render_gpu_sub_image\tfreeing memory\n");
        cudaFree(host_cuda_bvh);
        cudaFree(dev_cuda_bvh);

        cudaFree(dev_sub_image);
        free(host_sub_image);
    }

    void render_gpu(std::vector<uint8_t> &image, const World& world) {
        dim3 threadsPerBlock(2, 2, 16);
        dim3 numBlocks((width - 1) / threadsPerBlock.x + 1, (height - 1) / threadsPerBlock.y + 1, (samples - 1) / threadsPerBlock.z + 1);

        size_t newStackSize = 16 * 1024; //KB
        cudaDeviceSetLimit(cudaLimitStackSize, newStackSize);

        printf("render_gpu\tstarted\n");
        curandState *states;
        cudaMalloc(&states, sizeof(curandState) * (threadsPerBlock.x * numBlocks.x) * (threadsPerBlock.y * numBlocks.y) * (threadsPerBlock.z * numBlocks.z));
        
        initCurandStates<<<numBlocks, threadsPerBlock>>>(states, 1234, (threadsPerBlock.x * numBlocks.x), (threadsPerBlock.z * numBlocks.z));
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

        vec3 *host_image = (vec3 *)malloc(sizeof(vec3) * height * width * samples);
        vec3 *dev_image;
        cudaMalloc(&dev_image, sizeof(vec3) * height * width * samples);


        printf("render_gpu\tkernel started\n");
        render_kernel<<<numBlocks, threadsPerBlock>>>(states, (threadsPerBlock.x * numBlocks.x), (threadsPerBlock.z * numBlocks.z), dev_cuda_bvh, dev_camera, height, width, samples, dev_image);

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


        cudaMemcpy(host_image, dev_image, sizeof(vec3) * height * width * samples, cudaMemcpyDeviceToHost);

        printf("render_gpu\tprocessing image\n");
        for (int32_t h = 0; h < height; ++h) {
            for (int32_t w = 0; w < width; ++w) {
                vec3 pixel(0.0f, 0.0, 0.0f);
                for (int32_t s = 0; s < samples; ++s) {
                    pixel += host_image[(h * width + w) * samples + s];
                }
                pixel /= samples;
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
        printf("render_gpu\tfreeing memory\n");
        cudaFree(host_cuda_bvh);
        cudaFree(dev_cuda_bvh);

        cudaFree(dev_image);
        free(host_image);

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

    __device__ vec3 get_color(curandState *state, cuda_BVH *bvh, const Ray &r, int32_t depth) {
        if (depth <= 0) {
            return vec3(0.0, 0.0, 0.0);
        }
        cuda_BVHHit bvh_hit = bvh->hit(r, 0.001f, std::numeric_limits<float>::max());
        if (!bvh_hit.is_hit) {
            return vec3(0.0, 0.0, 0.0);
        }

        cuda_ColorHit hit;
        switch(bvh_hit.obj_type)
        {
        case OBJ_TYPE_CUDA_SPHERE:
            hit = ((cuda_Sphere *)(bvh_hit.obj))->hit(state, bvh_hit, r, 0.001f, std::numeric_limits<float>::max());
            break;
        case OBJ_TYPE_CUDA_TRIANGLE:
            hit = ((cuda_Triangle *)(bvh_hit.obj))->hit(state, bvh_hit, r, 0.001f, std::numeric_limits<float>::max());
            break;
        default:
            break;
        }
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

    __device__ vec3 get_color_with_given_BVHNode(curandState *state, cuda_BVH *bvh, const Ray &r, int32_t depth, cuda_BVHNode *shared_nodes) {
        if (depth <= 0) {
            return vec3(0.0, 0.0, 0.0);
        }
        cuda_BVHHit bvh_hit = bvh->hit_with_given_cuda_BVHNode(r, 0.001f, std::numeric_limits<float>::max(), shared_nodes);
        if (!bvh_hit.is_hit) {
            return vec3(0.0, 0.0, 0.0);
        }

        cuda_ColorHit hit;
        switch(bvh_hit.obj_type)
        {
        case OBJ_TYPE_CUDA_SPHERE:
            hit = ((cuda_Sphere *)(bvh_hit.obj))->hit(state, bvh_hit, r, 0.001f, std::numeric_limits<float>::max());
            break;
        case OBJ_TYPE_CUDA_TRIANGLE:
            hit = ((cuda_Triangle *)(bvh_hit.obj))->hit(state, bvh_hit, r, 0.001f, std::numeric_limits<float>::max());
            break;
        default:
            break;
        }
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
        return color_emitted + attenuation * get_color_with_given_BVHNode(state, bvh, ray_scatter, depth - 1, shared_nodes);
    }
};


__global__  void render_kernel(curandState *state, int32_t x_dim, int32_t z_dim, cuda_BVH *bvh, PerspectiveCamera *camera, int32_t height, int32_t width, int32_t samples, vec3* image) {
    int32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t h = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t s = blockIdx.z * blockDim.z + threadIdx.z;

    __shared__ cuda_BVH shared_cuda_bvh;
    __shared__ cuda_BVHNode *shared_nodes;

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        shared_cuda_bvh = *bvh;
        shared_nodes = shared_cuda_bvh.getCudaBVHNode();
    }

    __syncthreads();

    Ray r = camera->get_ray(&state[(h * x_dim + w) * z_dim + s], h, w);
    vec3 pixel = camera->get_color_with_given_BVHNode(&state[(h * x_dim + w) * z_dim + s], &shared_cuda_bvh, r, 50, shared_nodes);

    if ( w < width && h < height && s < samples) {
        image[(h * width + w) * samples + s] = pixel;
    }

    __syncthreads();
}


__global__  void render_kernel_sub_image(
    curandState *state,
    int32_t x_dim,
    int32_t z_dim,
    cuda_BVH *bvh,
    PerspectiveCamera *camera,
    int32_t sub_height,
    int32_t sub_width,
    int32_t samples,
    int32_t h_start_from,
    int32_t w_start_from,
    vec3* sub_image
    ) {
    int32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t h = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t s = blockIdx.z * blockDim.z + threadIdx.z;

    __shared__ cuda_BVH shared_cuda_bvh;
    __shared__ cuda_BVHNode *shared_nodes;

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        shared_cuda_bvh = *bvh;
        shared_nodes = shared_cuda_bvh.getCudaBVHNode();
    }

    __syncthreads();

    Ray r = camera->get_ray(&state[(h * x_dim + w) * z_dim + s], h + h_start_from, w + w_start_from);
    vec3 pixel = camera->get_color_with_given_BVHNode(&state[(h * x_dim + w) * z_dim + s], &shared_cuda_bvh, r, 50, shared_nodes);

    if ( w < sub_width && h < sub_height && s < samples) {
        sub_image[(h * sub_width + w) * samples + s] = pixel;
    }

    __syncthreads();
}
