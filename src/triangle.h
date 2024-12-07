#pragma once

#include <cmath>
#include <numbers>

#include <vec.h>
#include <object.h>


/*

        v3   
        | \
v (up)  |  \
        |   \
        v1---v2
            u (right)

front face is toward you
counter-clockwise, 1->2->3
*/


// P = v1 + (v2 - v1) * u + (v3 - v1) * v

class cuda_Triangle : public cuda_Object {
    vec3 v1, v2, v3;
    vec3 normal;
    cuda_Material *mat;
    int8_t mat_type;

public:
    __host__ cuda_Triangle(
        const vec3 &v1,
        const vec3 &v2,
        const vec3 &v3,
        const vec3 &normal,
        cuda_Material *mat,
        int8_t mat_type
    ) : v1(v1), v2(v2), v3(v3), normal(normal), mat(mat), mat_type(mat_type) {}

    __device__ cuda_ColorHit hit(curandState *state, const cuda_BVHHit &bvhhit, const Ray &r, float tmin, float tmax) const {
        cuda_ColorHit ret;
        ret.point = r.at(bvhhit.t);
        ret.direction = cuda_random_hemisphere(state, ret.normal);
        ret.is_front = dot(r.direction, normal) < 0.0f;
        ret.normal = ret.is_front ? normal : -normal;
        ret.mat = mat;
        ret.mat_type = mat_type;

        vec3 edge1 = v2 - v1;
        vec3 edge2 = v3 - v1;
        vec3 ray_cross_edge2 = cross(r.direction, edge2);
        float det = dot(edge1, ray_cross_edge2);
        float inv_det = 1.0f / det;
        vec3 s = r.origin - v1;
        vec3 q = cross(s, edge1);
        float u = inv_det * dot(s, ray_cross_edge2);
        float v = inv_det * dot(r.direction, q);

        ret.u = u;
        ret.v = v;

        return ret;
    }

    __device__ cuda_BVHHit bvh_hit(const Ray &r, float tmin, float tmax) const {
        cuda_BVHHit ret;
        ret.is_hit = false;

        vec3 edge1 = v2 - v1;
        vec3 edge2 = v3 - v1;
        vec3 ray_cross_edge2 = cross(r.direction, edge2);
        float det = dot(edge1, ray_cross_edge2);

        // the ray is parallel to the triangle.
        if (std::abs(det) <= std::numeric_limits<float>::epsilon()) {
            return ret;
        }

        float inv_det = 1.0f / det;
        vec3 s = r.origin - v1;
        float u = inv_det * dot(s, ray_cross_edge2);

        // the intersection is outside of the triangle.
        if (u < 0.0f || u > 1.0f) {
            return ret;
        }

        vec3 q = cross(s, edge1);
        float v = inv_det * dot(r.direction, q);

        // the intersection is outside of the triangle.
        if (v < 0.0f || u + v > 1.0f) {
            return ret;
        }

        float t = inv_det * dot(edge2, q);

        // the intersection is outside the valid t range.
        if (t < tmin || t > tmax) {
            return ret;
        }

        ret.is_hit = true;
        ret.t = t;

        return ret;
    }

    __host__ __device__ AABB aabb() const {
        vec3 min(
            std::min({v1.x, v2.x, v3.x}),
            std::min({v1.y, v2.y, v3.y}),
            std::min({v1.z, v2.z, v3.z})
        );

        vec3 max(
            std::max({v1.x, v2.x, v3.x}),
            std::max({v1.y, v2.y, v3.y}),
            std::max({v1.z, v2.z, v3.z})
        );

        return AABB(min, max);
    }
};



class Triangle : public Object {
    vec3 v1, v2, v3;
    vec3 normal;
    std::shared_ptr<Material> mat;

    cuda_Triangle *host_cuda_obj;

public:
    Triangle(const vec3 &v1, const vec3 &v2, const vec3 &v3, std::shared_ptr<Material> mat) : v1(v1), v2(v2), v3(v3), mat(mat) {
        vec3 u_edge = v2 - v1;
        vec3 v_edge = v3 - v1;
        normal = normalize(cross(u_edge, v_edge));
    }

    ~Triangle() {
        if (host_cuda_obj) delete host_cuda_obj;
    }

    AABB aabb() const override {
        vec3 min(
            std::min({v1.x, v2.x, v3.x}),
            std::min({v1.y, v2.y, v3.y}),
            std::min({v1.z, v2.z, v3.z})
        );

        vec3 max(
            std::max({v1.x, v2.x, v3.x}),
            std::max({v1.y, v2.y, v3.y}),
            std::max({v1.z, v2.z, v3.z})
        );

        return AABB(min, max);
    }

    __host__ cuda_Triangle *convertToDevice() override {
        host_cuda_obj = new cuda_Triangle(v1, v2, v3, normal, mat->convertToDevice(), mat->type());

        cuda_Triangle *dev_cuda_obj;

        cudaMalloc(&dev_cuda_obj, sizeof(cuda_Triangle));
        cudaMemcpy(dev_cuda_obj, host_cuda_obj, sizeof(cuda_Triangle), cudaMemcpyHostToDevice);

        return dev_cuda_obj;
    }

    __host__ int8_t type() const override {
        return OBJ_TYPE_CUDA_TRIANGLE;
    }
};