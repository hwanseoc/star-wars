#pragma once

#include <memory>

// #include <glm/glm.hpp>
#include <vec.h>
#include <random.h>
#include <ray.h>

class Material;
class Object;

struct BVHHit {
    bool is_hit;
    float t;
    const Object* obj;
};

struct ColorHit {
    vec3 point;
    vec3 normal;
    vec3 direction;
    float u; // texture x coord
    float v; // texture y coord
    Material *mat;
    bool is_front;
};

class AABB {
public:
    vec3 box_aa, box_bb;

    AABB() : box_aa(0.0f, 0.0f, 0.0f), box_bb(0.0f, 0.0f, 0.0f) {}

    AABB(const vec3 &box_aa_, const vec3 &box_bb_) {
        constexpr float delta = 0.0001f;
        for (int32_t i = 0; i < 3; ++i){
            float dif = box_bb_[i] - box_aa_[i];
            if (dif < delta) {
                box_aa[i] = box_aa_[i] - delta;
                box_bb[i] = box_bb_[i] + delta;
            } else {
                box_aa[i] = box_aa_[i];
                box_bb[i] = box_bb_[i];
            }
        }
    }

    AABB(const AABB &box0, const AABB &box1) {
        for (int32_t i = 0; i < 3; ++i){
            box_aa[i] = std::min(box0.box_aa[i], box1.box_aa[i]);
            box_bb[i] = std::max(box0.box_bb[i], box1.box_bb[i]);
        }
    }

    bool hit(const Ray &r, float tmin, float tmax) const {
        const vec3 &origin = r.origin;
        const vec3 &direction = r.direction;

        for (int32_t i = 0; i < 3; ++i) {
            float inverse_direction = 1.0f / direction[i];
            
            float t0 = (box_aa[i] - origin[i]) * inverse_direction;
            float t1 = (box_bb[i] - origin[i]) * inverse_direction;

            if (inverse_direction < 0.0f) {
                std::swap(t0, t1);
            }

            tmin = std::max(t0, tmin);
            tmax = std::min(t1, tmax);

            if (tmax <= tmin) {
                return false;
            }
        }

        return true;
    }

};


class Object {
public:
    Material *mat;
    __host__ __device__ virtual ~Object() = default;

    __host__ __device__ virtual AABB aabb() const = 0;

    __device__ virtual ColorHit hit(const BVHHit &bvhhit, const Ray &r, float tmin, float tmax) const = 0;

    __device__ virtual BVHHit bvh_hit(const Ray &r, float tmin, float tmax) const = 0;
};

class World {
    std::vector<Object*> objects;
    std::vector<Material*> materials;
    AABB box_aabb;

public:
    World() {}


    void destroy() {
        for (Object* &obj_ptr : objects) {
            delete obj_ptr;
        }
        for (Material* &mat_ptr : materials) {
            delete mat_ptr;
        }
    }

    AABB aabb() const {
        return box_aabb;
    }

    template <typename OBJ_T>
    void add(OBJ_T &obj) {
        OBJ_T *temp = new OBJ_T(obj);

        objects.push_back(temp);

        box_aabb = AABB(box_aabb, temp->aabb());
    }

    template <typename MAT_T>
    void add_mat(MAT_T mat) {
        materials.push_back(mat);
    }

    const std::vector<Object*>& get_objects() const {
        return objects;
    }
};

class cuda_World {
    Object** dev_objects;
    AABB *dev_box_aabb;
    int32_t dev_num_objects;
public:
    __host__ __device__ cuda_World() {}
    __host__ __device__ cuda_World(World w) {
        std::vector<Object*> objects_vector = w.get_objects();
        AABB host_box_aabb = w.aabb();
        int32_t num_objects = objects_vector.size();
        Object** host_objects = (Object **)malloc(sizeof(Object *) * num_objects);

        for(int32_t i=0; i<num_objects; ++i) {
            host_objects[i] = objects_vector[i];
        }

        cudaMalloc(&dev_objects, sizeof(Object *) * num_objects);
        cudaMemcpy(dev_objects, host_objects, sizeof(Object *) * num_objects, cudaMemcpyHostToDevice);

        cudaMalloc(&dev_box_aabb, sizeof(AABB));
        cudaMemcpy(dev_box_aabb, &host_box_aabb, sizeof(AABB), cudaMemcpyHostToDevice);

        dev_num_objects = num_objects;

        free(host_objects);
    }

    __host__ __device__ void destory() {
        cudaFree(dev_objects);
        cudaFree(dev_box_aabb);
    }

    __device__ AABB aabb() const {
        return *dev_box_aabb;
    }

    __device__ int32_t size() const {
        return dev_num_objects;
    }

    __device__ Object **get_objects() const {
        return dev_objects;
    }
};