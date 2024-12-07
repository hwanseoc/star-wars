#pragma once

#include <memory>

#include <vec.h>
#include <random.h>
#include <ray.h>

#include <object.h>

#define OBJ_TYPE_DEFAULT 0
#define OBJ_TYPE_CUDA_SPHERE 1
#define OBJ_TYPE_CUDA_TRIANGLE 2

class Material;
class cuda_Material;
class Object;
class cuda_Object;

struct BVHHit {
    bool is_hit;
    float t;
    const Object* obj;
};

struct cuda_BVHHit {
    bool is_hit;
    float t;
    cuda_Object* obj;
    int8_t obj_type;
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

struct cuda_ColorHit {
    vec3 point;
    vec3 normal;
    vec3 direction;
    float u; // texture x coord
    float v; // texture y coord
    cuda_Material *mat;
    int8_t mat_type;
    bool is_front;
};

class AABB {
public:
    vec3 box_aa, box_bb;

    __host__ __device__ AABB() : box_aa(0.0f, 0.0f, 0.0f), box_bb(0.0f, 0.0f, 0.0f) {}

    __host__ __device__ AABB(const vec3 &box_aa_, const vec3 &box_bb_) {
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

    __host__ __device__ AABB(const AABB &box0, const AABB &box1) {
        for (int32_t i = 0; i < 3; ++i){
            box_aa[i] = std::min(box0.box_aa[i], box1.box_aa[i]);
            box_bb[i] = std::max(box0.box_bb[i], box1.box_bb[i]);
        }
    }

    __host__ __device__ bool hit(const Ray &r, float tmin, float tmax) const {
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

class cuda_Object {};


class Object {
public:
    virtual ~Object() = default;

    virtual AABB aabb() const = 0;

    virtual cuda_Object *convertToDevice() = 0;

    virtual int8_t type() const {
        return OBJ_TYPE_DEFAULT;
    }
};

class World {
    std::vector<std::shared_ptr<Object>> objects;
    AABB box_aabb;

public:
    World() {}
    ~World() {}

    AABB aabb() const {
        return box_aabb;
    }

    template <typename OBJ_T>
    void add(OBJ_T obj) {
        objects.push_back(obj);

        box_aabb = AABB(box_aabb, obj->aabb());
    }

    const std::vector<std::shared_ptr<Object>>& get_objects() const {
        return objects;
    }
};
