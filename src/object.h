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
    virtual ~Object() = default;

    virtual AABB aabb() const = 0;

    virtual ColorHit hit(const BVHHit &bvhhit, const Ray &r, float tmin, float tmax) const = 0;

    virtual BVHHit bvh_hit(const Ray &r, float tmin, float tmax) const = 0;
};

class World {
    std::vector<const Object*> objects;
    std::vector<const Material*> materials;
    AABB box_aabb;

public:
    World() {}


    void destroy() {
        for (const Object* &obj_ptr : objects) {
            delete obj_ptr;
        }
        for (const Material* &mat_ptr : materials) {
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

    const std::vector<const Object*>& get_objects() const {
        return objects;
    }
};

class cuda_World {
    Object** objects;
    AABB box_aabb;
    int32_t num_objects;
public:
    cuda_World(World w) {
        
    }
};