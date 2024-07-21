#pragma once

#include <memory>

#include <glm/glm.hpp>

#include <random.h>
#include <ray.h>

class Material;

struct BVHHit {
    int64_t i; // -1 means no hit
    float t;

    void set_no_hit() {
        i = -1;
    }
    bool is_hit() const {
        return i != -1;
    }
};

class Hit {
public:
    bool is_hit;
    glm::vec3 point;
    glm::vec3 normal;
    glm::vec3 direction;
    std::shared_ptr<Material> mat;
    float t;
    bool is_front;

    void set_face_normal(const Ray &r, const glm::vec3 &outward_normal) {
        is_front = glm::dot(r.direction(), outward_normal) < 0.0f;
        normal = is_front ? outward_normal : -outward_normal;
    }
};

class AABB {
public:
    glm::vec3 box_aa, box_bb;

    AABB() : box_aa(0.0f, 0.0f, 0.0f), box_bb(0.0f, 0.0f, 0.0f) {}

    AABB(const glm::vec3 &box_aa, const glm::vec3 &box_bb) : box_aa(box_aa), box_bb(box_bb) {}

    AABB(const AABB &box0, const AABB &box1) {
        for (int32_t i = 0; i < 3; ++i){
            box_aa[i] = std::min(box0.box_aa[i], box1.box_aa[i]);
            box_bb[i] = std::max(box0.box_bb[i], box1.box_bb[i]);
        }
    }

    bool hit(const Ray &r, float tmin, float tmax) const {
        const glm::vec3 &origin = r.origin();
        const glm::vec3 &direction = r.direction();

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
    virtual Hit hit(const Ray &r, float tmin, float tmax) const = 0;

    virtual AABB aabb() const = 0;

    virtual BVHHit bvh_hit(const Ray &r, float tmin, float tmax) const = 0;
};

class World {
    std::vector<std::shared_ptr<Object>> objects;
    AABB box_aabb;

public:
    AABB aabb() const {
        return box_aabb;
    }

    void clear() {
        objects.clear();
    }

    void add(std::shared_ptr<Object> object) {
        objects.push_back(object);
        box_aabb = AABB(box_aabb, object->aabb());
    }

    const std::vector<std::shared_ptr<Object>>& get_objects() const {
        return objects;
    }
    
    const std::shared_ptr<Object>& get_object(int64_t i) const {
        return objects[i];
    }
};
