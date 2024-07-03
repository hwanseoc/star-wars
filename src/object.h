#pragma once

#include <memory>

#include <glm/glm.hpp>

#include <random.h>
#include <ray.h>

class Material;

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
    glm::vec3 box_aa, box_bb;

    friend class BVHNode;

public:
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
};

class BVHLeaf : public Object {
    std::vector<std::shared_ptr<Object>> objects;
    AABB box_aabb;

    friend class BVHNode;

public:
    AABB aabb() const override {
        return box_aabb;
    }

    void clear() {
        objects.clear();
    }

    void add(std::shared_ptr<Object> object) {
        objects.push_back(object);
        box_aabb = AABB(box_aabb, object->aabb());
    }

    Hit hit(const Ray &r, float tmin, float tmax) const override {
        Hit ret;
        ret.is_hit = false;

        float tclose = tmax;

        for (const std::shared_ptr<Object> &object : objects) {
            Hit h = object->hit(r, tmin, tclose);
            if (h.is_hit) {
                tclose = h.t;
                ret = h;
            }
        }

        return ret;
    }
};

class BVHNode : public Object {
    std::shared_ptr<Object> left;
    std::shared_ptr<Object> right;
    AABB box_aabb;
public:
    BVHNode() {}

    BVHNode(BVHLeaf &leaf) : BVHNode(leaf.objects, 0, leaf.objects.size()) {}

    BVHNode(std::vector<std::shared_ptr<Object>> &objects, int32_t start, int32_t end) {

        for (int32_t i = start; i < end; ++i) {
            box_aabb = AABB(box_aabb, objects[i]->aabb());
        }

        float x_size = box_aabb.box_bb.x - box_aabb.box_aa.x;
        float y_size = box_aabb.box_bb.y - box_aabb.box_aa.y;
        float z_size = box_aabb.box_bb.z - box_aabb.box_aa.z;

        int32_t axis;
        if (x_size > y_size) {
            axis = x_size > z_size ? 0 : 2;
        } else {
            axis = y_size > z_size ? 1 : 2;
        }


        int32_t n_objects = end - start;

        if (n_objects == 1) {
            left = right = objects[start];
        } else if (n_objects == 2) {
            left = objects[start];
            right = objects[start + 1];
        } else {
            auto comparator = [&](const std::shared_ptr<Object> a, const std::shared_ptr<Object> b) {
                return a->aabb().box_aa[axis] < b->aabb().box_aa[axis];
            };
            std::sort(objects.begin() + start, objects.begin() + end, comparator);
            int32_t mid = start + n_objects / 2;
            left = std::make_shared<BVHNode>(objects, start, mid);
            right = std::make_shared<BVHNode>(objects, mid, end);
        }
    }

    Hit hit(const Ray &r, float tmin, float tmax) const override {
        Hit ret;
        ret.is_hit = false;

        if (!box_aabb.hit(r, tmin, tmax)) {
            return ret;
        }

        Hit hit_left = left->hit(r, tmin, tmax);

        if (hit_left.is_hit) {
            tmax = hit_left.t;
        }

        Hit hit_right = right->hit(r, tmin, tmax);

        if (hit_right.is_hit) {
            return hit_right;
        } else if (hit_left.is_hit) {
            return hit_left;
        }

        return ret;
    }

    AABB aabb() const override {
        return box_aabb;
    }
};
