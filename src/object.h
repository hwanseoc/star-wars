#pragma once

#include <memory>

#include <glm/glm.hpp>

#include <ray.h>

class Hit {
public:
    bool is_hit;
    glm::vec3 point;
    glm::vec3 normal;
    glm::vec3 direction;
    glm::vec3 color;
    float t;
    bool is_front;

    void set_face_normal(const Ray &r, const glm::vec3 &outward_normal) {
        is_front = glm::dot(r.direction(), outward_normal) < 0.0;
        normal = is_front ? normal : -normal;
    }
};

class Object {
public:
    virtual Hit hit(const Ray &r, float tmin, float tmax) const = 0;
};

class ObjectList {
    std::vector<std::shared_ptr<Object>> objects;

public:
    void clear() {
        objects.clear();
    }

    void add(std::shared_ptr<Object> object) {
        objects.push_back(object);
    }

    Hit hit(const Ray &r, float tmin, float tmax) const {
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
