#pragma once

#include <vector>
#include <object.h>

struct BVHHit {
    int64_t i; // if no object is hit, i == -1
    float t;
};

class BVH {
    struct BVHNode {
        int64_t i; // if not a leaf node, i == -1
        AABB aabb;
        int64_t left;
        int64_t right;
    };

    std::vector<BVHNode> nodes;
    int64_t root = -1;

public:
    BVH(const World &w) {
        const std::vector<std::shared_ptr<Object>> &objects = w.get_objects();

        for (size_t i = 0; i < objects.size(); ++i) {
            BVHNode node = {
                .i = i,
                .aabb = objects[i]->aabb(),
                .left = 0,
                .right = 0
            };
            nodes.push_back(node);
        }
        
        root = build_recursive(0, objects.size());
    }

    int64_t build_recursive(int64_t start, int64_t end) {
        int64_t n_objects = end - start;

        if (n_objects == 1) {
            return start;
        }

        // sort
        AABB aabb;
        for (int64_t i = start; i < end; ++i) {
            aabb = AABB(aabb, nodes[i].aabb);
        }

        float x_size = aabb.box_bb.x - aabb.box_aa.x;
        float y_size = aabb.box_bb.y - aabb.box_aa.y;
        float z_size = aabb.box_bb.z - aabb.box_aa.z;

        // pick the longest axis
        int64_t axis;
        if (x_size > y_size) {
            axis = x_size > z_size ? 0 : 2;
        } else {
            axis = y_size > z_size ? 1 : 2;
        }

        auto comparator = [&](const BVHNode &a, const BVHNode &b) {
            return a.aabb.box_aa[axis] < b.aabb.box_aa[axis];
        };
        std::sort(nodes.begin() + start, nodes.begin() + end, comparator);

        int64_t mid = start + n_objects / 2;
        int64_t left = build_recursive(start, mid);
        int64_t right = build_recursive(mid, end);

        BVHNode node = {
            .i = -1,
            .aabb = aabb,
            .left = left,
            .right = right
        };

        nodes.push_back(node);
        return nodes.size() - 1;
    }

    BVHHit hit(const World &w, const Ray &r, float tmin, float tmax) const {
        return hit_recursive(w, r, tmin, tmax, root);
    }

    BVHHit hit_recursive(const World &w, const Ray &r, float tmin, float tmax, int64_t parent_node) const {
        const std::vector<std::shared_ptr<Object>> &objects = w.get_objects();
        const BVHNode &node = nodes[parent_node];

        // node.hit
        if (!node.aabb.hit(r, tmin, tmax)){
            return BVHHit{-1, 0.0f};
        }

        // object.hit
        bool is_leaf = node.i != -1;
        if (is_leaf) {
            const std::shared_ptr<Object> &object = objects[node.i];
            return 
        }


        //left.hit
        // right.hit

    }
};

// class BVHNode {
//     std::shared_ptr<Object> left;
//     std::shared_ptr<Object> right;
//     AABB box_aabb;
// public:

//     Hit hit(const Ray &r, float tmin, float tmax) const {
//         Hit ret;
//         ret.is_hit = false;

//         if (!box_aabb.hit(r, tmin, tmax)) {
//             return ret;
//         }

//         Hit hit_left = left->hit(r, tmin, tmax);

//         if (hit_left.is_hit) {
//             tmax = hit_left.t;
//         }

//         Hit hit_right = right->hit(r, tmin, tmax);

//         if (hit_right.is_hit) {
//             return hit_right;
//         } else if (hit_left.is_hit) {
//             return hit_left;
//         }

//         return ret;
//     }

//     AABB aabb() const {
//         return box_aabb;
//     }
// };

