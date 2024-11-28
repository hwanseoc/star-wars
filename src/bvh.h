#pragma once

#include <vector>

#include <object.h>

struct BVHNode {
    bool is_leaf;
    AABB aabb;
    int64_t left;
    int64_t right;
    const Object* obj;
};

class BVH {

    std::vector<BVHNode> nodes;
    int64_t root = -1;

public:
    __host__ __device__ BVH() {}
    __host__ BVH(const World &w) {
        const std::vector<Object*> &objects = w.get_objects();

        for (const Object* obj : objects) {
            BVHNode node = {
                .is_leaf = true,
                .aabb = obj->aabb(),
                .obj = obj
            };
            nodes.push_back(node);
        }

        root = build_recursive(0, objects.size());
    }

    int64_t get_root() {
        return root;
    }

    std::vector<BVHNode> &get_nodes(){
        return nodes;
    }

private:
    __host__ int64_t build_recursive(int64_t start, int64_t end) {
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
            .is_leaf = false,
            .aabb = aabb,
            .left = left,
            .right = right
        };

        nodes.push_back(node);
        return nodes.size() - 1;
    }

// public:
//     BVHHit hit(const World &w, const Ray &r, float tmin, float tmax) const {
//         return hit_recursive(w, r, tmin, tmax, root);
//     }

// private:
//     BVHHit hit_recursive(const World &w, const Ray &r, float tmin, float tmax, int64_t parent_node) const {
//         const BVHNode &node = nodes[parent_node];

//         BVHHit bvhhit;
//         bvhhit.is_hit = false;
//         bvhhit.t = 0.0f;

//         // if not node.hit
//         if (!node.aabb.hit(r, tmin, tmax)) {
//             return bvhhit;
//         }

//         // object.hit
//         if (node.is_leaf) {
//             const Object* object = node.obj;

//             bvhhit = object->bvh_hit(r, tmin, tmax);

//             if (bvhhit.is_hit){
//                 bvhhit.obj = object;
//             }

//             return bvhhit;
//         } else {
//             // TODO : make sure hit_right does not have dependency on hit_left,
//             // so that we can parallelize hit_left and hit_right
//             BVHHit bvhhit_left = hit_recursive(w, r, tmin, tmax, node.left);

//             if (bvhhit_left.is_hit) {
//                 tmax = bvhhit_left.t;
//             }

//             BVHHit bvhhit_right = hit_recursive(w, r, tmin, tmax, node.right);

//             if (bvhhit_right.is_hit) {
//                 return bvhhit_right;
//             } else if (bvhhit_left.is_hit){
//                 return bvhhit_left;
//             }

//             return bvhhit;
//         }
//     }
};


class cuda_BVH {
    BVHNode *dev_nodes;
    int32_t dev_num_nodes;
    int64_t dev_root = -1;

public:
    __host__ __device__ cuda_BVH() {}
    __host__ cuda_BVH(BVH bvh) {
        std::vector<BVHNode> nodes_vector = bvh.get_nodes();
        int32_t num_nodes = nodes_vector.size();

        BVHNode *host_nodes = (BVHNode *)malloc(sizeof(BVHNode) * num_nodes);
        for(int32_t i=0; i<num_nodes; ++i) {
            host_nodes[i] = nodes_vector[i];
        }

        cudaMalloc(&dev_nodes, sizeof(BVHNode) * num_nodes);
        cudaMemcpy(dev_nodes, host_nodes, sizeof(BVHNode) * num_nodes, cudaMemcpyHostToDevice);

        dev_root = bvh.get_root();
        dev_num_nodes = num_nodes;

        free(host_nodes);
    }
    __host__ ~cuda_BVH(){
        cudaFree(dev_nodes);
    }

    __device__ BVHHit hit(const cuda_World &w, const Ray &r, float tmin, float tmax) const {
        return hit_recursive(w, r, tmin, tmax, dev_root);
    }

private:
    __device__ BVHHit hit_recursive(const cuda_World &w, const Ray &r, float tmin, float tmax, int64_t parent_node) const {
        BVHNode node = dev_nodes[parent_node];

        BVHHit bvhhit;
        bvhhit.is_hit = false;
        bvhhit.t = 0.0f;

        // if not node.hit
        if (!node.aabb.hit(r, tmin, tmax)) {
            return bvhhit;
        }

        // object.hit
        if (node.is_leaf) {
            const Object* object = node.obj;

            bvhhit = object->bvh_hit(r, tmin, tmax);

            if (bvhhit.is_hit){
                bvhhit.obj = object;
            }

            return bvhhit;
        } else {
            // TODO : make sure hit_right does not have dependency on hit_left,
            // so that we can parallelize hit_left and hit_right
            BVHHit bvhhit_left = hit_recursive(w, r, tmin, tmax, node.left);

            if (bvhhit_left.is_hit) {
                tmax = bvhhit_left.t;
            }

            BVHHit bvhhit_right = hit_recursive(w, r, tmin, tmax, node.right);

            if (bvhhit_right.is_hit) {
                return bvhhit_right;
            } else if (bvhhit_left.is_hit){
                return bvhhit_left;
            }

            return bvhhit;
        }
    }    
};