#pragma once

#include <vector>
#include <typeinfo>

#include <object.h>
#include <sphere.h>
#include <triangle.h>


struct BVHNode {
    bool is_leaf;
    AABB aabb;
    std::shared_ptr<Object> obj;
    int64_t left;
    int64_t right;
};

struct cuda_BVHNode {
    bool is_leaf;
    AABB aabb;
    int64_t left;
    int64_t right;
    int8_t obj_type;
    cuda_Object* obj;
};

class cuda_BVH {

private:
    cuda_BVHNode *nodes;
    int32_t node_size;
    int64_t root = -1;

public:
    cuda_BVH() {}
    cuda_BVH(cuda_BVHNode *nodes, int32_t node_size, int64_t root) : nodes(nodes), node_size(node_size), root(root) {}

    __host__ ~cuda_BVH() {
        if(nodes) {
            cudaFree(nodes);
            nodes=nullptr;
        }
    }

    __device__ cuda_BVHNode *getCudaBVHNode() {
        return nodes;
    }

    __device__ int32_t getSize() {
        return node_size;
    }

    __device__ cuda_BVHHit hit_with_given_cuda_BVHNode(const Ray &r, float tmin, float tmax, cuda_BVHNode *shared_nodes) {
        return hit_recursive_with_given_BVHNode(r, tmin, tmax, root, shared_nodes);
    }

    __device__ cuda_BVHHit hit(const Ray &r, float tmin, float tmax) {
        return hit_recursive(r, tmin, tmax, root);
    }

private:
    __device__ cuda_BVHHit hit_recursive(const Ray &r, float tmin, float tmax, int64_t parent_node) {
        cuda_BVHNode *node = &nodes[parent_node];

        // if not node.hit
        if (!node->aabb.hit(r, tmin, tmax)) {
            cuda_BVHHit bvhhit;
            bvhhit.is_hit = false;
            bvhhit.t = 0.0f;
            return bvhhit;
        }

        if (node->is_leaf) {
            cuda_Object *object;

            cuda_BVHHit bvhhit;
            bvhhit.is_hit = false;
            bvhhit.t = 0.0f;
            switch (node->obj_type)
            {
            case OBJ_TYPE_CUDA_SPHERE:
                bvhhit = ((cuda_Sphere *)node->obj)->bvh_hit(r, tmin, tmax);
                break;
            case OBJ_TYPE_CUDA_TRIANGLE:
                bvhhit = ((cuda_Triangle *)node->obj)->bvh_hit(r, tmin, tmax);
                break;
            default:
                printf("wrong type\n");
                break;
            }           

            if (bvhhit.is_hit){
                bvhhit.obj = node->obj;
                bvhhit.obj_type = node->obj_type;
            }
            return bvhhit;
        } else {
            // TODO : make sure hit_right does not have dependency on hit_left,
            // so that we can parallelize hit_left and hit_right
            cuda_BVHHit bvhhit_left = hit_recursive(r, tmin, tmax, node->left);
            if (bvhhit_left.is_hit) {
                tmax = bvhhit_left.t;
            }
            cuda_BVHHit bvhhit_right = hit_recursive(r, tmin, tmax, node->right);
            if (bvhhit_right.is_hit) {
                return bvhhit_right;
            } else if (bvhhit_left.is_hit){
                return bvhhit_left;
            }

            cuda_BVHHit bvhhit;
            bvhhit.is_hit = false;
            bvhhit.t = 0.0f;
            return bvhhit;
        }
    }

    __device__ cuda_BVHHit hit_recursive_with_given_BVHNode(const Ray &r, float tmin, float tmax, int64_t parent_node, cuda_BVHNode *shared_nodes) {
        cuda_BVHNode *node = &shared_nodes[parent_node];

        // if not node.hit
        if (!node->aabb.hit(r, tmin, tmax)) {
            cuda_BVHHit bvhhit;
            bvhhit.is_hit = false;
            bvhhit.t = 0.0f;
            return bvhhit;
        }

        if (node->is_leaf) {
            cuda_Object *object;

            cuda_BVHHit bvhhit;
            bvhhit.is_hit = false;
            bvhhit.t = 0.0f;
            switch (node->obj_type)
            {
            case OBJ_TYPE_CUDA_SPHERE:
                bvhhit = ((cuda_Sphere *)node->obj)->bvh_hit(r, tmin, tmax);
                break;
            case OBJ_TYPE_CUDA_TRIANGLE:
                bvhhit = ((cuda_Triangle *)node->obj)->bvh_hit(r, tmin, tmax);
                break;
            default:
                printf("wrong type\n");
                break;
            }           

            if (bvhhit.is_hit){
                bvhhit.obj = node->obj;
                bvhhit.obj_type = node->obj_type;
            }
            return bvhhit;
        } else {
            // TODO : make sure hit_right does not have dependency on hit_left,
            // so that we can parallelize hit_left and hit_right
            cuda_BVHHit bvhhit_left = hit_recursive_with_given_BVHNode(r, tmin, tmax, node->left, shared_nodes);
            if (bvhhit_left.is_hit) {
                tmax = bvhhit_left.t;
            }
            cuda_BVHHit bvhhit_right = hit_recursive_with_given_BVHNode(r, tmin, tmax, node->right, shared_nodes);
            if (bvhhit_right.is_hit) {
                return bvhhit_right;
            } else if (bvhhit_left.is_hit){
                return bvhhit_left;
            }

            cuda_BVHHit bvhhit;
            bvhhit.is_hit = false;
            bvhhit.t = 0.0f;
            return bvhhit;
        }
    }
};



class BVH {
    std::vector<BVHNode> nodes;
    int64_t root = -1;
    cuda_BVH *host_cuda_bvh;
    cuda_BVHNode *host_nodes;

public:
    BVH() {}
    BVH(const World &w) {
        const std::vector<std::shared_ptr<Object>> &objects = w.get_objects();
        //printf("inside bvh constrcutor\n");
        for (int32_t i = 0; i < objects.size(); ++i) {
            std::shared_ptr<Object> obj = objects[i];
            //printf("[%d]: %p\n", i, obj.get());
            std::shared_ptr<Object> obj_copy = obj;
            struct BVHNode node = {
                .is_leaf = true,
                .aabb = obj->aabb(),
                .obj = obj
            };
            //printf("[%d]: %p\n", i, node.obj.get());
            nodes.push_back(node);
        }

        root = build_recursive(0, objects.size());

        //printf("before end\n");
        for(int32_t i = 0; i < nodes.size(); ++i) {
            //printf("[%d]:[%d] %p\n", i, (int)(nodes[i].is_leaf), nodes[i].obj.get());
        }
    }

    ~BVH() {
        for (int32_t i=0; i<nodes.size(); ++i) {
            if (host_nodes[i].obj) {
                cudaFree(host_nodes[i].obj);
            }
        }
        if (host_nodes) delete host_nodes;
        if (host_cuda_bvh) delete host_cuda_bvh;
    }

    int32_t getSize() {
        return nodes.size();
    }
private:
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
            .is_leaf = false,
            .aabb = aabb,
            .obj = nullptr,
            .left = left,
            .right = right
        };

        nodes.push_back(node);
        return nodes.size() - 1;
    }
public:
    cuda_BVH *convertToDevice() {
        cuda_BVHNode *dev_nodes;
        int32_t num_nodes = nodes.size();
        host_nodes = (cuda_BVHNode *)malloc(sizeof(cuda_BVHNode) * num_nodes);
        cudaMalloc(&dev_nodes, sizeof(cuda_BVHNode) * num_nodes);

        //copy vector to host_nodes
        for (int32_t i = 0; i < num_nodes; ++i) {
            if (nodes[i].is_leaf) {
                host_nodes[i] = {
                    .is_leaf = true,
                    .aabb = nodes[i].aabb,
                    .left = nodes[i].left,
                    .right = nodes[i].right,
                    .obj_type = nodes[i].obj->type(),
                    .obj = (cuda_Object *)(nodes[i].obj->convertToDevice())
                };
            } else {
                //printf("inside else\n");
                host_nodes[i] = {
                    .is_leaf = false,
                    .aabb = nodes[i].aabb,
                    .left = nodes[i].left,
                    .right = nodes[i].right,
                };
            }
        }
        //printf("after copy\n");

        cudaMemcpy(dev_nodes, host_nodes, sizeof(cuda_BVHNode) * num_nodes, cudaMemcpyHostToDevice);

        host_cuda_bvh = new cuda_BVH(dev_nodes, num_nodes, root);
        cuda_BVH *dev_cuda_bvh;

        cudaMalloc(&dev_cuda_bvh, sizeof(cuda_BVH));
        cudaMemcpy(dev_cuda_bvh, host_cuda_bvh, sizeof(cuda_BVH), cudaMemcpyHostToDevice);

        return dev_cuda_bvh;
    }
};