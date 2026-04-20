#ifndef GRAPH_H
#define GRAPH_H

#include "bench.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

const std::chrono::nanoseconds BASELINE_GRAPH{5000000};

struct Edge {
    int to;
    Edge* next;
};

struct Node {
    Edge* edges;
};

struct Graph {
    int n;
    Node* nodes;
};

// struct OptimizedEdge {
//     int w;
//     int next;

//     OptimizedEdge(int w, int next) : w(w), next(next) {}
// };

// Optimized graph
struct OptimizedGraph {
    int n;
    int m;
    // which edge index the i-th node starts at, and the end is before the edge index which the (i+1)-th node starts at
    std::vector<int> offsets; 
    std::vector<int> edge_dests; // size: m, which node the edge points to

    // what if i just store the sum of edge_dists for each node immediately?
    // NOT USED!!! it was just for experiment
    std::vector<int> sum;
    uint64_t tot_sum;

    // my first approach:
    // adjacency list
    // std::vector<std::vector<OptimizedEdge>> adj;
    // realized: might be too slow because not contiguous for the node traversal...
};

struct graph_args {
    Graph graph;
    std::vector<Node> nodes;
    std::vector<Edge> edge_storage;
    std::uint64_t out;
    double epsilon;
    // TODO: You may want to add new params at the end...
    OptimizedGraph opt_graph;

    explicit graph_args(double epsilon_in = 1e-6)
        : graph{0, nullptr}, out{0}, epsilon{epsilon_in} {}
};

void naive_graph(std::uint64_t& out, const Graph& graph);
// TODO: You may need to add a function to convert data structure (not 
// included in time measurement), then implement your version in 
// stu_graph, whch is called by stu_graph_wrapper.
void stu_graph(std::uint64_t& out, const OptimizedGraph& graph);

void naive_graph_wrapper(void* ctx);
void stu_graph_wrapper(void* ctx);

void initialize_graph(graph_args* args,
                       std::size_t node_count,
                       int avg_degree,
                       std::uint_fast64_t seed);

void initialize_optimized_graph(graph_args* args);

bool graph_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func);

#endif
