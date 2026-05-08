#ifndef GRAPH_H
#define GRAPH_H

#include "bench.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

const std::chrono::nanoseconds BASELINE_GRAPH{5000000};
inline constexpr double NAIVE_SPEEDUP_LOWER_BOUND_GRAPH{2.50};

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

// Optimized graph
struct OptimizedGraph {
    int n;
    // which edge index the i-th node starts at, and the end is before the edge index which the (i+1)-th node starts at
    std::vector<int> offsets; 
    std::vector<int> edge_dests; // size: m, which node the edge points to
};

struct graph_args {
    Graph graph;
    std::vector<Node> nodes;
    std::vector<Edge> edge_storage;
    std::uint64_t out;
    double epsilon;
    // TODO: You may want to add new params at the end...
    OptimizedGraph graph_csr;

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

void convert_graph_to_csr(OptimizedGraph& graph_csr, const Graph& graph_stu);

bool graph_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func);

#endif
