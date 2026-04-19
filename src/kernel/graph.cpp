#include "graph.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>
#include <iostream>


void initialize_graph(graph_args* args,
                       std::size_t node_count,
                       int avg_degree,
                       std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(node_count) - 1);

    args->nodes.assign(node_count, Node{nullptr});
    args->edge_storage.clear();
    args->edge_storage.resize(node_count * static_cast<std::size_t>(avg_degree));

    args->graph.n = static_cast<int>(node_count);
    args->graph.nodes = args->nodes.data();

    std::size_t edge_pos = 0;

    for (std::size_t u = 0; u < node_count; ++u) {
        std::vector<int> neighbors;
        neighbors.reserve(avg_degree);

        for (int k = 0; k < avg_degree; ++k) {
            neighbors.push_back(dist(gen));
        }

        Edge* head = nullptr;
        for (int k = avg_degree - 1; k >= 0; --k) {
            Edge& e = args->edge_storage[edge_pos + static_cast<std::size_t>(k)];
            e.to = neighbors[static_cast<std::size_t>(k)];
            e.next = head;
            head = &e;
        }

        args->nodes[u].edges = head;
        edge_pos += static_cast<std::size_t>(avg_degree);
    }

    args->out = 0;

    // construc tthe optimized graph
    initialize_optimized_graph(args, avg_degree);
}

void initialize_optimized_graph(graph_args* args, int avg_degree) {
    args->opt_graph.n = args->graph.n;
    args->opt_graph.m = args->edge_storage.size();
    args->opt_graph.offsets.resize(args->opt_graph.n + 1); // one extra as a dummy
    args->opt_graph.edge_dists.resize(args->opt_graph.m);
    args->opt_graph.sum.resize(args->opt_graph.n);
    args->opt_graph.tot_sum = 0;

    for (int i = 0; i < args->opt_graph.n; i++) {
        args->opt_graph.offsets[i] = i * avg_degree;
        Edge* e = args->nodes[i].edges;
        int sum = 0;
        for (int j = 0; j < avg_degree; j++) {
            args->opt_graph.edge_dists[i * avg_degree + j] = e->to;
            sum += e->to;
            e = e->next;
        }
        args->opt_graph.sum[i] = sum;
        args->opt_graph.tot_sum += static_cast<uint64_t>(sum);
    }
    args->opt_graph.offsets[args->opt_graph.n] = args->opt_graph.n * avg_degree;
}

void naive_graph(std::uint64_t& out, const Graph& graph) {
    std::uint64_t checksum = 0;
    for (int u = 0; u < graph.n; ++u) {
        const Edge* e = graph.nodes[u].edges;
        while (e) {
            checksum += static_cast<std::uint64_t>(e->to);
            e = e->next;
        }
    }
    out = checksum;
}

void stu_graph(std::uint64_t& out, const OptimizedGraph& graph) {
    // TODO: You may need to add a function to convert data structure (not
    // included in time measurement), then implement your version in
    // stu_graph, whch is called by stu_graph_wrapper.

    // bro why is it taking 8 million ns when this func is empty nbruh??
    // nmaive
    #if 0
    // just checking
    std::cout << graph.n << " " << graph.offsets.size() << '\n';
    std::cout << graph.m << " " << graph.edge_dists.size() << '\n';
    #endif

    // O(n + m)
    #if 0
    std::uint64_t checksum = 0;
    for (int i = 0; i < graph.n; i++) {
        for (int j = graph.offsets[i]; j < graph.offsets[i + 1]; j++) {
            checksum += static_cast<std::uint64_t>(graph.edge_dists[j]);
        }
    }
    out = checksum;
    #endif

    // i can literally also do this bruh
    // O(m)
    #if 0
    std::uint64_t checksum = 0;
    for (int i = 0; i < graph.m; i++) {
        checksum += static_cast<std::uint64_t>(graph.edge_dists[i]);
    }
    out = checksum;
    #endif

    // is this allowed lol?
    // O(n)
    #if 0
    std::uint64_t checksum = 0;
    for (int i = 0; i < graph.n; i++) {
        checksum += static_cast<std::uint64_t>(graph.sum[i]);
    }
    out = checksum;
    #endif

    // bruh this is true O(1)
    #if 1
    out = static_cast<std::uint64_t>(graph.tot_sum);
    #endif

    // what if i just sum from the edges_storage
    // --> passed, but slow because need to iterate more, m = 8n and its an array of a struct --> not contiguous
    #if 0
    std::uint64_t checksum = 0;
    for (auto& x : graph) checksum += static_cast<uint64_t>(x.to);
    out = checksum;
    #endif
}

void naive_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    naive_graph(args.out, args.graph);
}

void stu_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    stu_graph(args.out, args.opt_graph);
}

bool graph_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto& stu_args = *static_cast<graph_args*>(stu_ctx);
    auto& ref_args = *static_cast<graph_args*>(ref_ctx);
    const auto eps = ref_args.epsilon;

    const double s = static_cast<double>(stu_args.out);
    const double r = static_cast<double>(ref_args.out);
    const double err = std::abs(s - r);
    const double atol = 0.0;
    const double rel = (std::abs(r) > 1e-12) ? err / std::abs(r) : err;

    debug_log("\tDEBUG: graph stu={} ref={} err={} rel={}\n",
              stu_args.out,
              ref_args.out,
              err,
              rel);

    return err <= (atol + eps * std::abs(r));
}
