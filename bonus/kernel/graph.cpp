#include "graph.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>
#include <iostream>

#include <immintrin.h>

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
}

void initialize_optimized_graph(graph_args* args) {
    args->opt_graph.n = args->graph.n;
    args->opt_graph.m = args->edge_storage.size();
    args->opt_graph.offsets.resize(args->opt_graph.n + 1); // one extra as a dummy
    args->opt_graph.edge_dests.resize(args->opt_graph.m);

    // THESE ARE NOT USED!!!
    args->opt_graph.sum.resize(args->opt_graph.n);
    args->opt_graph.tot_sum = 0;

    int edge_idx = 0;
    for (int i = 0; i < args->opt_graph.n; i++) {
        args->opt_graph.offsets[i] = edge_idx;
        Edge* e = args->nodes[i].edges;
        int sum = 0;
        while (e != nullptr) {
            args->opt_graph.edge_dests[edge_idx] = e->to;
            sum += e->to;
            e = e->next;
            edge_idx++;
        }
        args->opt_graph.sum[i] = sum;
        args->opt_graph.tot_sum += static_cast<uint64_t>(sum);
    }
    args->opt_graph.offsets[args->opt_graph.n] = edge_idx;
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
    // just from normal implementation
    #if 0
    std::uint64_t checksum = 0;
    const std::vector<int>& offsets = graph.offsets;
    const std::vector<int>& edge_dests = graph.edge_dests;
    for (int u = 0; u < graph.n; u++) {
        int j = offsets[u];
        int r = offsets[u + 1];

        for (; j + 7 < r; j += 8) {
            checksum += edge_dests[j];
            checksum += edge_dests[j + 1];
            checksum += edge_dests[j + 2];
            checksum += edge_dests[j + 3];
            checksum += edge_dests[j + 4];
            checksum += edge_dests[j + 5];
            checksum += edge_dests[j + 6];
            checksum += edge_dests[j + 7];
        }
        for (; j < r; j++) {
            checksum += edge_dests[j];
        }
    }
    out = checksum;
    #endif

    // explicit simd because the server doesnt auto vectorize for some whatever reason
    #if 1
    std::uint64_t checksum = 0;
    const std::vector<int>& offsets = graph.offsets;
    const std::vector<int>& edge_dests = graph.edge_dests;
    for (int u = 0; u < graph.n; u++) {
        int j = offsets[u];
        int r = offsets[u + 1];

        __m256i psum = _mm256_setzero_si256(); // 256/32 = 8 int or 4 longlong

        for (; j + 7 < r; j += 8) {
            // 8 int
            __m256i vals = _mm256_loadu_si256((__m256i*)&graph.edge_dests[j]);

            // split into 2 128 bits
            // each half has 4 int --> convert it to 4 long long
            __m256i lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(vals));
            __m256i hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(vals, 1));

            // psum = lo + hi --> psum now contains the partial sums
            // |psum1|psum2|psum3|psum4|, with psum1 = val1 + val5, psum2 = val2 + val6, ..., psum4 = val4 + val8
            psum = _mm256_add_epi64(psum, lo);
            psum = _mm256_add_epi64(psum, hi);
        }

        // convert sum (partial sums) into the actual sum
        // |sum1 + sum3|sum2 + sum4|
        __m128i psum128 = _mm_add_epi64(_mm256_castsi256_si128(psum), _mm256_extracti128_si256(psum, 1));
        
        // |sum1 + sum2 + sum3 + sum4| is the wanted
        uint64_t sum = _mm_extract_epi64(psum128, 0) + _mm_extract_epi64(psum128, 1);
        checksum += sum;

        // leftovers
        for (; j < r; j++) {
            checksum += edge_dests[j];
        }
    }
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
