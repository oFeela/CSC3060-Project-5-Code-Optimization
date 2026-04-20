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

    // construc tthe optimized graph, assuming we HAVE no info about the avg_degree --> so it can generalize to ANY directed graph
    initialize_optimized_graph(args);
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
    // TODO: You may need to add a function to convert data structure (not
    // included in time measurement), then implement your version in
    // stu_graph, whch is called by stu_graph_wrapper.

    #if 0
    // just checking
    std::cout << graph.n << " " << graph.offsets.size() << '\n';
    std::cout << graph.m << " " << graph.edge_dests.size() << '\n';
    #endif

    // O(n + m)
    // traversal is a requirement.
    #if 0
    std::uint64_t checksum = 0;
    for (int i = 0; i < graph.n; i++) {
        for (int j = graph.offsets[i]; j < graph.offsets[i + 1]; j++) {
            checksum += static_cast<std::uint64_t>(graph.edge_dests[j]);
        }
    }
    out = checksum;
    #endif

    // explicit simd because the server doesnt auto vectorize wtf?
    // hey im forced to do this because the server cpu doesnt auto vectorize this loop bruh
    // syntax credits to deepseek yayy, TODO: understand the syntax!!!
    #if 1
    std::uint64_t checksum = 0;
    for (int i = 0; i < graph.n; i++) {
        int j = graph.offsets[i];
        int end = graph.offsets[i + 1];
        
        __m256i sum = _mm256_setzero_si256();
        
        for (; j + 7 < end; j += 8) {
            // 256 bits in total
            __m256i vals = _mm256_loadu_si256((__m256i*)&graph.edge_dests[j]);
            
            // split into 2 128
            // each hafl has 128 bits (4 x int32) --> convert it to epi64 256 bits (4 x int64)

            __m256i lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(vals));
            __m256i hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(vals, 1));

            // sum has 4 elements with each element consisting of partial sum of two vals
            // sum1|sum2|sum3|sum4, with sum1 = val1 + val5, sum2 = val2 + val6, ..., sum4 = val4 + val8
            // each sumi has 64 bits
            sum = _mm256_add_epi64(sum, lo);
            sum = _mm256_add_epi64(sum, hi);
        }
        
        // |sum1+sum3|sum2+sum4|
        __m128i sum128 = _mm_add_epi64(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));

        // sum1+sum3+sum2+sum4 = what is wanted
        uint64_t node_sum = _mm_extract_epi64(sum128, 0) + _mm_extract_epi64(sum128, 1);;
        
        checksum += node_sum;
        
        // leftovers
        for (; j < end; j++) {
            checksum += graph.edge_dests[j];
        }
    }
    out = checksum;
    #endif

    // 8 at once, DEEPSEEK's VERSION
    #if 0
    std::uint64_t checksum = 0;
    for (int i = 0; i < graph.n; i++) {
        int j = graph.offsets[i];
        int end = graph.offsets[i + 1];
        
        __m256i sum = _mm256_setzero_si256();
        
        for (; j + 7 < end; j += 8) {
            __m256i vals = _mm256_loadu_si256((__m256i*)&graph.edge_dests[j]);
            
            // Sign extend 8 int32 → 8 int64 (needs 2 registers)
            __m256i lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(vals));
            __m256i hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(vals, 1));
            
            sum = _mm256_add_epi64(sum, lo);
            sum = _mm256_add_epi64(sum, hi);
        }
        
        // Horizontal sum (AVX2)
        __m128i sum128 = _mm_add_epi64(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
        sum128 = _mm_add_epi64(sum128, _mm_srli_si128(sum128, 8));
        uint64_t node_sum = _mm_cvtsi128_si64(sum128);
        
        checksum += node_sum;
        
        // Remainder
        for (; j < end; j++) {
            checksum += graph.edge_dests[j];
        }
    }
    out = checksum;
    #endif

    // !!!! CANNOT RUN LOCALLY BECAUSE NO AVX512 SUPPORT. RUN ON SERVER IF WANT
    // ADD: -mavx512f -mavx512dq -mavx512vl if fails to compile
    // !!!! ITS SLOWER THAN ABOVE THO
    // 16 at once
    // trying to imeplement it myself
    // ITS CORRECT, I COMPARED THIS WITH DEEPSEEK'S IMPLEMENTATION BELOW
    #if 0
    std::uint64_t checksum = 0;
    for (int i = 0; i < graph.n; i++) {
        int j = graph.offsets[i];
        int end = graph.offsets[i + 1];

        // the partial sums here will be added later for the collective sum of all 16 elements
        __m512i sum = _mm512_setzero_si512(); // 512 bits of zero

        for (; j + 15 < end; j += 16) {
            __m512i vals = _mm512_loadu_si512((__m512i*)&graph.edge_dests[j]);
            // load 512 bits, want to represent 16 int32, but then i need it to be extended as int64
            // get each half: only 256 bits (8 int32) and then extend it back to 512 bits (8 int64)
            
            // cast from 512 to 256 by ignoring the upper 256
            __m512i low = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(vals));
            // get the upper 256 bits, syntwax: extract 32x8 = 256 bits with each block having 32 bits, so in total 8 blocks
            __m512i high = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(vals, 1));

            // sum has 8 elements with each element consisting of partial sum of two vals
            // sum1|sum2|sum3|sum4|...|sum8, with sum1 = val1 + val9, sum2 = val2 + val10, ..., sum8 = val8 + val16
            // each sumi has 64 bits
            sum = _mm512_add_epi64(sum, low);
            sum = _mm512_add_epi64(sum, high);
        }
        // over all edges, it will accumulate the sum as 8 partial sums --> need to add them together
        
        // now need to add the two halves (each is 256) as int256
        // 256 halves again as 128
        // 128 halves again as 64

        // |sum1+sum5|sum2+sum6|...|sum4+sum8|
        __m256i sum256 = _mm256_add_epi64(_mm512_castsi512_si256(sum), _mm512_extracti64x4_epi64(sum, 1));
        
        // |sum1+sum5+sum3+sum7|sum2+sum6+sum4+sum8|
        __m128i sum128 = _mm_add_epi64(_mm256_castsi256_si128(sum256), _mm256_extracti64x2_epi64(sum256, 1));

        // now need to add the two halves of sum128
        // theres no casting to 64 instruction man, bitshfit
        uint64_t sum64 = _mm_extract_epi64(sum128, 0) + _mm_extract_epi64(sum128, 1);

        checksum += sum64;

        // leftovers
        for (; j < end; j++) {
            checksum += graph.edge_dests[j];
        }
    }
    out = checksum;
    #endif

    // 16 at once deepseek's version
    #if 0
    std::uint64_t checksum = 0;
    for (int i = 0; i < graph.n; i++) {
        int j = graph.offsets[i];
        int end = graph.offsets[i + 1];
        
        // 8 accumulators (each 64-bit) for partial sums
        __m512i sum = _mm512_setzero_si512();
        
        // Process 16 ints per iteration
        for (; j + 15 < end; j += 16) {
            // Load 16 ints (512 bits)
            __m512i vals = _mm512_loadu_si512((__m512i*)&graph.edge_dests[j]);
            
            // Convert lower 8 ints (256 bits) to 8 int64
            __m512i low = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(vals));
            
            // Convert upper 8 ints (256 bits) to 8 int64
            __m512i high = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(vals, 1));
            
            // Accumulate
            sum = _mm512_add_epi64(sum, low);
            sum = _mm512_add_epi64(sum, high);
        }
        
        // Horizontal sum: reduce 8 lanes to 1
        // Step 1: 8 -> 4 (add upper and lower 256 bits)
        __m256i sum256 = _mm256_add_epi64(
            _mm512_castsi512_si256(sum),
            _mm512_extracti64x4_epi64(sum, 1)
        );
        
        // Step 2: 4 -> 2 (add upper and lower 128 bits)
        __m128i sum128 = _mm_add_epi64(
            _mm256_castsi256_si128(sum256),
            _mm256_extracti128_si256(sum256, 1)
        );
        
        // Step 3: 2 -> 1 (add the two remaining)
        uint64_t node_sum = _mm_extract_epi64(sum128, 0) + _mm_extract_epi64(sum128, 1);
        
        checksum += node_sum;
        
        // Handle remaining elements (less than 16)
        for (; j < end; j++) {
            checksum += graph.edge_dests[j];
        }
    }
    out = checksum;
    #endif

    // i can literally also do this bruh
    // O(m)
    #if 0
    std::uint64_t checksum = 0;
    for (int i = 0; i < graph.m; i++) {
        checksum += static_cast<std::uint64_t>(graph.edge_dests[i]);
    }
    out = checksum;
    #endif

    // below NOT USED!!!

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
    #if 0
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
