/*
This is all helpers that I use for the bonus part
*/

#ifndef HELPERS_H
#define HELPERS_H

#include <thread>
#include <vector>
#include <functional>

/*
Create threads for loops
*/
template<typename Func>
void parallel_for(int start, int end, Func&& func) {
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int chunk_sz = (end - start) / num_threads;

    for (unsigned int t = 0; t < num_threads; t++) {
      int chunk_st = start + t * chunk_sz;
      int chunk_en = (t != num_threads - 1) ? chunk_st + chunk_sz : end;

      threads.emplace_back([&func, chunk_st, chunk_en]() {
        func(chunk_st, chunk_en);
      });
    }

    for (std::thread& th : threads) {
      th.join();
    }
}

#endif