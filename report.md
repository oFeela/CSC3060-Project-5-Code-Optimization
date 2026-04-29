
## Optimized Implementation

### sparse_spmm <!--! check requirement in PDF
`perf` showed that accessing `bt_row[csr.col_idx[p]]` had 70% miss rate, so the problem lies in this line. Accesses to sparse matrix don't usually cause cache misses because the CSR format made the data sequential and contiguous.
- iterate through dense_cols of B first then rows of A is the main idea, because we have already paid the random access memory load price for `bt_row` once and want to maximize using it before we throw it away.
- when we load `bt_row` we access the `bt_row[k]` with a `k` that is not sequential because it jumps according to the location of A's nonzero values.
- thus our initial accesses to `bt_row` are likely all cache misses if the strides are not cache-friendly so that's why we better not trigger these misses again in the future.
- that's why we keep `bt_row` in cache while we iterate through rows of A.

## image_proc <!--! check requirement in PDF
there are many function calls here and the task is to inline when possible, especially since they are called within a nested loop so that means a lot of function call overhead. Systematically inlining shows that theory aligns with practicality.
- some functions are so small that their overhead is doing much more work than the actual computation, so better inline.
- nested function calls also cause a lot of overhead, inlining them reduces this and provides opportunities for more optimization by the compiler as discussed in lecture
- although some functions are big and complex, they are only called once within the loop body so inlining doesn't hurt that much but actually gives better performance when tested on real hardware.
- maybe if the functions are super large that they affect I-cache of the loop too drastically and give little benefit, or if they are called many times within the same loop body (causing inlining to bloat the code), then inlining them might not be worth it.

## filter_gradient <!--! check requirement in PDF
the problem here is the cache misses caused by data arrangement. Original SoA format places each channel of the image super far away from each other and inside the loop we access all channels at an index.
- this means accessing each channel would have high chance of being a cache miss and this happens to all channels
- reformatting the data into AoS format as an array of `struct pixel`s (each containing index `i` of all channels) helps but the channels contiguously near each other when we access index `i` so that we just need to trigger one miss for `a` and all other channels at that index are available to us <!-- TODO is this right? -->

## blackscholes
this kernel's optimization lie mainly in simplifying or approximating the mathematical expressions but it's really hard.
- first, I noticed that there was a long chain of data dependence so I transformed it in a way that would allow the compiler to maybe parallelize it in someway. This did speed up a little bit approximately from 0.6x to 0.8x
```cpp
// before
    float local = k * coefficient_a1;
    local += k_2 * coefficient_a2;
    local += k_3 * coefficient_a3;
    local += k_4 * coefficient_a4;
    local += k_5 * coefficient_a5;
    local = 1.0f - local * xNPrimeofX;
// after
    float local = k * coefficient_a1
    + k_2 * coefficient_a2
    + k_3 * coefficient_a3
    + k_4 * coefficient_a4
    + k_5 * coefficient_a5;
    local = 1.0f - local * xNPrimeofX;
```
- all operations were done in `float` so I tried to add `f` suffixes to the macros and the speedup got to 1x
- then I learned about Horner polynomial expressions and how it can really reduce the number of multiplications we have been doing for `k, k*k, k*k*k, ...` and tried that
```cpp
// k(a_1 + k(a_2 + k(a_3 + k(a_4 + k(a_5)))))
const float k = 1.0f / (1.0f + p_val * x);

    float poly = coefficient_a5;
    poly = poly * k + coefficient_a4;
    poly = poly * k + coefficient_a3;
    poly = poly * k + coefficient_a2;
    poly = poly * k + coefficient_a1;
    poly = poly * k;
```
- for the student CNDF function I also didn't understand why we were overwriting a referenced value instead of returning so I tried returning a value and it got faster. I'm not sure how C++ handles references as compared to pointers but maybe aliasing or load/store could have a role that affected the speed previously
- also, doing `xD1/xDen` after computing `xD1` is somehow slow. Replacing it by multiplying `invXDen` while computing final value of `xD1` is faster by a bit maybe because of scheduling dependencies
- the naive way used `inline` keyword for helpers so I also used that, and without it, the speedup got worse. So I guess inlining can help the compiler see more opportunities of what to optimize, also since we're calling the helper functions multiple times because of loop, minimizing the function overhead is a good thing
- for approximating `exp, log` using Taylor series I naively tried replacing them with four terms of the series and the output was totally wrong, so I tried more terms and still not good. I gave up on this approach but realized one part of the code had `exp(-rate*time)` where `rate, time` ranges `[0.0275,0.1], [0.1,1]` respectively and thus their product magnitude is close to zero. Finally, I can use an easy way to approximate using Taylor series but I still give up on other ones because I really have no idea
```cpp
// this is the taylor series of e^x but I expressed the polynomial using Horner since it is amazing to reduce the number of multiplications
return 1.0f + x * (1.0f + x * (0.5f + x * ((1.0f / 6.0f) + x * ((1.0f / 24.0f) + x * (1.0f / 120.0f))))); 
// also is an inline function for the same reasons of optimization
```
- I asked around for any more optimization techniques that I could understand and that actually work and one that worked is called Put-call Parity though I can't really understand why it's true since I don't do options trading analysis. Anyway, using this formula the speedup increased by 1x. Even though I didn't need to make this optimization, just wanted to make the speedup a nice looking number