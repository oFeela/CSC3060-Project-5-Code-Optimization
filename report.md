
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
the apparent bottleneck is the exponent, log functions that may take a long time to compute, plus some data dependence stalls
- since `stu_BlkSchls` calls other functions in a loop, it is important to optimize every iteration by optimizing the functions that were called in the loop body.
- replacing log and exponent by their respective taylor series (1/ln10 constant multiplied to ln() taylor series) gives us a much faster way to compute a good-enough approximation of the true value
- also optimized `CNDF()` by converting an unnecessary data-dependent operation into an expression that the compiler could optimize further (maybe by using vectorization reduction): `float local = k * coefficient_a1 + k_2 * coefficient_a2 + k_3 * coefficient_a3 + k_4 * coefficient_a4 + k_5 * coefficient_a5;`

