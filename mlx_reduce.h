#ifndef NN_MLX_REDUCE_H
#define NN_MLX_REDUCE_H

/* F32 register pairs retain product and addition residuals through reductions.
 * Keep the high and low parts separate until the final rounding, including
 * SIMD reductions where opposing terms can hide a small remainder. */
#define NN_METAL(...) #__VA_ARGS__
// clang-format off
static const char nn_mlx_reduce_source[] = NN_METAL(
struct fp32_pair {
    float hi, lo;
};
inline fp32_pair pair_add(fp32_pair a, fp32_pair b) {
    float sum = a.hi + b.hi, z = sum - a.hi;
    float tail = (a.hi - (sum - z)) + (b.hi - z);
    tail += a.lo + b.lo;
    float next = sum + tail;
    z = next - sum;
    return {next, (sum - (next - z)) + (tail - z)};
}
inline fp32_pair pair_scale(fp32_pair a, float b) {
    float product = a.hi * b;
    float tail = fma(a.hi, b, -product);
    tail = fma(a.lo, b, tail);
    // A normalized pair's product dominates its rounding residual. QuickTwoSum
    // retains that residual without another general addition of two pairs.
    float next = product + tail;
    return {next, tail - (next - product)};
}
inline fp32_pair pair_multiply(fp32_pair a, fp32_pair b) {
    return pair_add(pair_scale(a,b.hi), pair_scale({a.hi,0.0f},b.lo));
}
/* Dot2: accumulate exact product/TwoSum residuals in a separate F32 sum.
 * Normalize the pair once after the dot, rather than after every term. */
inline fp32_pair pair_dot_step(fp32_pair acc, float a, float b, float low) {
    float product=a*b, error=fma(a,b,-product);
    float sum=acc.hi+product,z=sum-acc.hi;
    float tail=(acc.hi-(sum-z))+(product-z);
    return {sum,acc.lo+((tail+error)+low)};
}
inline fp32_pair pair_reduce(fp32_pair a, uint lane) {
    for (uint delta = 16; delta > 0; delta /= 2) {
        fp32_pair b = {simd_shuffle_down(a.hi, delta), simd_shuffle_down(a.lo, delta)};
        if (lane + delta < 32)
            a = pair_add(a, b);
    }
    return {simd_broadcast(a.hi, 0), simd_broadcast(a.lo, 0)};
}
inline float pair_inverse(fp32_pair acc, float divisor, float eps, uint lane) {
    for (uint delta = 16; delta > 0; delta /= 2) {
        fp32_pair b = {simd_shuffle_down(acc.hi, delta), simd_shuffle_down(acc.lo, delta)};
        if (lane + delta < 32)
            acc = pair_add(acc, b);
    }
    float inv = 0;
    if (lane == 0) {
        float mean = acc.hi / divisor;
        float tail = (fma(-mean, divisor, acc.hi) + acc.lo) / divisor;
        fp32_pair m = pair_add({mean, tail}, {eps, 0.0f});
        float r = metal::precise::rsqrt(m.hi), square = r * r;
        float residual = fma(-m.hi, square, 1.0f);
        residual = fma(-m.hi, fma(r, r, -square), residual);
        residual = fma(-m.lo, square, residual);
        inv = fma(0.5f * r, residual, r);
    }
    return simd_broadcast(inv, 0);
}
);
// clang-format on
#undef NN_METAL
#endif
