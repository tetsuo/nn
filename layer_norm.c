#include "nn.h"

#include <math.h>

/* Mean-subtracting LayerNorm with bias. Safe in place (out may alias x): each
 * row's mean and variance are reduced before the row is rewritten. Scalar for
 * now; LayerNorm is not GEMM-bound, so SIMD/threading is a later perf step. */
void nn_cpu_layer_norm(float *out,
                           const float *x,
                           const float *gamma,
                           const float *beta,
                           int seq_len,
                           int hidden,
                           float eps) {
    if (seq_len <= 0 || hidden <= 0)
        return;
    float inv_h = 1.0f / (float)hidden;
    for (int s = 0; s < seq_len; s++) {
        const float *row = x + (size_t)s * hidden;
        float *dst = out + (size_t)s * hidden;
        float mean = 0.0f;
        for (int d = 0; d < hidden; d++)
            mean += row[d];
        mean *= inv_h;
        float var = 0.0f;
        for (int d = 0; d < hidden; d++) {
            float diff = row[d] - mean;
            var += diff * diff;
        }
        var *= inv_h;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int d = 0; d < hidden; d++)
            dst[d] = gamma[d] * (row[d] - mean) * inv_std + beta[d];
    }
}
