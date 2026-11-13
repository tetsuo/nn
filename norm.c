/*
 * RMSNorm and the per-head RMSNorm used for QK-norm.
 */

#include "nn.h"
#include "threadpool.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_BLAS
#    ifdef __APPLE__
#        include <Accelerate/Accelerate.h>
#    else
#        include <cblas.h>
#    endif
#endif

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

/* Normalization */

/* Rows split across the pool only when the tensor is big enough to pay
 * for the dispatch. */
#define RMS_NORM_PARALLEL_ELEMS 262144

/* The sum of squares and the scaled write-back are plain loops: the compiler
 * auto-vectorizes both in place, which is faster than reducing through a
 * two-input dot kernel that reloads the row and is reduction-latency bound. */
static void rms_norm_range(
    float *out, const float *x, const float *weight, int start, int end, int hidden, float eps) {
    for (int s = start; s < end; s++) {
        const float *x_row = x + (size_t)s * hidden;
        float *out_row = out + (size_t)s * hidden;

        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; i++)
            sum_sq += x_row[i] * x_row[i];
        float rms_inv = 1.0f / sqrtf(sum_sq / hidden + eps);
        for (int i = 0; i < hidden; i++)
            out_row[i] = x_row[i] * rms_inv * weight[i];
    }
}

typedef struct {
    float *out;
    const float *x;
    const float *weight;
    int seq_len;
    int hidden;
    float eps;
} rms_norm_task_t;

static void rms_norm_worker(int tid, int n_threads, void *arg) {
    rms_norm_task_t *t = (rms_norm_task_t *)arg;
    int chunk = (t->seq_len + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->seq_len)
        end = t->seq_len;
    if (start >= end)
        return;
    rms_norm_range(t->out, t->x, t->weight, start, end, t->hidden, t->eps);
}

void nn_rms_norm(
    float *out, const float *x, const float *weight, int seq_len, int hidden, float eps) {
    if (seq_len <= 0 || hidden <= 0)
        return;

    long long elems = (long long)seq_len * hidden;
    if (nn_num_threads() > 1 && elems >= RMS_NORM_PARALLEL_ELEMS) {
        rms_norm_task_t task = {
            .out = out,
            .x = x,
            .weight = weight,
            .seq_len = seq_len,
            .hidden = hidden,
            .eps = eps,
        };
        nn_parallel_for(rms_norm_worker, &task);
    } else {
        rms_norm_range(out, x, weight, 0, seq_len, hidden, eps);
    }
}

void nn_rms_norm_per_head(
    float *x, const float *weight, int seq_len, int n_heads, int head_dim, float eps) {
    /* x is [seq, n_heads * head_dim] - normalize each [head_dim] segment */
    int hidden = n_heads * head_dim;
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; d++)
                sum_sq += vec[d] * vec[d];
            float rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);
            for (int d = 0; d < head_dim; d++)
                vec[d] = vec[d] * rms_inv * weight[d];
        }
    }
}
