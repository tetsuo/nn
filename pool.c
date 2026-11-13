#include "nn.h"

#include <math.h>
#include <string.h>

int nn_cpu_pool_rms(float *out_embeddings,
                        const float *x,
                        const float *norm_weight,
                        const int *offsets,
                        int batch,
                        int hidden,
                        float eps,
                        int last_token,
                        int normalize) {
    if (hidden <= 0 || !x)
        return -1;
    for (int b = 0; b < batch; b++) {
        int start = offsets[b];
        int end = offsets[b + 1];
        int len = end - start;
        float *emb = out_embeddings + (size_t)b * hidden;

        if (last_token) {
            /* Only the final row needs normalization for last-row pooling. */
            int idx = end - 1;
            const float *row = x + (size_t)idx * hidden;
            if (norm_weight) {
                float sum_sq = 0.0f;
                for (int d = 0; d < hidden; d++)
                    sum_sq += row[d] * row[d];
                float rms_inv = 1.0f / sqrtf(sum_sq / (float)hidden + eps);
                for (int d = 0; d < hidden; d++)
                    emb[d] = row[d] * rms_inv * norm_weight[d];
            } else {
                memcpy(emb, row, (size_t)hidden * sizeof(float));
            }
        } else {
            memset(emb, 0, (size_t)hidden * sizeof(float));
            for (int i = start; i < end; i++) {
                const float *row = x + (size_t)i * hidden;
                if (norm_weight) {
                    float sum_sq = 0.0f;
                    for (int d = 0; d < hidden; d++)
                        sum_sq += row[d] * row[d];
                    float rms_inv = 1.0f / sqrtf(sum_sq / (float)hidden + eps);
                    for (int d = 0; d < hidden; d++)
                        emb[d] += row[d] * rms_inv * norm_weight[d];
                } else {
                    for (int d = 0; d < hidden; d++)
                        emb[d] += row[d];
                }
            }
            float inv_len = 1.0f / (float)len;
            for (int d = 0; d < hidden; d++)
                emb[d] *= inv_len;
        }

        if (normalize && nn_cpu_l2_normalize(emb, hidden) != 0)
            return -1;
    }
    return 0;
}
