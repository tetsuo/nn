#include "nn.h"

#include <math.h>

/* Vector helpers */

int nn_cpu_l2_normalize(float *vec, int dim) {
    if (!vec || dim <= 0)
        return -1;

    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++)
        norm_sq += vec[i] * vec[i];

    if (!(norm_sq > 0.0f) || !isfinite(norm_sq))
        return -1;

    float inv_norm = 1.0f / sqrtf(norm_sq);
    for (int i = 0; i < dim; i++)
        vec[i] *= inv_norm;
    return 0;
}
