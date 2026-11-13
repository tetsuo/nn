#include "nn_mlx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int arr_ok(mlx_array a) { return a.ctx != NULL; }

void nn_mlx_linear_free(nn_mlx_linear_t *l) {
    mlx_array_free(l->w);
    mlx_array_free(l->wt);
    memset(l, 0, sizeof(*l));
}

int nn_mlx_linear_prepare(nn_mlx_linear_t *l, mlx_stream s) {
    if (!l || !arr_ok(l->w) || arr_ok(l->wt))
        return 0;

    mlx_array transposed = mlx_array_new();
    mlx_transpose(&transposed, l->w, s);
    if (!arr_ok(transposed)) {
        mlx_array_free(transposed);
        fprintf(stderr, "mlx: failed to prepare transposed linear weight\n");
        return -1;
    }

    l->wt = mlx_array_new();
    mlx_contiguous(&l->wt, transposed, false, s);
    mlx_array_free(transposed);
    if (!arr_ok(l->wt)) {
        mlx_array_free(l->wt);
        l->wt = (mlx_array){0};
        fprintf(stderr, "mlx: failed to materialize transposed linear weight\n");
        return -1;
    }
    mlx_array_eval(l->wt);
    mlx_synchronize(s);
    mlx_array_free(l->w);
    l->w = (mlx_array){0};
    return 0;
}

mlx_array nn_mlx_linear(mlx_array x, const nn_mlx_linear_t *W, mlx_stream s) {
    mlx_array Wt;
    int free_wt = 0;
    if (arr_ok(W->wt)) {
        Wt = W->wt;
    } else if (arr_ok(W->w)) {
        Wt = mlx_array_new();
        mlx_transpose(&Wt, W->w, s);
        free_wt = 1;
    } else {
        return (mlx_array){0};
    }
    mlx_array res = mlx_array_new();
    if (mlx_array_dtype(Wt) == MLX_BFLOAT16 && mlx_array_dtype(x) == MLX_FLOAT32) {
        /* BF16 weight storage with F32 activations requires F32 matrix arithmetic. */
        mlx_array Wf = mlx_array_new();
        mlx_astype(&Wf, Wt, MLX_FLOAT32, s);
        mlx_matmul(&res, x, Wf, s);
        mlx_array_free(Wf);
    } else {
        mlx_matmul(&res, x, Wt, s);
    }
    if (free_wt)
        mlx_array_free(Wt);
    return res;
}

void nn_mlx_upcast_f32(mlx_array *a, mlx_stream s) {
    if (!a || !arr_ok(*a) || mlx_array_dtype(*a) == MLX_FLOAT32)
        return;
    mlx_array f = mlx_array_new();
    mlx_astype(&f, *a, MLX_FLOAT32, s);
    mlx_array_free(*a);
    *a = f;
}

mlx_array nn_mlx_linear_split(nn_mlx_matmul_t *ctx,
                                    mlx_array x,
                                    const nn_mlx_linear_t *w,
                                    mlx_stream stream) {
    if (!ctx || !w || !arr_ok(x))
        return (mlx_array){0};
    int transposed = arr_ok(w->wt);
    mlx_array weight = transposed ? w->wt : w->w;
    if (!arr_ok(weight) || mlx_array_ndim(weight) != 2)
        return (mlx_array){0};
    int k = mlx_array_dim(weight, transposed ? 0 : 1);
    int n = mlx_array_dim(weight, transposed ? 1 : 0);
    mlx_dtype dtype = mlx_array_dtype(weight);
    if (k <= 0 || n <= 0)
        return (mlx_array){0};
    /* Small products need native split-K parallelism. Tall reductions use a fixed partition count. */
    if (k < n || mlx_array_size(x) / k <= 64 || mlx_array_dtype(x) != MLX_FLOAT32 ||
        (dtype != MLX_FLOAT32 && dtype != MLX_BFLOAT16))
        return nn_mlx_linear(x, w, stream);
    mlx_array operand = mlx_array_new(), result = mlx_array_new();
    int rc = transposed ? mlx_array_set(&operand, weight) : mlx_transpose(&operand, weight, stream);
    if (!rc) {
        nn_mlx_upcast_f32(&operand, stream);
        rc = nn_mlx_matmul_split(ctx, &result, x, operand, stream);
    }
    mlx_array_free(operand);
    if (rc) {
        mlx_array_free(result);
        result = (mlx_array){0};
    }
    return result;
}
