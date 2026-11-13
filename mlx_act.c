#include "nn_mlx.h"

/* Exact erf GeLU: 0.5 * x * (1 + erf(x / sqrt(2))). Frees nothing it is given. */
mlx_array nn_mlx_gelu_erf(mlx_array x, mlx_stream S) {
    mlx_array c = mlx_array_new_float32(0.70710678118654752f);
    mlx_array scaled = mlx_array_new();
    mlx_multiply(&scaled, x, c, S);
    mlx_array_free(c);
    mlx_array e = mlx_array_new();
    mlx_erf(&e, scaled, S);
    mlx_array_free(scaled);
    mlx_array one = mlx_array_new_float32(1.0f);
    mlx_array e1 = mlx_array_new();
    mlx_add(&e1, e, one, S);
    mlx_array_free(e);
    mlx_array_free(one);
    mlx_array half = mlx_array_new_float32(0.5f);
    mlx_array xh = mlx_array_new();
    mlx_multiply(&xh, x, half, S);
    mlx_array_free(half);
    mlx_array g = mlx_array_new();
    mlx_multiply(&g, xh, e1, S);
    mlx_array_free(xh);
    mlx_array_free(e1);
    return g;
}

/* Tanh-approximation GeLU (gelu_new / gelu_pytorch_tanh): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))). Frees nothing given. */
mlx_array nn_mlx_gelu_tanh(mlx_array x, mlx_stream S) {
    mlx_array x2 = mlx_array_new();
    mlx_multiply(&x2, x, x, S);
    mlx_array x3 = mlx_array_new();
    mlx_multiply(&x3, x2, x, S);
    mlx_array_free(x2);
    mlx_array c = mlx_array_new_float32(0.044715f);
    mlx_array cx3 = mlx_array_new();
    mlx_multiply(&cx3, x3, c, S);
    mlx_array_free(x3);
    mlx_array_free(c);
    mlx_array sum = mlx_array_new();
    mlx_add(&sum, x, cx3, S);
    mlx_array_free(cx3);
    mlx_array s2pi = mlx_array_new_float32(0.79788456080286536f);
    mlx_array inner = mlx_array_new();
    mlx_multiply(&inner, sum, s2pi, S);
    mlx_array_free(sum);
    mlx_array_free(s2pi);
    mlx_array t = mlx_array_new();
    mlx_tanh(&t, inner, S);
    mlx_array_free(inner);
    mlx_array one = mlx_array_new_float32(1.0f);
    mlx_array t1 = mlx_array_new();
    mlx_add(&t1, t, one, S);
    mlx_array_free(t);
    mlx_array_free(one);
    mlx_array half = mlx_array_new_float32(0.5f);
    mlx_array xh = mlx_array_new();
    mlx_multiply(&xh, x, half, S);
    mlx_array_free(half);
    mlx_array g = mlx_array_new();
    mlx_multiply(&g, xh, t1, S);
    mlx_array_free(xh);
    mlx_array_free(t1);
    return g;
}
