#include "nn_mlx.h"
#include "mlx_reduce.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>

#define METAL(...) #__VA_ARGS__
/* One SIMD group normalizes each row. Retain reduction and multiplication
 * residuals until the output so repeated norms do not amplify avoidable loss. */
// clang-format off
static const char *rms_source = METAL(
    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.y;
    uint width = weight_shape[0];
    ulong offset = ulong(row) * width;
    fp32_pair acc = {0.0f, 0.0f};
    for (uint d = lane; d < width; d += 32) {
        float v = x[offset + d];
        float square = v * v;
        acc = pair_add(acc, {square, fma(v, v, -square)});
    }
    acc = pair_reduce(acc, lane);
    fp32_pair inv = {0.0f, 0.0f};
    if (lane == 0) {
        float mean = acc.hi / float(width);
        float tail = (fma(-mean, float(width), acc.hi) + acc.lo) / float(width);
        fp32_pair m = pair_add({mean, tail}, {epsilon[0], 0.0f});
        float r = metal::precise::rsqrt(m.hi);
        float square = r * r;
        float residual = fma(-m.hi, square, 1.0f);
        residual = fma(-m.hi, fma(r, r, -square), residual);
        residual = fma(-m.lo, square, residual);
        inv = pair_add({r, 0.0f}, {0.5f * r * residual, 0.0f});
    }
    inv.hi = simd_broadcast(inv.hi, 0);
    inv.lo = simd_broadcast(inv.lo, 0);
    for (uint d = lane; d < width; d += 32) {
        fp32_pair normalized = pair_scale(inv, x[offset + d]);
        fp32_pair result = pair_scale(normalized, weight[d]);
        out[offset + d] = result.hi + result.lo;
    }
);
static const char *layer_source = METAL(
    uint lane=thread_index_in_simdgroup,row=thread_position_in_grid.y;
    uint width=weight_shape[0]; ulong offset=ulong(row)*width;
    fp32_pair sum={0.0f,0.0f};
    for(uint d=lane;d<width;d+=32)sum=pair_add(sum,{x[offset+d],0.0f});
    sum=pair_reduce(sum,lane);
    float mh=sum.hi/float(width);
    float ml=(fma(-mh,float(width),sum.hi)+sum.lo)/float(width);
    fp32_pair variance={0.0f,0.0f};
    for(uint d=lane;d<width;d+=32) {
        fp32_pair centered=pair_add({x[offset+d],0.0f},{-mh,-ml});
        variance=pair_add(variance,pair_multiply(centered,centered));
    }
    variance=pair_reduce(variance,lane);
    fp32_pair inverse={0.0f,0.0f};
    if(lane==0) {
        float mean=variance.hi/float(width);
        float low=(fma(-mean,float(width),variance.hi)+variance.lo)/float(width);
        fp32_pair v=pair_add({mean,low},{epsilon[0],0.0f});
        float r=metal::precise::rsqrt(v.hi),square=r*r;
        float residual=fma(-v.hi,square,1.0f);
        residual=fma(-v.hi,fma(r,r,-square),residual);
        residual=fma(-v.lo,square,residual);
        inverse=pair_add({r,0.0f},{0.5f*r*residual,0.0f});
    }
    inverse.hi=simd_broadcast(inverse.hi,0);inverse.lo=simd_broadcast(inverse.lo,0);
    for(uint d=lane;d<width;d+=32) {
        fp32_pair centered=pair_add({x[offset+d],0.0f},{-mh,-ml});
        fp32_pair norm=pair_scale(pair_multiply(centered,inverse),weight[d]);
        fp32_pair value=pair_add(norm,{bias[d],0.0f});
        out[offset+d]=value.hi+value.lo;
    }
);
// clang-format on
#undef METAL

struct nn_mlx_norm {
    mlx_fast_metal_kernel rms;
    mlx_fast_metal_kernel layer;
};

nn_mlx_norm_t *nn_mlx_norm_new(void) {
    nn_mlx_norm_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return NULL;
    const char *inputs[] = {"x", "weight", "epsilon"}, *outputs[] = {"out"};
    mlx_vector_string in = mlx_vector_string_new_data(inputs, 3);
    mlx_vector_string out = mlx_vector_string_new_data(outputs, 1);
    ctx->rms = mlx_fast_metal_kernel_new("nn_rms", in, out, rms_source, nn_mlx_reduce_source,
                                         true, false);
    mlx_vector_string_free(in);
    mlx_vector_string_free(out);
    const char *ln_inputs[] = {"x", "weight", "bias", "epsilon"};
    in = mlx_vector_string_new_data(ln_inputs, 4);
    out = mlx_vector_string_new_data(outputs, 1);
    ctx->layer = mlx_fast_metal_kernel_new("nn_layer_norm", in, out, layer_source,
                                           nn_mlx_reduce_source, true, false);
    mlx_vector_string_free(in);
    mlx_vector_string_free(out);
    if (!ctx->rms.ctx || !ctx->layer.ctx) {
        nn_mlx_norm_free(ctx);
        return NULL;
    }
    return ctx;
}

void nn_mlx_norm_free(nn_mlx_norm_t *ctx) {
    if (!ctx)
        return;
    mlx_fast_metal_kernel_free(ctx->rms);
    mlx_fast_metal_kernel_free(ctx->layer);
    free(ctx);
}

int nn_mlx_rms_norm(nn_mlx_norm_t *ctx,
                        mlx_array *out,
                        mlx_array x,
                        mlx_array weight,
                        float eps,
                        mlx_stream stream) {
    if (!ctx || !out || !x.ctx || !weight.ctx || !isfinite(eps) || eps <= 0 ||
        mlx_array_dtype(x) != MLX_FLOAT32 || mlx_array_dtype(weight) != MLX_FLOAT32 ||
        mlx_array_ndim(weight) != 1)
        return -1;
    int rank = (int)mlx_array_ndim(x), dims[4];
    if (rank < 1 || rank > 4)
        return -1;
    for (int i = 0; i < rank; ++i) {
        dims[i] = mlx_array_dim(x, i);
        if (dims[i] <= 0)
            return -1;
    }
    int width = dims[rank - 1];
    size_t rows = mlx_array_size(x) / width;
    if (width != mlx_array_dim(weight, 0) || rows > INT_MAX)
        return -1;
    int one = 1;
    mlx_array epsilon = mlx_array_new_data(&eps, &one, 1, MLX_FLOAT32);
    mlx_array args[] = {x, weight, epsilon};
    mlx_vector_array inputs = mlx_vector_array_new_data(args, 3);
    mlx_vector_array outputs = mlx_vector_array_new();
    mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
    int rc = mlx_fast_metal_kernel_config_add_output_arg(config, dims, rank, MLX_FLOAT32) ||
             mlx_fast_metal_kernel_config_set_grid(config, 32, (int)rows, 1) ||
             mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1) ||
             mlx_fast_metal_kernel_apply(&outputs, ctx->rms, inputs, config, stream) ||
             mlx_vector_array_get(out, outputs, 0);
    mlx_fast_metal_kernel_config_free(config);
    mlx_vector_array_free(inputs);
    mlx_vector_array_free(outputs);
    mlx_array_free(epsilon);
    return rc ? -1 : 0;
}

int nn_mlx_layer_norm(nn_mlx_norm_t *ctx,
                          mlx_array *out,
                          mlx_array x,
                          mlx_array weight,
                          mlx_array bias,
                          float eps,
                          mlx_stream stream) {
    if (!ctx || !out || !x.ctx || !weight.ctx || !bias.ctx || !isfinite(eps) || eps <= 0 ||
        mlx_array_dtype(x) != MLX_FLOAT32 || mlx_array_dtype(weight) != MLX_FLOAT32 ||
        mlx_array_ndim(weight) != 1 || mlx_array_dtype(bias) != MLX_FLOAT32 ||
        mlx_array_ndim(bias) != 1)
        return -1;
    int rank = (int)mlx_array_ndim(x), dims[4];
    if (rank < 1 || rank > 4)
        return -1;
    for (int i = 0; i < rank; ++i) {
        dims[i] = mlx_array_dim(x, i);
        if (dims[i] <= 0)
            return -1;
    }
    int width = dims[rank - 1];
    size_t rows = mlx_array_size(x) / width;
    if (width != mlx_array_dim(weight, 0) || width != mlx_array_dim(bias, 0) || rows > INT_MAX)
        return -1;
    int one = 1;
    mlx_array epsilon = mlx_array_new_data(&eps, &one, 1, MLX_FLOAT32);
    mlx_array args[] = {x, weight, bias, epsilon};
    mlx_vector_array inputs = mlx_vector_array_new_data(args, 4);
    mlx_vector_array outputs = mlx_vector_array_new();
    mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
    int rc = mlx_fast_metal_kernel_config_add_output_arg(config, dims, rank, MLX_FLOAT32) ||
             mlx_fast_metal_kernel_config_set_grid(config, 32, (int)rows, 1) ||
             mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1) ||
             mlx_fast_metal_kernel_apply(&outputs, ctx->layer, inputs, config, stream) ||
             mlx_vector_array_get(out, outputs, 0);
    mlx_fast_metal_kernel_config_free(config);
    mlx_vector_array_free(inputs);
    mlx_vector_array_free(outputs);
    mlx_array_free(epsilon);
    return rc ? -1 : 0;
}
