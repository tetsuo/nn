/* SPDX-License-Identifier: Apache-2.0
 * See LICENSES/lily-NOTICE.txt for required notices. */
#include "nn_mlx.h"
#include "mlx_reduce.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* Stringify shader bodies so the library embeds its kernels and needs no
 * runtime source files or offline Metal compiler. */
#define METAL(...) #__VA_ARGS__

// clang-format off
static const char *conv_source = METAL(
    uint c = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint batch = thread_position_in_grid.z;
    uint T = x_shape[1], C = x_shape[2];
    if (c >= C || t >= T || batch >= uint(x_shape[0])) return;
    uint length = lengths[batch];
    ulong row = (ulong(batch) * T + t) * C;
    float acc = 0.0f;
    if (t < length) {
        for (uint tap = 0; tap < 4; ++tap) {
            int source = int(t) + int(tap) - 3;
            float v = source < 0 ? window_in[(ulong(batch) * 3 + (source + 3)) * C + c]
                                 : x[(ulong(batch) * T + source) * C + c];
            acc += v * weight[c * 4 + tap];
        }
        out[row + c] = acc / (1.0f + exp(-acc));
    } else {
        out[row + c] = 0.0f;
    }
    if (t == 0) {
        for (int tap = 0; tap < 3; ++tap) {
            int source = int(length) - 3 + tap;
            window_out[(ulong(batch) * 3 + tap) * C + c] =
                source < 0 ? window_in[(ulong(batch) * 3 + (source + 3)) * C + c]
                           : x[(ulong(batch) * T + source) * C + c];
        }
    }
);

/* One SIMD group per value head and token. Q/K are shared by groups of
 * value heads; only the first HK groups write their normalized Q/K rows. */
static const char *prepare_source = METAL(
    uint lane = thread_index_in_simdgroup;
    uint h = thread_position_in_grid.y;
    uint row = thread_position_in_grid.z;
    uint H = a_shape[2], HK = key_heads[0];
    uint C = qkv_shape[2];
    if (h < HK) {
        float q[4], k[4];
        fp32_pair sq = {0.0f, 0.0f}, sk = {0.0f, 0.0f};
        for (uint i = 0; i < 4; ++i) {
            uint d = lane * 4 + i;
            q[i] = qkv[ulong(row) * C + h * 128 + d];
            k[i] = qkv[ulong(row) * C + (HK + h) * 128 + d];
            float qs = q[i] * q[i], ks = k[i] * k[i];
            sq = pair_add(sq, {qs, fma(q[i], q[i], -qs)});
            sk = pair_add(sk, {ks, fma(k[i], k[i], -ks)});
        }
        float iq = pair_inverse(sq, 1.0f, 1e-6f, lane);
        float ik = pair_inverse(sk, 1.0f, 1e-6f, lane);
        for (uint i = 0; i < 4; ++i) {
            uint d = lane * 4 + i;
            ulong offset = (ulong(row) * HK + h) * 256 + d;
            // Normalize first, then apply the attention scale, as in FLA.
            qk[offset] = (q[i] * iq) * 0.08838834764831845f;
            qk[offset + 128] = k[i] * ik;
        }
    }
    if (lane == 0) {
        float v = a[ulong(row) * H + h] + dt_bias[h];
        float softplus = v > 20.0f ? v : log1p(exp(v));
        ulong offset = (ulong(row) * H + h) * 3;
        float g = -exp(a_log[h]) * softplus;
        float increment = 0.0f;
        float decay = exp(g);
        if (g > -0.01f) {
            // expm1(g) through degree five; truncation is below F32 rounding
            // on [-.01, 0]. Keep 1 + increment as two coefficients.
            float p = fma(g, 1.0f / 120.0f, 1.0f / 24.0f);
            p = fma(g, p, 1.0f / 6.0f);
            p = fma(g, p, 0.5f);
            increment = fma(g * g, p, g);
            decay = 1.0f;
        }
        gates[offset] = decay;
        gates[offset + 1] = increment;
        gates[offset + 2] = 1.0f / (1.0f + exp(-b[ulong(row) * H + h]));
    }
);

/* One SIMD group owns one value column; each lane holds four key rows.
 * High/low F32 register pairs retain decay and update rounding within each
 * sequence; state arguments remain F32. This preserves weak decay across
 * thousands of steps. Prediction and output dots also retain the small
 * remainders between opposing terms. Distinct input/output state arrays
 * preserve MLX graph dependencies, including when chunks are chained. */
static const char *scan_source = METAL(
    uint lane = thread_index_in_simdgroup;
    uint dv = thread_position_in_grid.y;
    uint bh = thread_position_in_grid.z;
    uint H = state_in_shape[1], HK = qk_shape[2], T = qkv_shape[1];
    uint batch = bh / H, h = bh % H, hk = h / (H / HK);
    uint C = qkv_shape[2];
    ulong state_offset = (ulong(bh) * 128 + lane * 4) * 128 + dv;
    float state[4], low[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint i = 0; i < 4; ++i)
        state[i] = state_in[state_offset + i * 128];
    for (uint t = 0; t < T; ++t) {
        ulong row = ulong(batch) * T + t;
        float value = 0.0f;
        if (t < uint(lengths[batch])) {
            ulong offset = (row * HK + hk) * 256 + lane * 4;
            float decay = gates[(row * H + h) * 3];
            float increment = gates[(row * H + h) * 3 + 1];
            float beta = gates[(row * H + h) * 3 + 2];
            fp32_pair prediction = {0.0f, 0.0f};
            for (uint i = 0; i < 4; ++i) {
                float product = state[i] * decay;
                float tail = fma(state[i], decay, -product);
                tail = fma(low[i], decay, tail);
                tail = fma(state[i], increment, tail);
                tail = fma(low[i], increment, tail);
                float next = product + tail;
                float z = next - product;
                low[i] = (product - (next - z)) + (tail - z);
                state[i] = next;
                prediction =
                    pair_add(prediction, pair_scale({state[i], low[i]}, qk[offset + 128 + i]));
            }
            prediction = pair_reduce(prediction, lane);
            float input = qkv[row * C + (2 * HK + h) * 128 + dv];
            fp32_pair delta =
                pair_scale(pair_add({input, 0.0f}, {-prediction.hi, -prediction.lo}), beta);
            fp32_pair output = {0.0f, 0.0f};
            for (uint i = 0; i < 4; ++i) {
                fp32_pair next =
                    pair_add({state[i], low[i]}, pair_scale(delta, qk[offset + 128 + i]));
                state[i] = next.hi;
                low[i] = next.lo;
                output = pair_add(output, pair_scale(next, qk[offset + i]));
            }
            output = pair_reduce(output, lane);
            value = output.hi + output.lo;
        }
        if (lane == 0)
            out[(row * H + h) * 128 + dv] = value;
    }
    for (uint i = 0; i < 4; ++i)
        state_out[state_offset + i * 128] = state[i];
);

static const char *norm_source = METAL(
    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.y;
    float values[4];
    fp32_pair ss = {0.0f, 0.0f};
    for (uint i = 0; i < 4; ++i) {
        values[i] = x[ulong(row) * 128 + lane * 4 + i];
        float square = values[i] * values[i];
        ss = pair_add(ss, {square, fma(values[i], values[i], -square)});
    }
    float inv = pair_inverse(ss, 128.0f, epsilon[0], lane);
    for (uint i = 0; i < 4; ++i) {
        uint d = lane * 4 + i;
        ulong offset = ulong(row) * 128 + d;
        float gate = z[offset];
        out[offset] = values[i] * inv * weight[d] * (gate / (1.0f + exp(-gate)));
    }
);
// clang-format on
#undef METAL

struct nn_mlx_gdn {
    mlx_fast_metal_kernel conv, prepare, scan, norm;
};

static mlx_fast_metal_kernel kernel(const char *name,
                                    const char **inputs,
                                    size_t n_inputs,
                                    const char **outputs,
                                    size_t n_outputs,
                                    const char *source) {
    mlx_vector_string in = mlx_vector_string_new_data(inputs, n_inputs);
    mlx_vector_string out = mlx_vector_string_new_data(outputs, n_outputs);
    mlx_fast_metal_kernel k =
        mlx_fast_metal_kernel_new(name, in, out, source, nn_mlx_reduce_source, true, false);
    mlx_vector_string_free(in);
    mlx_vector_string_free(out);
    return k;
}

nn_mlx_gdn_t *nn_mlx_gdn_new(void) {
    nn_mlx_gdn_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return NULL;
    const char *conv_in[] = {"x", "weight", "window_in", "lengths"};
    const char *conv_out[] = {"out", "window_out"};
    const char *prepare_in[] = {"qkv", "a", "b", "a_log", "dt_bias", "key_heads"};
    const char *prepare_out[] = {"qk", "gates"};
    const char *scan_in[] = {"qkv", "qk", "gates", "state_in", "lengths"};
    const char *scan_out[] = {"out", "state_out"};
    const char *norm_in[] = {"x", "z", "weight", "epsilon"};
    const char *norm_out[] = {"out"};
    ctx->conv = kernel("nn_gdn_conv", conv_in, 4, conv_out, 2, conv_source);
    ctx->prepare = kernel("nn_gdn_prepare", prepare_in, 6, prepare_out, 2, prepare_source);
    ctx->scan = kernel("nn_gdn_scan", scan_in, 5, scan_out, 2, scan_source);
    ctx->norm = kernel("nn_gdn_norm", norm_in, 4, norm_out, 1, norm_source);
    if (!ctx->conv.ctx || !ctx->prepare.ctx || !ctx->scan.ctx || !ctx->norm.ctx) {
        nn_mlx_gdn_free(ctx);
        return NULL;
    }
    return ctx;
}

void nn_mlx_gdn_free(nn_mlx_gdn_t *ctx) {
    if (!ctx)
        return;
    mlx_fast_metal_kernel_free(ctx->conv);
    mlx_fast_metal_kernel_free(ctx->prepare);
    mlx_fast_metal_kernel_free(ctx->scan);
    mlx_fast_metal_kernel_free(ctx->norm);
    free(ctx);
}

static int shape(mlx_array a, const int *dims, int rank) {
    if (!a.ctx || mlx_array_dtype(a) != MLX_FLOAT32 || mlx_array_ndim(a) != (size_t)rank)
        return 0;
    for (int i = 0; i < rank; ++i)
        if (dims[i] <= 0 || mlx_array_dim(a, i) != dims[i])
            return 0;
    return 1;
}

static int lengths_valid(const int *lengths, int batch, int tokens) {
    if (!lengths || batch <= 0 || tokens <= 0 || batch > INT_MAX / tokens)
        return 0;
    for (int i = 0; i < batch; ++i)
        if (lengths[i] < 0 || lengths[i] > tokens)
            return 0;
    return 1;
}

static int apply(mlx_fast_metal_kernel k,
                 mlx_array *out,
                 mlx_array *second,
                 const mlx_array *inputs,
                 size_t n_inputs,
                 const int *dims,
                 int rank,
                 const int *second_dims,
                 int second_rank,
                 const int grid[3],
                 const int group[3],
                 mlx_stream stream) {
    mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
    mlx_vector_array in = mlx_vector_array_new_data(inputs, n_inputs);
    mlx_vector_array outputs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_config_add_output_arg(config, dims, rank, MLX_FLOAT32);
    if (!rc && second)
        rc = mlx_fast_metal_kernel_config_add_output_arg(config, second_dims, second_rank,
                                                         MLX_FLOAT32);
    if (!rc)
        rc = mlx_fast_metal_kernel_config_set_grid(config, grid[0], grid[1], grid[2]);
    if (!rc)
        rc = mlx_fast_metal_kernel_config_set_thread_group(config, group[0], group[1], group[2]);
    if (!rc)
        rc = mlx_fast_metal_kernel_apply(&outputs, k, in, config, stream);
    if (!rc)
        rc = mlx_vector_array_get(out, outputs, 0);
    if (!rc && second)
        rc = mlx_vector_array_get(second, outputs, 1);
    mlx_vector_array_free(outputs);
    mlx_vector_array_free(in);
    mlx_fast_metal_kernel_config_free(config);
    return rc ? -1 : 0;
}

int nn_mlx_gdn_conv(nn_mlx_gdn_t *ctx,
                        mlx_array *out,
                        mlx_array *window_out,
                        mlx_array x,
                        mlx_array weight,
                        mlx_array window_in,
                        const int *lengths,
                        mlx_stream stream) {
    if (!ctx || !out || !window_out || out == window_out || !x.ctx || mlx_array_ndim(x) != 3)
        return -1;
    int batch = mlx_array_dim(x, 0), tokens = mlx_array_dim(x, 1), channels = mlx_array_dim(x, 2);
    int dims[] = {batch, tokens, channels}, wshape[] = {channels, 4},
        history[] = {batch, 3, channels};
    if (!shape(x, dims, 3) || !shape(weight, wshape, 2) || !shape(window_in, history, 3) ||
        !lengths_valid(lengths, batch, tokens))
        return -1;
    mlx_array lens = mlx_array_new_data(lengths, &batch, 1, MLX_INT32);
    mlx_array inputs[] = {x, weight, window_in, lens};
    int grid[] = {channels, tokens, batch}, group[] = {channels < 256 ? channels : 256, 1, 1};
    int rc = apply(ctx->conv, out, window_out, inputs, 4, dims, 3, history, 3, grid, group, stream);
    mlx_array_free(lens);
    return rc;
}

int nn_mlx_gdn_scan(nn_mlx_gdn_t *ctx,
                        mlx_array *out,
                        mlx_array *state_out,
                        mlx_array qkv,
                        mlx_array a,
                        mlx_array b,
                        mlx_array a_log,
                        mlx_array dt_bias,
                        mlx_array state_in,
                        const int *lengths,
                        int key_heads,
                        mlx_stream stream) {
    if (!ctx || !out || !state_out || out == state_out || !qkv.ctx || !a_log.ctx ||
        mlx_array_ndim(qkv) != 3 || mlx_array_ndim(a_log) != 1)
        return -1;
    int batch = mlx_array_dim(qkv, 0), tokens = mlx_array_dim(qkv, 1),
        heads = mlx_array_dim(a_log, 0);
    if (heads <= 0 || key_heads <= 0 || heads % key_heads || heads > INT_MAX / 384 ||
        !lengths_valid(lengths, batch, tokens) || batch > INT_MAX / heads)
        return -1;
    int channels = (2 * key_heads + heads) * 128;
    int input_shape[] = {batch, tokens, channels}, gate_shape[] = {batch, tokens, heads};
    int output_shape[] = {batch, tokens, heads, 128}, state_shape[] = {batch, heads, 128, 128};
    if (!shape(qkv, input_shape, 3) || !shape(a, gate_shape, 3) || !shape(b, gate_shape, 3) ||
        !shape(a_log, &heads, 1) || !shape(dt_bias, &heads, 1) || !shape(state_in, state_shape, 4))
        return -1;
    int one = 1;
    mlx_array hk = mlx_array_new_data(&key_heads, &one, 1, MLX_INT32);
    mlx_array lens = mlx_array_new_data(lengths, &batch, 1, MLX_INT32);
    mlx_array qk = mlx_array_new(), gates = mlx_array_new();
    mlx_array prep[] = {qkv, a, b, a_log, dt_bias, hk};
    int qk_shape[] = {batch, tokens, key_heads, 256}, gates_shape[] = {batch, tokens, heads, 3};
    int prep_grid[] = {32, heads, batch * tokens}, prep_group[] = {32, 1, 1};
    int rc = apply(ctx->prepare, &qk, &gates, prep, 6, qk_shape, 4, gates_shape, 4, prep_grid,
                   prep_group, stream);
    if (!rc) {
        mlx_array inputs[] = {qkv, qk, gates, state_in, lens};
        int grid[] = {32, 128, batch * heads}, group[] = {32, 4, 1};
        rc = apply(ctx->scan, out, state_out, inputs, 5, output_shape, 4, state_shape, 4, grid, group,
                   stream);
    }
    mlx_array_free(qk);
    mlx_array_free(gates);
    mlx_array_free(hk);
    mlx_array_free(lens);
    return rc;
}

int nn_mlx_gdn_norm(nn_mlx_gdn_t *ctx,
                        mlx_array *out,
                        mlx_array x,
                        mlx_array z,
                        mlx_array weight,
                        float eps,
                        mlx_stream stream) {
    if (!ctx || !out || !x.ctx || mlx_array_ndim(x) != 4 || !isfinite(eps) || eps <= 0)
        return -1;
    int dims[] = {mlx_array_dim(x, 0), mlx_array_dim(x, 1), mlx_array_dim(x, 2), 128};
    int width = 128;
    if (!shape(x, dims, 4) || !shape(z, dims, 4) || !shape(weight, &width, 1) ||
        mlx_array_size(x) / 128 > INT_MAX)
        return -1;
    int rows = (int)(mlx_array_size(x) / 128), one = 1;
    mlx_array epsilon = mlx_array_new_data(&eps, &one, 1, MLX_FLOAT32);
    mlx_array inputs[] = {x, z, weight, epsilon};
    int grid[] = {32, rows, 1}, group[] = {32, 1, 1};
    int rc = apply(ctx->norm, out, NULL, inputs, 4, dims, 4, NULL, 0, grid, group, stream);
    mlx_array_free(epsilon);
    return rc;
}
