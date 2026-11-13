#include "nn_mlx.h"

#include <limits.h>
#include <stdlib.h>

/* Each group computes a 32x32 output tile with four SIMD groups. Sixteen-wide
 * shared tiles bound local memory; padding avoids shared-memory bank conflicts.
 * Combine 64-term products with a binary carry stack, retaining the balanced
 * addition order without writing partial output matrices to device memory. */
#define METAL(...) #__VA_ARGS__
// clang-format off
static const char *matmul_source = METAL(
    constexpr uint BM = 32, BN = 32, RM = 2, RN = 2;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    // Each lane owns two adjacent columns of an 8x8 matrix fragment.
    uint fr = ((lane / 4) & 4) + ((lane / 2) % 4);
    uint fc = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    uint M = x_shape[0] / BATCHES, K = x_shape[1], N = w_shape[1];
    ulong batch = BATCHES == 1 ? 0 : threadgroup_position_in_grid.z;
    auto batch_x = x + batch * M * K;
    auto batch_w = w + batch * K * N;
    auto batch_out = out + batch * M * N;
    // Traverse each strip by row first so nearby groups can reuse weight tiles.
    // The final strip may contain fewer than eight row tiles.
    uint tile_rows = (M + BM - 1) / BM, tile_cols = (N + BN - 1) / BN;
    uint first_row = (threadgroup_position_in_grid.y / 8) * 8;
    uint group_rows = min(uint(8), tile_rows - first_row);
    uint index = (threadgroup_position_in_grid.y - first_row) * tile_cols
               + threadgroup_position_in_grid.x;
    uint row = (first_row + index % group_rows) * BM;
    uint col = (index / group_rows) * BN;
    alignas(16) threadgroup float A[BM * 20];
    alignas(16) threadgroup float B[16 * (BN + 4)];
    simdgroup_float8x8 partial[RM][RN];
    float bins[LEVELS][RM][RN][2];
    float accum[RM][RN][2];
    for (uint base = 0; base < K; base += 64) {
        NN_UNROLL
        for (uint i = 0; i < RM; i++) {
            NN_UNROLL
            for (uint j = 0; j < RN; j++) {
                partial[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0);
            }
        }
        for (uint tile = 0; tile < 64; tile += 16) {
            if (VECTOR) {
                for (uint i = lid * 4; i < BM * 16; i += 512) {
                    uint r = row + i / 16, k = base + tile + i % 16;
                    ulong pos = ulong(r) * K + k;
                    float4 v = r < M ? float4(batch_x[pos], batch_x[pos+1], batch_x[pos+2], batch_x[pos+3]) : float4(0);
                    *reinterpret_cast<threadgroup float4*>(A + (i / 16) * 20 + i % 16) = v;
                }
                for (uint i = lid * 4; i < 16 * BN; i += 512) {
                    uint k = i / BN, n = i % BN;
                    ulong pos = ulong(base + tile + k) * N + col + n;
                    float4 v = col + n < N ? float4(batch_w[pos], batch_w[pos+1], batch_w[pos+2], batch_w[pos+3]) : float4(0);
                    *reinterpret_cast<threadgroup float4*>(B + k * (BN + 4) + n) = v;
                }
            } else {
                for (uint i = lid; i < BM * 16; i += 128) {
                    uint r = row + i / 16, k = base + tile + i % 16;
                    A[(i / 16) * 20 + i % 16] = r < M && k < K ? batch_x[ulong(r) * K + k] : 0;
                }
                for (uint i = lid; i < 16 * BN; i += 128) {
                    uint k = i / BN, n = i % BN;
                    B[k * (BN + 4) + n] = base + tile + k < K && col + n < N
                        ? batch_w[ulong(base + tile + k) * N + col + n] : 0;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            NN_UNROLL
            for (uint k = 0; k < 16; k += 8) {
                simdgroup_float8x8 a[RM], b[RN];
                NN_UNROLL
                for (uint i = 0; i < RM; i++) {
                    NN_UNROLL
                    for (uint e = 0; e < 2; e++) {
                        a[i].thread_elements()[e] = A[(((sg / 2) * RM + i) * 8 + fr) * 20 + k + fc + e];
                    }
                }
                NN_UNROLL
                for (uint j = 0; j < RN; j++) {
                    NN_UNROLL
                    for (uint e = 0; e < 2; e++) {
                        b[j].thread_elements()[e] = B[(k + fr) * (BN + 4) + ((sg % 2) * RN + j) * 8 + fc + e];
                    }
                }
                NN_UNROLL
                for (uint i = 0; i < RM; i++) {
                    NN_UNROLL
                    for (uint j = 0; j < RN; j++) {
                        simdgroup_multiply_accumulate(partial[i][j], a[i], b[j], partial[i][j]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        NN_UNROLL
        for (uint i = 0; i < RM; i++) {
            NN_UNROLL
            for (uint j = 0; j < RN; j++) {
                NN_UNROLL
                for (uint e = 0; e < 2; e++) {
                    accum[i][j][e] = partial[i][j].thread_elements()[e];
                }
            }
        }
        // Occupied low bits select completed subtrees of equal size.
        uint index = base / 64;
        NN_UNROLL
        for (uint level = 0; level < LEVELS; level++) {
            if (index & (1u << level)) {
                NN_UNROLL
                for (uint i = 0; i < RM; i++) {
                    NN_UNROLL
                    for (uint j = 0; j < RN; j++) {
                        NN_UNROLL
                        for (uint e = 0; e < 2; e++) {
                            accum[i][j][e] = bins[level][i][j][e] + accum[i][j][e];
                        }
                    }
                }
            } else {
                NN_UNROLL
                for (uint i = 0; i < RM; i++) {
                    NN_UNROLL
                    for (uint j = 0; j < RN; j++) {
                        NN_UNROLL
                        for (uint e = 0; e < 2; e++) {
                            bins[level][i][j][e] = accum[i][j][e];
                        }
                    }
                }
                break;
            }
        }
    }
    bool first = true;
    uint leaves = (K + 63) / 64;
    NN_UNROLL
    for (int level = LEVELS - 1; level >= 0; level--) {
        if (leaves & (1u << level)) {
            NN_UNROLL
            for (uint i = 0; i < RM; i++) {
                NN_UNROLL
                for (uint j = 0; j < RN; j++) {
                    NN_UNROLL
                    for (uint e = 0; e < 2; e++) {
                        accum[i][j][e] = first ? bins[level][i][j][e]
                            : accum[i][j][e] + bins[level][i][j][e];
                    }
                }
            }
            first = false;
        }
    }
    uint r = row + (sg / 2) * RM * 8 + fr;
    uint c = col + (sg % 2) * RN * 8 + fc;
    NN_UNROLL
    for (uint i = 0; i < RM; i++) {
        NN_UNROLL
        for (uint j = 0; j < RN; j++) {
            NN_UNROLL
            for (uint e = 0; e < 2; e++) {
                if (r + i * 8 < M && c + j * 8 + e < N) {
                    batch_out[ulong(r + i * 8) * N + c + j * 8 + e] = accum[i][j][e];
                }
            }
        }
    }
);
/* Two fixed K partitions use the output buffer for partial sums. Each lane
 * owns its output elements, so no partial matrix allocation is required. */
static const char *split_source = METAL(
    constexpr uint BM = 32, BN = 64, RM = 4, RN = 4, BK = 16;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint fr = ((lane / 4) & 4) + ((lane / 2) % 4);
    uint fc = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    uint M = x_shape[0]; constexpr uint K = INNER, N = COLS;
    uint row = threadgroup_position_in_grid.y * BM;
    uint col = threadgroup_position_in_grid.x * BN;
    alignas(16) threadgroup float A[BM * (BK + 4)];
    alignas(16) threadgroup float B[BK * (BN + 4)];
    simdgroup_float8x8 partial[RM][RN];
    uint r = row + fr;
    uint c = col + (sg % 2) * 8 + fc;
    volatile device float *result = out;
    uint chunk = ((K + 31) / 32) * 16;
    for (uint base = 0; base < K; base += chunk) {
        NN_UNROLL
        for (uint i = 0; i < RM; i++) {
            NN_UNROLL
            for (uint j = 0; j < RN; j++) {
                partial[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0);
            }
        }
        for (uint tile = 0; tile < chunk; tile += BK) {
            if (VECTOR) {
                NN_UNROLL
                for (uint i = lid * 8; i < BM * BK; i += 64 * 8) {
                    uint ar = row + i / BK, ak = base + tile + i % BK;
                    threadgroup Copy8 *dst = reinterpret_cast<threadgroup Copy8*>(A + (i / BK) * (BK + 4) + i % BK);
                    if (ar < M) *dst = load8(x + ulong(ar) * K + ak);
                    else { NN_UNROLL for (uint j=0;j<8;j++) dst->values[j]=0; }
                }
                NN_UNROLL
                for (uint i = lid * 8; i < BK * BN; i += 64 * 8) {
                    uint bk = base + tile + i / BN, bc = col + i % BN;
                    threadgroup Copy8 *dst = reinterpret_cast<threadgroup Copy8*>(B + (i / BN) * (BN + 4) + i % BN);
                    if (bc + 8 <= N) *dst = load8(w + ulong(bk)*N+bc);
                    else { NN_UNROLL for (uint j=0;j<8;j++) dst->values[j]=bc+j<N ? w[ulong(bk)*N+bc+j] : 0; }
                }
            } else {
                for (uint i = lid; i < BM * BK; i += 64) {
                    uint r = row + i / BK, k = base + tile + i % BK;
                    A[(i / BK) * (BK + 4) + i % BK] = r < M && k < K ? x[ulong(r) * K + k] : 0;
                }
                for (uint i = lid; i < BK * BN; i += 64) {
                    uint k = i / BN, n = i % BN;
                    B[k * (BN + 4) + n] = base + tile + k < K && col + n < N
                        ? w[ulong(base + tile + k) * N + col + n] : 0;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            NN_UNROLL
            for (uint k = 0; k < BK; k += 8) {
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_float8x8 a[RM], b[RN];
                NN_UNROLL
                for (uint i = 0; i < RM; i++) {
                    a[i].thread_elements()[0] = A[(i * 8 + fr) * (BK + 4) + k + fc];
                    a[i].thread_elements()[1] = A[(i * 8 + fr) * (BK + 4) + k + fc + 1];
                }
                simdgroup_barrier(mem_flags::mem_none);
                NN_UNROLL
                for (uint j = 0; j < RN; j++) {
                    b[j].thread_elements()[0] = B[(k + fr) * (BN + 4) + ((sg % 2) + 2 * j) * 8 + fc];
                    b[j].thread_elements()[1] = B[(k + fr) * (BN + 4) + ((sg % 2) + 2 * j) * 8 + fc + 1];
                }
                simdgroup_barrier(mem_flags::mem_none);
                NN_UNROLL
                for (uint i = 0; i < RM; i++) {
                    NN_UNROLL
                    for (uint j = 0; j < RN; j++) {
                        uint t = (i % 2) ? RN - 1 - j : j;
                        simdgroup_multiply_accumulate(partial[i][t], a[i], b[t], partial[i][t]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        NN_UNROLL
        for (uint i = 0; i < RM; i++) {
            NN_UNROLL
            for (uint j = 0; j < RN; j++) {
                NN_UNROLL
                for (uint e = 0; e < 2; e++) {
                    if (r + i * 8 < M && c + j * 16 + e < N) {
                        ulong pos = ulong(r + i * 8) * N + c + j * 16 + e;
                        float value = partial[i][j].thread_elements()[e];
                        result[pos] = base == 0 ? value : result[pos] + value;
                    }
                }
            }
        }
    }

);
// clang-format on
#undef METAL

struct nn_mlx_matmul {
    mlx_fast_metal_kernel kernel;
    mlx_fast_metal_kernel split;
};

nn_mlx_matmul_t *nn_mlx_matmul_new(void) {
    nn_mlx_matmul_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return NULL;
    const char *names[] = {"x", "w"}, *results[] = {"out"};
    mlx_vector_string in = mlx_vector_string_new_data(names, 2);
    mlx_vector_string out = mlx_vector_string_new_data(results, 1);
    const char *header = "#include <metal_simdgroup_matrix>\n"
                         "#define NN_UNROLL _Pragma(\"clang loop unroll(full)\")\n";
    ctx->kernel =
        mlx_fast_metal_kernel_new("nn_matmul", in, out, matmul_source, header, true, false);
    mlx_vector_string_free(in);
    mlx_vector_string_free(out);
    if (!ctx->kernel.ctx) {
        nn_mlx_matmul_free(ctx);
        return NULL;
    }
    return ctx;
}

void nn_mlx_matmul_free(nn_mlx_matmul_t *ctx) {
    if (!ctx)
        return;
    mlx_fast_metal_kernel_free(ctx->kernel);
    mlx_fast_metal_kernel_free(ctx->split);
    free(ctx);
}

int nn_mlx_matmul(
    nn_mlx_matmul_t *ctx, mlx_array *out, mlx_array x, mlx_array w, mlx_stream stream) {
    if (!ctx || !out || !x.ctx || !w.ctx || (mlx_array_ndim(w) != 2 && mlx_array_ndim(w) != 3) ||
        mlx_array_dtype(x) != MLX_FLOAT32 || mlx_array_dtype(w) != MLX_FLOAT32)
        return -1;
    int rank = (int)mlx_array_ndim(x), result[3];
    if (rank < 2 || rank > 3)
        return -1;
    for (int i = 0; i < rank; i++) {
        result[i] = mlx_array_dim(x, i);
        if (result[i] <= 0)
            return -1;
    }
    int wrank = (int)mlx_array_ndim(w);
    int batch = wrank == 3 ? mlx_array_dim(w, 0) : 1;
    if (batch <= 0 || (wrank == 3 && (rank != 3 || batch != result[0])))
        return -1;
    int k = result[rank - 1], n = mlx_array_dim(w, wrank - 1);
    size_t total_rows = mlx_array_size(x) / k, rows = total_rows / batch;
    if (mlx_array_dim(w, wrank - 2) != k || n <= 0 || total_rows > INT_MAX || batch > INT_MAX / k ||
        (n - 1) / 32 + 1 > INT_MAX / 128)
        return -1;
    int shape[] = {(int)total_rows, k};
    result[rank - 1] = n;
    mlx_array flat = mlx_array_new(), weight = mlx_array_new(), temp = mlx_array_new();
    int rc = mlx_reshape(&flat, x, shape, 2, stream) ||
             mlx_reshape(&weight, w, (int[]){batch * k, n}, 2, stream);
    if (rc)
        goto done;
    int levels = 0;
    for (unsigned leaves = (k - 1) / 64 + 1; leaves; leaves >>= 1)
        levels++;
    shape[1] = n;
    /* MLX copies strided views before launch. Normal backend weights and
     * activations are already row-contiguous. Scalar component reads also
     * allow unaligned offsets in contiguous slice views. */
    int vector = k % 64 == 0 && n % 4 == 0;
    mlx_array args[] = {flat, weight};
    mlx_vector_array inputs = mlx_vector_array_new_data(args, 2), outputs = mlx_vector_array_new();
    mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
    rc = mlx_fast_metal_kernel_config_add_template_arg_int(config, "BATCHES", batch) ||
         mlx_fast_metal_kernel_config_add_template_arg_bool(config, "VECTOR", vector) ||
         mlx_fast_metal_kernel_config_add_template_arg_int(config, "LEVELS", levels) ||
         mlx_fast_metal_kernel_config_add_output_arg(config, shape, 2, MLX_FLOAT32) ||
         mlx_fast_metal_kernel_config_set_grid(config, ((n - 1) / 32 + 1) * 128,
                                               ((int)rows - 1) / 32 + 1, batch) ||
         mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1) ||
         mlx_fast_metal_kernel_apply(&outputs, ctx->kernel, inputs, config, stream) ||
         mlx_vector_array_get(&temp, outputs, 0) || mlx_reshape(out, temp, result, rank, stream);
    mlx_vector_array_free(inputs);
    mlx_vector_array_free(outputs);
    mlx_fast_metal_kernel_config_free(config);
done:
    mlx_array_free(flat);
    mlx_array_free(weight);
    mlx_array_free(temp);
    return rc ? -1 : 0;
}

int nn_mlx_matmul_split(
    nn_mlx_matmul_t *ctx, mlx_array *out, mlx_array x, mlx_array w, mlx_stream stream) {
    if (!ctx || !out || !x.ctx || !w.ctx || mlx_array_ndim(w) != 2 ||
        mlx_array_dtype(x) != MLX_FLOAT32 || mlx_array_dtype(w) != MLX_FLOAT32)
        return -1;
    int rank = (int)mlx_array_ndim(x), result[3];
    if (rank < 2 || rank > 3)
        return -1;
    for (int i = 0; i < rank; i++) {
        result[i] = mlx_array_dim(x, i);
        if (result[i] <= 0)
            return -1;
    }
    int k = result[rank - 1], n = mlx_array_dim(w, 1);
    size_t rows = mlx_array_size(x) / k;
    if (mlx_array_dim(w, 0) != k || n <= 0 || rows > INT_MAX || (n - 1) / 64 + 1 > INT_MAX / 64)
        return -1;
    if (!ctx->split.ctx) {
        const char *names[] = {"x", "w"}, *results[] = {"out"};
        mlx_vector_string in = mlx_vector_string_new_data(names, 2);
        mlx_vector_string outputs = mlx_vector_string_new_data(results, 1);
        const char *header = "#include <metal_simdgroup_matrix>\n"
                             "#define NN_UNROLL _Pragma(\"clang loop unroll(full)\")\n"
                             "struct Copy8 { float values[8]; };\n"
                             "inline Copy8 load8(const device float *p) { return "
                             "*reinterpret_cast<const device Copy8*>(p); }\n"
                             "inline Copy8 load8(const constant float *p) { return "
                             "*reinterpret_cast<const constant Copy8*>(p); }\n";
        ctx->split = mlx_fast_metal_kernel_new("nn_matmul_split", in, outputs, split_source,
                                               header, true, false);
        mlx_vector_string_free(in);
        mlx_vector_string_free(outputs);
        if (!ctx->split.ctx)
            return -1;
    }
    int shape[] = {(int)rows, k};
    result[rank - 1] = n;
    mlx_array flat = mlx_array_new(), temp = mlx_array_new();
    int rc = mlx_reshape(&flat, x, shape, 2, stream);
    if (rc)
        goto done;
    shape[1] = n;
    /* Copy8 has float alignment, including slice views with unaligned offsets. */
    int vector = k % 64 == 0 && n % 4 == 0;
    mlx_array args[] = {flat, w};
    mlx_vector_array inputs = mlx_vector_array_new_data(args, 2), outputs = mlx_vector_array_new();
    mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
    rc = mlx_fast_metal_kernel_config_add_template_arg_int(config, "INNER", k) ||
         mlx_fast_metal_kernel_config_add_template_arg_int(config, "COLS", n) ||
         mlx_fast_metal_kernel_config_add_template_arg_bool(config, "VECTOR", vector) ||
         mlx_fast_metal_kernel_config_add_output_arg(config, shape, 2, MLX_FLOAT32) ||
         mlx_fast_metal_kernel_config_set_grid(config, ((n - 1) / 64 + 1) * 64,
                                               ((int)rows - 1) / 32 + 1, 1) ||
         mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1) ||
         mlx_fast_metal_kernel_apply(&outputs, ctx->split, inputs, config, stream) ||
         mlx_vector_array_get(&temp, outputs, 0) || mlx_reshape(out, temp, result, rank, stream);
    mlx_vector_array_free(inputs);
    mlx_vector_array_free(outputs);
    mlx_fast_metal_kernel_config_free(config);
done:
    mlx_array_free(flat);
    mlx_array_free(temp);
    return rc ? -1 : 0;
}
