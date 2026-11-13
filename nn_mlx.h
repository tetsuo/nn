#ifndef NN_MLX_H
#define NN_MLX_H

#include "nn.h"
#include <mlx/c/mlx.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    mlx_array w, wt;
} nn_mlx_linear_t;

/* Start with an owned [out,in] weight in w and zero other fields. Preparation
 * materializes [in,out] storage and releases w.
 * Apply borrows x and returns an owned array. Use and free on one thread. */
NN_API void nn_mlx_linear_free(nn_mlx_linear_t *);
NN_API int nn_mlx_linear_prepare(nn_mlx_linear_t *, mlx_stream);
NN_API mlx_array nn_mlx_linear(mlx_array x, const nn_mlx_linear_t *, mlx_stream);
/* Replace an owned array with its F32 cast. */
NN_API void nn_mlx_upcast_f32(mlx_array *, mlx_stream);

/* F32 normalization kernels. Create/use/free on the inference thread.
 * x/out: rank 1-4 with normalization on the last axis; weight: [width].
 * Weight contains the multiplier directly. Inputs remain immutable. */
typedef struct nn_mlx_norm nn_mlx_norm_t;

NN_API nn_mlx_norm_t *nn_mlx_norm_new(void);
NN_API void nn_mlx_norm_free(nn_mlx_norm_t *ctx);
NN_API int nn_mlx_rms_norm(nn_mlx_norm_t *ctx,
                                   mlx_array *out,
                                   mlx_array x,
                                   mlx_array weight,
                                   float eps,
                                   mlx_stream stream);

/* F32 LayerNorm over the last axis, including learned gain and bias. */
NN_API int nn_mlx_layer_norm(nn_mlx_norm_t *ctx,
                                     mlx_array *out,
                                     mlx_array x,
                                     mlx_array weight,
                                     mlx_array bias,
                                     float eps,
                                     mlx_stream stream);

/* F32 matrix product with balanced accumulation. x: [M,K] or [B,M,K].
 * w: [K,N] or [B,K,N] with a matching rank-3 x. out: [...,N].
 * Inputs remain immutable. Create/use/free on the inference thread. */
typedef struct nn_mlx_matmul nn_mlx_matmul_t;
NN_API nn_mlx_matmul_t *nn_mlx_matmul_new(void);
NN_API void nn_mlx_matmul_free(nn_mlx_matmul_t *ctx);
NN_API int nn_mlx_matmul(
    nn_mlx_matmul_t *ctx, mlx_array *out, mlx_array x, mlx_array w, mlx_stream stream);

/* F32 product with up to two K partitions, independent of the output row count.
 * x: [M,K] or [B,M,K]; w: [K,N]. Ownership matches nn_mlx_matmul. */
NN_API int nn_mlx_matmul_split(
    nn_mlx_matmul_t *ctx, mlx_array *out, mlx_array x, mlx_array w, mlx_stream stream);

/* Uses split accumulation for K >= N and more than 64 rows; native kernels
 * handle small row counts and expanding projections. Weights remain borrowed. */
NN_API mlx_array nn_mlx_linear_split(nn_mlx_matmul_t *ctx,
                                                 mlx_array x,
                                                 const nn_mlx_linear_t *weight,
                                                 mlx_stream stream);

/* F32 Qwen3.5 recurrent primitives. Create/use/free on the model's
 * inference thread. Calls build lazy graphs on the selected stream and never
 * mutate an input array. The caller owns all returned arrays.
 *
 * lengths[B] describes right-padded sequences, including zero-length rows.
 * Padding emits zero and leaves each sequence's state/window unchanged.
 * Start each independent document with zero state and zero convolution history. */
typedef struct nn_mlx_gdn nn_mlx_gdn_t;

NN_API nn_mlx_gdn_t *nn_mlx_gdn_new(void);
NN_API void nn_mlx_gdn_free(nn_mlx_gdn_t *ctx);

/* x/out: [B,T,C]; weight: [C,4] in checkpoint order;
 * window_in/window_out: [B,3,C], oldest first. Applies convolution then SiLU. */
NN_API int nn_mlx_gdn_conv(nn_mlx_gdn_t *ctx,
                                   mlx_array *out,
                                   mlx_array *window_out,
                                   mlx_array x,
                                   mlx_array weight,
                                   mlx_array window_in,
                                   const int *lengths,
                                   mlx_stream stream);

/* qkv: [B,T,(2*key_heads+value_heads)*128]; a/b: [B,T,value_heads];
 * a_log/dt_bias: [value_heads]; state_in/state_out: [B,value_heads,128,128]
 * in key/value order; out: [B,T,value_heads,128]. Q/K are normalized internally
 * with rsqrt(sum(x*x)+1e-6), and Q is scaled by 1/sqrt(128).
 * value_heads must be a positive multiple of key_heads. */
NN_API int nn_mlx_gdn_scan(nn_mlx_gdn_t *ctx,
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
                                   mlx_stream stream);

/* x/z/out: [B,T,value_heads,128]; weight: [128]. Direct-weight RMSNorm
 * followed by SiLU(z), unlike Qwen3.5's other norms which use 1+weight. */
NN_API int nn_mlx_gdn_norm(nn_mlx_gdn_t *ctx,
                                   mlx_array *out,
                                   mlx_array x,
                                   mlx_array z,
                                   mlx_array weight,
                                   float eps,
                                   mlx_stream stream);

#define NN_MLX_MAX_PROBABILITY_BYTES ((size_t)4 * 1024 * 1024 * 1024)

/* F32 attention with rotary and logit rounding residuals retained
 * until softmax subtraction. qkv:[N,3*width], coefficients:[N,head_dim].
 * Bounds the nominal [heads,N,N] F32 probability workspace before use.
 * Large inputs evaluate one complete head at a time to reuse score scratch. */
typedef struct nn_mlx_rotary_attention nn_mlx_rotary_attention_t;
NN_API nn_mlx_rotary_attention_t *nn_mlx_rotary_attention_new(void);
NN_API void nn_mlx_rotary_attention_free(nn_mlx_rotary_attention_t *ctx);
/* QKV needs one final rounding of the complete dot plus bias: its large
 * components make the following attention sensitive to accumulation error. */
NN_API int nn_mlx_rotary_attention_project(nn_mlx_rotary_attention_t *ctx,
                                                   mlx_array *out,
                                                   mlx_array x,
                                                   mlx_array weight,
                                                   mlx_array bias,
                                                   mlx_stream stream);
NN_API int nn_mlx_rotary_attention(nn_mlx_rotary_attention_t *ctx,
                                           mlx_array *out,
                                           mlx_array qkv,
                                           mlx_array cosine,
                                           mlx_array sine,
                                           mlx_array cos_low,
                                           mlx_array sin_low,
                                           int heads,
                                           mlx_stream stream);

/* F32 noncausal attention with balanced matrix products: q [B,H,Q,D], k [B,HK,K,D], v [B,HK,K,DV].
 * Optional additive key mask: [B or 1,1,1,K]. H must be a multiple of HK.
 * score_bytes bounds one score matrix. Query blocks retain all keys and values.
 * Blocks are evaluated serially. The matrix context and inputs are borrowed. The caller owns out. */
NN_API int nn_mlx_attention(nn_mlx_matmul_t *ctx,
                                        mlx_array *out,
                                        mlx_array q,
                                        mlx_array k,
                                        mlx_array v,
                                        mlx_array mask,
                                        float scale,
                                        size_t score_bytes,
                                        mlx_stream stream);

/* Borrow x and return an owned array. */
NN_API mlx_array nn_mlx_gelu_erf(mlx_array x, mlx_stream);
NN_API mlx_array nn_mlx_gelu_tanh(mlx_array x, mlx_stream);

#ifdef __cplusplus
}
#endif
#endif
