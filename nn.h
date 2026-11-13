#ifndef NN_H
#define NN_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__GNUC__)
#    define NN_API __attribute__((visibility("default")))
#else
#    define NN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Transformer inference kernels (CPU, float32).
 *
 * Low-level math operations. All operate on float32 tensors in row-major order.
 */

/* Basic operations */

NN_API void nn_cpu_add_inplace(float *a, const float *b, int n);

/* Matrix operations */

/* C = A @ B^T: A[M,K], B[N,K], C[M,N] */
NN_API void nn_cpu_matmul_t(float *C, const float *A, const float *B, int M, int K, int N);

/* y = x @ W^T + b: x[seq,in], W[out,in], b[out], y[seq,out] */
NN_API void nn_cpu_linear(
    float *y, const float *x, const float *W, const float *b, int seq_len, int in_dim, int out_dim);

NN_API void nn_cpu_linear_nobias(
    float *y, const float *x, const float *W, int seq_len, int in_dim, int out_dim);

/* bf16 weight variants */

NN_API void nn_cpu_bf16_to_f32_buf(float *dst, const uint16_t *src, size_t n);
NN_API void nn_cpu_linear_nobias_bf16(
    float *y, const float *x, const uint16_t *W_bf16, int seq_len, int in_dim, int out_dim);

/* Small-sequence BF16 path: compute two projections with one threaded dispatch. */
NN_API void nn_cpu_linear_nobias_bf16_pair(float *a,
                                                   float *b,
                                                   const float *x,
                                                   const uint16_t *Wa_bf16,
                                                   const uint16_t *Wb_bf16,
                                                   int seq_len,
                                                   int in_dim,
                                                   int a_dim,
                                                   int b_dim);

/* Small-sequence BF16 path: compute Q/K/V matvecs with one threaded dispatch. */
NN_API void nn_cpu_linear_nobias_bf16_qkv(float *q,
                                                  float *k,
                                                  float *v,
                                                  const float *x,
                                                  const uint16_t *Wq_bf16,
                                                  const uint16_t *Wk_bf16,
                                                  const uint16_t *Wv_bf16,
                                                  int seq_len,
                                                  int in_dim,
                                                  int q_dim,
                                                  int kv_dim);

/* Dot product using the best available local SIMD implementation. */
NN_API float nn_cpu_dot_f32(const float *a, const float *b, int n);

/* Normalization */

/* RMS Normalization: out = x / rms(x) * weight */
NN_API void nn_cpu_rms_norm(
    float *out, const float *x, const float *weight, int seq_len, int hidden, float eps);

/* Per-head RMS Normalization for Q/K norms in decoder
 * x: [seq, n_heads, head_dim], weight: [head_dim]
 * Normalizes each head independently */
NN_API void nn_cpu_rms_norm_per_head(
    float *x, const float *weight, int seq_len, int n_heads, int head_dim, float eps);

/* Layer normalization for BERT-family models:
 * out = gamma * (x - mean) / sqrt(var + eps) + beta.
 * Mean and biased/population variance are computed over the hidden axis per row.
 * Unlike RMSNorm, this subtracts the mean and adds a bias. */
NN_API void nn_cpu_layer_norm(float *out,
                                      const float *x,
                                      const float *gamma,
                                      const float *beta,
                                      int seq_len,
                                      int hidden,
                                      float eps);

/* Activation functions */

NN_API void nn_cpu_softmax(float *x, int rows, int cols);
/* gate = SiLU(gate) * up */
NN_API void nn_cpu_silu_mul_inplace(float *gate, const float *up, int n);

/* Exact (erf) GeLU in place: x = 0.5 * x * (1 + erf(x / sqrt(2))).
 * This is the "gelu" activation HF BERT/BGE/MiniLM use, not the tanh approximation. */
NN_API void nn_cpu_gelu_inplace(float *x, int n);

/* Tanh-approximation GeLU in place (HF gelu_new / gelu_pytorch_tanh):
 * x = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))). */
NN_API void nn_cpu_gelu_tanh_inplace(float *x, int n);

/* Attention operations */

/*
 * Bidirectional GQA attention over a packed/ragged batch.
 *
 * Q: [total_seq, n_heads * head_dim]
 * K,V: [total_seq, n_kv_heads * head_dim]
 * offsets: [batch + 1], token spans for each sequence.
 *
 * Attention is block-diagonal: tokens only attend within their own
 * offsets[b]..offsets[b+1] span.
 */
NN_API void nn_cpu_bidirectional_gqa_attention_packed(float *out,
                                                              const float *Q,
                                                              const float *K,
                                                              const float *V,
                                                              const int *offsets,
                                                              int batch,
                                                              int n_heads,
                                                              int n_kv_heads,
                                                              int head_dim,
                                                              float scale);

/* Optional reusable scratch for the BLAS-tiled packed attention path. */
NN_API size_t nn_cpu_bidirectional_gqa_attention_packed_scratch_bytes(const int *offsets,
                                                                              int batch);
NN_API void nn_cpu_bidirectional_gqa_attention_packed_with_scratch(float *out,
                                                                           const float *Q,
                                                                           const float *K,
                                                                           const float *V,
                                                                           const int *offsets,
                                                                           int batch,
                                                                           int n_heads,
                                                                           int n_kv_heads,
                                                                           int head_dim,
                                                                           float scale,
                                                                           float *scratch,
                                                                           size_t scratch_bytes);

/*
 * Causal GQA attention over a packed/ragged batch. Query position i attends
 * only to positions at or before i within its sequence.
 */
NN_API void nn_cpu_causal_gqa_attention_packed(float *out,
                                                       const float *Q,
                                                       const float *K,
                                                       const float *V,
                                                       const int *offsets,
                                                       int batch,
                                                       int n_heads,
                                                       int n_kv_heads,
                                                       int head_dim,
                                                       float scale);
NN_API void nn_cpu_causal_gqa_attention_packed_with_scratch(float *out,
                                                                    const float *Q,
                                                                    const float *K,
                                                                    const float *V,
                                                                    const int *offsets,
                                                                    int batch,
                                                                    int n_heads,
                                                                    int n_kv_heads,
                                                                    int head_dim,
                                                                    float scale,
                                                                    float *scratch,
                                                                    size_t scratch_bytes);

/* Position embeddings */

/*
 * NeoX-style RoPE: compute cos/sin for positions.
 * cos_out, sin_out: [seq, head_dim]
 * cos[d] and cos[half+d] are the same (duplicated for full head_dim).
 */
NN_API void nn_cpu_compute_rope_neox(
    float *cos_out, float *sin_out, const int *positions, int seq, int head_dim, float theta);

/*
 * Apply NeoX-style RoPE to Q or K.
 * x: [seq, n_heads * head_dim] (in-place)
 * cos_vals, sin_vals: [seq, head_dim]
 */
NN_API void nn_cpu_apply_rope_neox(
    float *x, const float *cos_vals, const float *sin_vals, int seq, int n_heads, int head_dim);

NN_API int nn_cpu_l2_normalize(float *vec, int dim);

/* Host F32 cosine similarity. Returns zero for invalid arguments, zero norms
 * or non-finite accumulated values. Does not modify or normalize the inputs. */
NN_API float nn_cpu_cosine_similarity(const float *a, const float *b, int dim);

/* Sum each query row's largest dot product with a document row. Inputs are contiguous F32 rows. */
NN_API float nn_cpu_maxsim(
    const float *query, int query_tokens, const float *document, int document_tokens, int dim);
/* offsets[docs + 1] partitions packed document rows, starts at zero, and increases strictly. */
NN_API int nn_cpu_maxsim_batch(const float *query,
                                       int query_tokens,
                                       const float *documents,
                                       const int *offsets,
                                       int docs,
                                       int dim,
                                       float *scores);

/* BF16 is the upper 16 bits of an IEEE-754 binary32, so widening one value is a
 * left shift reinterpreted as a float. Embedding lookup uses this scalar helper;
 * nn_cpu_bf16_to_f32_buf handles hot, contiguous SIMD conversions. */
static inline float nn_cpu_bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* IEEE-754 binary16 -> binary32. Unlike BF16, F16 has a 5-bit exponent and a
 * 10-bit mantissa, so widening is a field re-pack, not a shift: rebias the
 * exponent (15 -> 127), shift the mantissa up by 13, and special-case zeros,
 * subnormals, and inf/NaN. Used by the loaders to accept F16 safetensors
 * snapshots (e.g. the public multilingual-e5 weights); there is no fused F16
 * kernel, so weights are widened to F32 once at load. */
static inline float nn_cpu_f16_to_f32(uint16_t f16) {
    uint32_t sign = (uint32_t)(f16 & 0x8000u) << 16;
    uint32_t exp = (f16 >> 10) & 0x1fu;
    uint32_t mant = f16 & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign; /* signed zero */
        } else {
            /* Subnormal half: renormalize into a normal float. Shift the
             * mantissa left until its leading 1 reaches the hidden-bit
             * position, decrementing the exponent for each shift. */
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13); /* inf or NaN */
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Configure the process-wide pool before execution. Calls using the pool must be serialized. */
NN_API void nn_cpu_set_threads(int n);
NN_API int nn_cpu_count(void);
NN_API int nn_cpu_num_threads(void);

/* Pool packed rows with final RMS normalization. Offsets must describe nonempty sequences. */
NN_API int nn_cpu_pool_rms(float *out,
                                   const float *states,
                                   const float *weight,
                                   const int *offsets,
                                   int batch,
                                   int hidden,
                                   float eps,
                                   int last_token,
                                   int normalize);

#ifdef __cplusplus
}
#endif

#endif
