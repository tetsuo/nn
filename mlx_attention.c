#include "nn_mlx.h"
#include "mlx_reduce.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static int attention_block(nn_mlx_matmul_t *ctx,
                           mlx_array *out,
                           mlx_array q,
                           mlx_array k,
                           mlx_array v,
                           mlx_array mask,
                           float scale,
                           mlx_stream stream) {
    int batch = mlx_array_dim(q, 0), heads = mlx_array_dim(q, 1);
    int rows = mlx_array_dim(q, 2), width = mlx_array_dim(q, 3);
    int kv_heads = mlx_array_dim(k, 1), keys = mlx_array_dim(k, 2);
    int value_width = mlx_array_dim(v, 3), groups = heads / kv_heads;
    if (batch > INT_MAX / kv_heads || rows > INT_MAX / groups)
        return -1;
    /* Query heads in one group share K/V. Each matrix covers one batch and KV head. */
    int matrices = batch * kv_heads, query_rows = groups * rows;
    mlx_array factor = mlx_array_new_float32(scale), scaled = mlx_array_new();
    mlx_array kt = mlx_array_new(), qf = mlx_array_new(), kf = mlx_array_new(), vf = mlx_array_new();
    mlx_array scores = mlx_array_new(), shaped = mlx_array_new(), masked = mlx_array_new();
    mlx_array probs = mlx_array_new(), pf = mlx_array_new(), product = mlx_array_new();
    int rc = mlx_multiply(&scaled, q, factor, stream) ||
             mlx_transpose_axes(&kt, k, (int[]){0, 1, 3, 2}, 4, stream) ||
             mlx_reshape(&qf, scaled, (int[]){matrices, query_rows, width}, 3, stream) ||
             mlx_reshape(&kf, kt, (int[]){matrices, width, keys}, 3, stream) ||
             mlx_reshape(&vf, v, (int[]){matrices, keys, value_width}, 3, stream) ||
             nn_mlx_matmul(ctx, &scores, qf, kf, stream) ||
             mlx_reshape(&shaped, scores, (int[]){batch, heads, rows, keys}, 4, stream);
    if (!rc)
        rc = mask.ctx ? mlx_add(&masked, shaped, mask, stream) : mlx_array_set(&masked, shaped);
    if (!rc)
        rc = mlx_softmax_axis(&probs, masked, -1, true, stream) ||
             mlx_reshape(&pf, probs, (int[]){matrices, query_rows, keys}, 3, stream) ||
             nn_mlx_matmul(ctx, &product, pf, vf, stream) ||
             mlx_reshape(out, product, (int[]){batch, heads, rows, value_width}, 4, stream);
    mlx_array_free(factor);
    mlx_array_free(scaled);
    mlx_array_free(kt);
    mlx_array_free(qf);
    mlx_array_free(kf);
    mlx_array_free(vf);
    mlx_array_free(scores);
    mlx_array_free(shaped);
    mlx_array_free(masked);
    mlx_array_free(probs);
    mlx_array_free(pf);
    mlx_array_free(product);
    return rc;
}

int nn_mlx_attention(nn_mlx_matmul_t *ctx,
                           mlx_array *out,
                           mlx_array q,
                           mlx_array k,
                           mlx_array v,
                           mlx_array mask,
                           float scale,
                           size_t score_bytes,
                           mlx_stream stream) {
    mlx_array arrays[] = {q, k, v};
    if (!ctx || !out || !isfinite(scale))
        return -1;
    for (int i = 0; i < 3; ++i) {
        if (!arrays[i].ctx || mlx_array_ndim(arrays[i]) != 4 ||
            mlx_array_dtype(arrays[i]) != MLX_FLOAT32)
            return -1;
        for (int d = 0; d < 4; ++d)
            if (mlx_array_dim(arrays[i], d) <= 0)
                return -1;
    }
    int batch = mlx_array_dim(q, 0), heads = mlx_array_dim(q, 1);
    int rows = mlx_array_dim(q, 2), width = mlx_array_dim(q, 3);
    int keys = mlx_array_dim(k, 2), kv_heads = mlx_array_dim(k, 1);
    if (mlx_array_dim(k, 0) != batch || mlx_array_dim(v, 0) != batch ||
        mlx_array_dim(k, 3) != width || mlx_array_dim(v, 1) != kv_heads ||
        mlx_array_dim(v, 2) != keys || heads % kv_heads)
        return -1;
    if (mask.ctx && (mlx_array_dtype(mask) != MLX_FLOAT32 || mlx_array_ndim(mask) != 4 ||
                     (mlx_array_dim(mask, 0) != 1 && mlx_array_dim(mask, 0) != batch) ||
                     mlx_array_dim(mask, 1) != 1 || mlx_array_dim(mask, 2) != 1 ||
                     mlx_array_dim(mask, 3) != keys))
        return -1;
    size_t row_bytes = sizeof(float);
    int factors[] = {batch, heads, keys};
    for (int i = 0; i < 3; ++i) {
        if ((size_t)factors[i] > SIZE_MAX / row_bytes)
            return -1;
        row_bytes *= (size_t)factors[i];
    }
    size_t capacity = score_bytes / row_bytes;
    if (!capacity)
        return -1;
    if (capacity >= (size_t)rows)
        return attention_block(ctx, out, q, k, v, mask, scale, stream) ? -1 : 0;

    int block = (int)capacity, rc = 0;
    mlx_vector_array parts = mlx_vector_array_new();
    if (!parts.ctx)
        return -1;
    for (int start = 0; start < rows && !rc;) {
        int stop = rows - start < block ? rows : start + block;
        mlx_array slice = mlx_array_new(), part = mlx_array_new();
        rc = mlx_slice(&slice, q, (int[]){0, 0, start, 0}, 4, (int[]){batch, heads, stop, width}, 4,
                       (int[]){1, 1, 1, 1}, 4, stream) ||
             attention_block(ctx, &part, slice, k, v, mask, scale, stream) || mlx_array_eval(part) ||
             mlx_vector_array_append_value(parts, part);
        mlx_array_free(slice);
        mlx_array_free(part);
        start = stop;
    }
    if (!rc)
        rc = mlx_concatenate_axis(out, parts, 2, stream);
    mlx_vector_array_free(parts);
    return rc ? -1 : 0;
}

#define METAL(...) #__VA_ARGS__
// clang-format off
static const char *projection_source = METAL(
    uint row=threadgroup_position_in_grid.y*(4*ROWS)+simdgroup_index_in_threadgroup*ROWS;
    uint col=threadgroup_position_in_grid.x*32+thread_index_in_simdgroup;
    uint M=x_shape[0],K=x_shape[1],N=weight_shape[1];
    if(row>=M || col>=N)return;
    // Independent complete dots reuse each weight without reordering terms.
    fp32_pair acc[ROWS]={};
    for(uint k=0;k<K;k++) {
        float b=weight[ulong(k)*N+col];
        _Pragma("clang loop unroll(full)")
        for(uint r=0;r<ROWS;r++) {
            float a=row+r<M?x[ulong(row+r)*K+k]:0.0f;
            acc[r]=pair_dot_step(acc[r],a,b,0.0f);
        }
    }
    _Pragma("clang loop unroll(full)")
    for(uint r=0;r<ROWS;r++) {
        fp32_pair value=pair_add(acc[r],{bias[col],0.0f});
        if(row+r<M)projection[ulong(row+r)*N+col]=value.hi+value.lo;
    }
);
static const char *rotary_source = METAL(
    uint at = thread_position_in_grid.x;
    uint N = qkv_shape[0], W = qkv_shape[1]/3, D = cosine_shape[1];
    if (at >= N*W) return;
    uint row = at/W, col = at%W, dim = col%D;
    uint other = dim<D/2 ? col+D/2 : col-D/2;
    float sign = dim<D/2 ? -1.0f : 1.0f;
    uint pos = row*D+dim;
    fp32_pair c = {cosine[pos], cos_low[pos]}, s = {sine[pos], sin_low[pos]};
    fp32_pair q = pair_add(pair_scale(c,qkv[row*3*W+col]), pair_scale(s,sign*qkv[row*3*W+other]));
    fp32_pair k = pair_add(pair_scale(c,qkv[row*3*W+W+col]), pair_scale(s,sign*qkv[row*3*W+W+other]));
    queries[2*at]=q.hi; queries[2*at+1]=q.lo;
    // Store [width,N,2] so probability lanes read consecutive keys.
    ulong key_at=(ulong(col)*N+row)*2;
    keys[key_at]=k.hi; keys[key_at+1]=k.lo;
);

static const char *logit_source = METAL(
    uint tid=thread_position_in_threadgroup.x, lane=thread_index_in_simdgroup;
    uint row=threadgroup_position_in_grid.y*32, key=threadgroup_position_in_grid.x*32;
    uint N=queries_shape[0], W=queries_shape[1], head=head_index[0];
    // A 32x32 output tile reuses eight dimensions at a time in shared memory.
    // Each lane keeps eight query dots in their original dimension order.
    threadgroup fp32_pair a[32*8], b[8*32];
    fp32_pair dots[8];
    _Pragma("clang loop unroll(full)")
    for(uint r=0;r<8;r++)dots[r]={0.0f,0.0f};
    uint first=simdgroup_index_in_threadgroup*8;
    for(uint base=0;base<D;base+=8) {
        for(uint i=tid;i<32*8;i+=128) {
            uint r=i/8, d=base+i%8;
            ulong at=(ulong(row+r)*W+head*D+d)*2;
            a[i]=row+r<N && d<D ? fp32_pair{queries[at],queries[at+1]} : fp32_pair{0.0f,0.0f};
        }
        for(uint i=tid;i<8*32;i+=128) {
            uint d=base+i/32, c=i%32;
            ulong at=(ulong(head*D+d)*N+key+c)*2;
            b[i]=key+c<N && d<D ? fp32_pair{keys[at],keys[at+1]} : fp32_pair{0.0f,0.0f};
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("clang loop unroll(full)")
        for(uint d=0;d<8;d++) {
            if(base+d<D) {
                fp32_pair k=b[d*32+lane];
                _Pragma("clang loop unroll(full)")
                for(uint r=0;r<8;r++) {
                    fp32_pair q=a[(first+r)*8+d];
                    float low=fma(q.hi,k.lo,q.lo*k.hi);
                    dots[r]=pair_dot_step(dots[r],q.hi,k.hi,low);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    _Pragma("clang loop unroll(full)")
    for(uint r=0;r<8;r++) {
        if(row+first+r<N && key+lane<N) {
            fp32_pair dot=pair_add(dots[r],{0.0f,0.0f});
            ulong at=(ulong(row+first+r)*N+key+lane)*2;
            logits[at]=dot.hi;logits[at+1]=dot.lo;
        }
    }
);

static const char *probability_source = METAL(
    uint tid=thread_position_in_threadgroup.x, lane=thread_index_in_simdgroup;
    uint sg=simdgroup_index_in_threadgroup, row=threadgroup_position_in_grid.y;
    uint head=threadgroup_position_in_grid.z;
    uint N=queries_shape[0], W=queries_shape[1];
    fp32_pair scores[SLOTS];
    fp32_pair best={-INFINITY,0.0f};
    for(uint slot=0;slot<SLOTS;slot++) {
        uint key=tid+slot*128;
        fp32_pair dot={0.0f,0.0f};
        if(key<N) {
            if(TILED) {
                ulong at=(ulong(row)*N+key)*2;
                dot={keys[at],keys[at+1]};
            } else {
                for(uint d=0;d<D;d++) {
                    ulong qi=(ulong(row)*W+head*D+d)*2;
                    ulong ki=(ulong(head*D+d)*N+key)*2;
                    fp32_pair q={queries[qi],queries[qi+1]}, k={keys[ki],keys[ki+1]};
                    float low=fma(q.hi,k.lo,q.lo*k.hi);
                    dot=pair_dot_step(dot,q.hi,k.hi,low);
                }
                dot=pair_add(dot,{0.0f,0.0f});
            }
            if(dot.hi>best.hi || (dot.hi==best.hi && dot.lo>best.lo)) best=dot;
        }
        scores[slot]=dot;
    }
    threadgroup float partial_hi[4],partial_lo[4];
    float hi=simd_max(best.hi);
    float lo=simd_max(best.hi==hi?best.lo:-INFINITY);
    if(lane==0){partial_hi[sg]=hi;partial_lo[sg]=lo;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) {
        best={partial_hi[0],partial_lo[0]};
        for(uint j=1;j<4;j++) {
            fp32_pair next={partial_hi[j],partial_lo[j]};
            if(next.hi>best.hi || (next.hi==best.hi && next.lo>best.lo)) best=next;
        }
        partial_hi[0]=best.hi;partial_lo[0]=best.lo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    best={partial_hi[0],partial_lo[0]};
    // Copy the maximum in every group before scratch is reused for sums.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fp32_pair sum={0.0f,0.0f};
    for(uint slot=0;slot<SLOTS;slot++) {
        uint key=tid+slot*128;
        float p=0;
        if(key<N) {
            fp32_pair delta=pair_add(scores[slot],{-best.hi,-best.lo});
            p=metal::precise::exp((delta.hi+delta.lo)*metal::precise::rsqrt(float(D)));
            sum=pair_add(sum,{p,0.0f});
        }
        scores[slot]={p,0.0f};
    }
    sum=pair_reduce(sum,lane);
    if(lane==0){partial_hi[sg]=sum.hi;partial_lo[sg]=sum.lo;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) {
        sum={0.0f,0.0f};
        for(uint j=0;j<4;j++)sum=pair_add(sum,{partial_hi[j],partial_lo[j]});
        partial_hi[0]=sum.hi;partial_lo[0]=sum.lo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=partial_hi[0]+partial_lo[0];
    for(uint slot=0;slot<SLOTS;slot++) {
        uint key=tid+slot*128;
        if(key<N) probabilities[(ulong(head)*N+row)*N+key]=scores[slot].hi/total;
    }
);
// clang-format on
#undef METAL

struct nn_mlx_rotary_attention {
    mlx_fast_metal_kernel projection, rotary, logit, probability;
    nn_mlx_matmul_t *matmul;
};

static mlx_fast_metal_kernel kernel(
    const char *name, const char **inputs, int ni, const char **outputs, int no, const char *source) {
    mlx_vector_string in = mlx_vector_string_new_data(inputs, ni),
                      out = mlx_vector_string_new_data(outputs, no);
    mlx_fast_metal_kernel k =
        mlx_fast_metal_kernel_new(name, in, out, source, nn_mlx_reduce_source, true, false);
    mlx_vector_string_free(in);
    mlx_vector_string_free(out);
    return k;
}

nn_mlx_rotary_attention_t *nn_mlx_rotary_attention_new(void) {
    nn_mlx_rotary_attention_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return NULL;
    const char *inputs[] = {"qkv", "cosine", "sine", "cos_low", "sin_low"},
               *outputs[] = {"queries", "keys"};
    ctx->rotary = kernel("nn_rotary_rotary", inputs, 5, outputs, 2, rotary_source);
    const char *linputs[] = {"queries", "keys", "head_index"}, *loutputs[] = {"logits"};
    ctx->logit = kernel("nn_rotary_logits", linputs, 3, loutputs, 1, logit_source);
    const char *pinputs[] = {"queries", "keys"}, *poutputs[] = {"probabilities"};
    ctx->probability =
        kernel("nn_rotary_probabilities", pinputs, 2, poutputs, 1, probability_source);
    ctx->matmul = nn_mlx_matmul_new();
    const char *proj_in[] = {"x", "weight", "bias"}, *proj_out[] = {"projection"};
    ctx->projection = kernel("nn_rotary_projection", proj_in, 3, proj_out, 1, projection_source);
    if (!ctx->rotary.ctx || !ctx->logit.ctx || !ctx->probability.ctx || !ctx->projection.ctx ||
        !ctx->matmul) {
        nn_mlx_rotary_attention_free(ctx);
        return NULL;
    }
    return ctx;
}

void nn_mlx_rotary_attention_free(nn_mlx_rotary_attention_t *ctx) {
    if (!ctx)
        return;
    mlx_fast_metal_kernel_free(ctx->rotary);
    mlx_fast_metal_kernel_free(ctx->projection);
    mlx_fast_metal_kernel_free(ctx->logit);
    mlx_fast_metal_kernel_free(ctx->probability);
    nn_mlx_matmul_free(ctx->matmul);
    free(ctx);
}

int nn_mlx_rotary_attention_project(nn_mlx_rotary_attention_t *ctx,
                                        mlx_array *out,
                                        mlx_array x,
                                        mlx_array weight,
                                        mlx_array bias,
                                        mlx_stream stream) {
    if (!ctx || !out || !x.ctx || !weight.ctx || !bias.ctx || mlx_array_dtype(x) != MLX_FLOAT32 ||
        mlx_array_dtype(weight) != MLX_FLOAT32 || mlx_array_dtype(bias) != MLX_FLOAT32 ||
        mlx_array_ndim(x) != 2 || mlx_array_ndim(weight) != 2 || mlx_array_ndim(bias) != 1)
        return -1;
    int m = mlx_array_dim(x, 0), k = mlx_array_dim(x, 1), n = mlx_array_dim(weight, 1);
    if (m < 1 || m > 8192 || k < 1 || k > 8192 || n < 1 || n > 24576 ||
        mlx_array_dim(weight, 0) != k || mlx_array_dim(bias, 0) != n)
        return -1;
    /* Small inputs need more independent SIMD groups. Larger inputs can reuse weights across eight rows without reducing useful parallelism. */
    int rows = m >= 256 ? 8 : 4, tile_rows = 4 * rows;
    mlx_array args[] = {x, weight, bias};
    mlx_vector_array in = mlx_vector_array_new_data(args, 3), result = mlx_vector_array_new();
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int rc = mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", rows) ||
             mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){m, n}, 2, MLX_FLOAT32) ||
             mlx_fast_metal_kernel_config_set_grid(cfg, ((n + 31) / 32) * 128,
                                                   (m + tile_rows - 1) / tile_rows, 1) ||
             mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1) ||
             mlx_fast_metal_kernel_apply(&result, ctx->projection, in, cfg, stream) ||
             mlx_vector_array_get(out, result, 0);
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(result);
    return rc ? -1 : 0;
}

static int probabilities(nn_mlx_rotary_attention_t *ctx,
                         mlx_array *out,
                         mlx_array q,
                         mlx_array k,
                         int n,
                         int heads,
                         int d,
                         int tiled,
                         mlx_stream stream) {
    mlx_array args[] = {q, k};
    mlx_vector_array in = mlx_vector_array_new_data(args, 2), result = mlx_vector_array_new();
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    int rc = mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", d) ||
             mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "TILED", tiled) ||
             mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SLOTS", (n + 127) / 128) ||
             mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){heads, n, n}, 3, MLX_FLOAT32) ||
             mlx_fast_metal_kernel_config_set_grid(cfg, 128, n, heads) ||
             mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1) ||
             mlx_fast_metal_kernel_apply(&result, ctx->probability, in, cfg, stream) ||
             mlx_vector_array_get(out, result, 0);
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(result);
    return rc ? -1 : 0;
}

int nn_mlx_rotary_attention(nn_mlx_rotary_attention_t *ctx,
                                mlx_array *out,
                                mlx_array qkv,
                                mlx_array cosine,
                                mlx_array sine,
                                mlx_array cos_low,
                                mlx_array sin_low,
                                int heads,
                                mlx_stream stream) {
    if (!ctx || !out || !qkv.ctx || !cosine.ctx || !sine.ctx || !cos_low.ctx || !sin_low.ctx ||
        heads < 1 || heads > 128 || mlx_array_dtype(qkv) != MLX_FLOAT32 || mlx_array_ndim(qkv) != 2 ||
        mlx_array_ndim(cosine) != 2)
        return -1;
    int n = mlx_array_dim(qkv, 0), width = mlx_array_dim(qkv, 1) / 3, d = mlx_array_dim(cosine, 1);
    if (n < 1 || n > 8192 || d < 4 || d > 256 || d % 4 || width != heads * d ||
        mlx_array_dim(qkv, 1) != 3 * width || (size_t)n * n * heads > SIZE_MAX / sizeof(float) ||
        (size_t)n * n * heads * sizeof(float) > NN_MLX_MAX_PROBABILITY_BYTES)
        return -1;
    mlx_array coeffs[] = {cosine, sine, cos_low, sin_low};
    for (int i = 0; i < 4; i++)
        if (mlx_array_dtype(coeffs[i]) != MLX_FLOAT32 || mlx_array_ndim(coeffs[i]) != 2 ||
            mlx_array_dim(coeffs[i], 0) != n || mlx_array_dim(coeffs[i], 1) != d)
            return -1;
    mlx_array args[] = {qkv, cosine, sine, cos_low, sin_low};
    mlx_vector_array in = mlx_vector_array_new_data(args, 5), result = mlx_vector_array_new();
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_array q = mlx_array_new(), k = mlx_array_new(), p = mlx_array_new();
    int rc = mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){n, width, 2}, 3, MLX_FLOAT32) ||
             mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){width, n, 2}, 3, MLX_FLOAT32) ||
             mlx_fast_metal_kernel_config_set_grid(cfg, n * width, 1, 1) ||
             mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1) ||
             mlx_fast_metal_kernel_apply(&result, ctx->rotary, in, cfg, stream) ||
             mlx_vector_array_get(&q, result, 0) || mlx_vector_array_get(&k, result, 1);
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(in);
    mlx_vector_array_free(result);
    /* Small matrices avoid the extra launches. Larger ones reuse shared Q/K tiles, retaining the dimension order of every compensated dot. */
    int tiled = n >= 1280;
    if (!rc && !tiled)
        rc = probabilities(ctx, &p, q, k, n, heads, d, 0, stream);
    mlx_array *parts = calloc(heads, sizeof(*parts));
    if (!parts)
        rc = -1;
    for (int head = 0; head < heads && !rc; head++) {
        mlx_array prob = mlx_array_new(), flat = mlx_array_new(), v = mlx_array_new(),
                  logits = mlx_array_new();
        parts[head] = mlx_array_new();
        if (tiled) {
            mlx_array index = mlx_array_new_data(&head, (int[]){1}, 1, MLX_INT32),
                      args[] = {q, k, index};
            in = mlx_vector_array_new_data(args, 3);
            result = mlx_vector_array_new();
            cfg = mlx_fast_metal_kernel_config_new();
            rc =
                mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", d) ||
                mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){n, n, 2}, 3, MLX_FLOAT32) ||
                mlx_fast_metal_kernel_config_set_grid(cfg, ((n + 31) / 32) * 128, (n + 31) / 32, 1) ||
                mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1) ||
                mlx_fast_metal_kernel_apply(&result, ctx->logit, in, cfg, stream) ||
                mlx_vector_array_get(&logits, result, 0) ||
                probabilities(ctx, &prob, q, logits, n, 1, d, 1, stream);
            mlx_fast_metal_kernel_config_free(cfg);
            mlx_vector_array_free(in);
            mlx_vector_array_free(result);
            mlx_array_free(index);
        } else {
            rc = mlx_slice(&prob, p, (int[]){head, 0, 0}, 3, (int[]){head + 1, n, n}, 3,
                           (int[]){1, 1, 1}, 3, stream);
        }
        if (!rc)
            rc = mlx_reshape(&flat, prob, (int[]){n, n}, 2, stream) ||
                 mlx_slice(&v, qkv, (int[]){0, 2 * width + head * d}, 2,
                           (int[]){n, 2 * width + (head + 1) * d}, 2, (int[]){1, 1}, 2, stream) ||
                 nn_mlx_matmul(ctx->matmul, &parts[head], flat, v, stream);
        /* Evaluate a complete head before releasing its quadratic scratch. PV still sees all query rows, preserving its full matrix tiles. */
        if (!rc && tiled)
            rc = mlx_array_eval(parts[head]);
        mlx_array_free(prob);
        mlx_array_free(flat);
        mlx_array_free(v);
        mlx_array_free(logits);
    }
    if (!rc) {
        mlx_vector_array values = mlx_vector_array_new_data(parts, heads);
        rc = mlx_concatenate_axis(out, values, 1, stream);
        mlx_vector_array_free(values);
    }
    if (parts) {
        for (int i = 0; i < heads; i++)
            mlx_array_free(parts[i]);
        free(parts);
    }
    mlx_array_free(q);
    mlx_array_free(k);
    mlx_array_free(p);
    return rc ? -1 : 0;
}
