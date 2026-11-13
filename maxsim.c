#include "nn.h"
#include "impl.h"
#include <float.h>
#include <stdlib.h>

float nn_maxsim(
    const float *query_vectors, int query_tokens, const float *doc_vectors, int doc_tokens, int dim) {
    if (!query_vectors || !doc_vectors || query_tokens <= 0 || doc_tokens <= 0 || dim <= 0)
        return 0.0f;

    /* Keep small per-token maxima in long queries; round only the final
     * score to F32. Dot products and stored vectors remain F32. */
    double score = 0.0;
    for (int qi = 0; qi < query_tokens; qi++) {
        const float *q = query_vectors + (size_t)qi * dim;
        float best = -FLT_MAX;
        for (int di = 0; di < doc_tokens; di++) {
            const float *d = doc_vectors + (size_t)di * dim;
            float dot = nn_dot_f32_impl(q, d, dim);
            if (dot > best)
                best = dot;
        }
        score += best;
    }
    return (float)score;
}

int nn_maxsim_batch(const float *query_vectors,
                            int query_tokens,
                            const float *doc_vectors,
                            const int *doc_offsets,
                            int docs,
                            int dim,
                            float *scores) {
    if (!query_vectors || !doc_vectors || !doc_offsets || !scores || query_tokens <= 0 || docs <= 0 ||
        dim <= 0 || doc_offsets[0] != 0)
        return -1;

    for (int i = 0; i < docs; i++) {
        if (doc_offsets[i] < 0 || doc_offsets[i + 1] <= doc_offsets[i])
            return -1;
    }

#ifdef USE_BLAS
    /* Bound similarity scratch by groups of whole documents. An oversized document runs alone. */
    enum { LATE_GROUP_TOKEN_BUDGET = 4096 };
    int total = doc_offsets[docs];
    int budget = LATE_GROUP_TOKEN_BUDGET;
    size_t cap = (size_t)query_tokens * (size_t)(total < budget ? total : budget);
    int largest = 0;
    for (int i = 0; i < docs; i++) {
        int len = doc_offsets[i + 1] - doc_offsets[i];
        if (len > largest)
            largest = len;
    }
    if ((size_t)query_tokens * (size_t)largest > cap)
        cap = (size_t)query_tokens * (size_t)largest;
    if (cap > SIZE_MAX / sizeof(float))
        return -1;
    float *sim = (float *)malloc(cap * sizeof(float));
    if (sim) {
        int doc = 0;
        while (doc < docs) {
            int first = doc;
            int start = doc_offsets[first];
            doc++;
            while (doc < docs && doc_offsets[doc + 1] - start <= budget &&
                   doc_offsets[doc + 1] - start <= (int)(cap / (size_t)query_tokens))
                doc++;
            int group_tokens = doc_offsets[doc] - start;

            nn_matmul_t(sim, query_vectors, doc_vectors + (size_t)start * dim, query_tokens,
                                dim, group_tokens);

            for (int i = first; i < doc; i++) {
                int s0 = doc_offsets[i] - start;
                int s1 = doc_offsets[i + 1] - start;
                /* Accumulate token maxima in F64 before the final F32 rounding. */
                double score = 0.0;
                for (int qi = 0; qi < query_tokens; qi++) {
                    const float *row = sim + (size_t)qi * group_tokens;
                    float best = row[s0];
                    for (int t = s0 + 1; t < s1; t++)
                        if (row[t] > best)
                            best = row[t];
                    score += best;
                }
                scores[i] = (float)score;
            }
        }
        free(sim);
        return 0;
    }
#endif

    for (int i = 0; i < docs; i++) {
        int start = doc_offsets[i];
        int end = doc_offsets[i + 1];
        scores[i] = nn_maxsim(query_vectors, query_tokens, doc_vectors + (size_t)start * dim,
                                      end - start, dim);
    }
    return 0;
}
