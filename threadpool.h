#ifndef NN_THREADPOOL_H
#define NN_THREADPOOL_H

#include "nn.h"

typedef void (*nn_cpu_parallel_fn_t)(int tid, int n_threads, void *arg);
void nn_cpu_parallel_for(nn_cpu_parallel_fn_t fn, void *arg);

#endif
