# libnn

`libnn` is a C library of low-level neural network inference kernels. It provides matrix multiplication, linear layers, normalization, activations, and causal and bidirectional attention. It supports CPU execution and GPU acceleration via MLX on Apple silicon.

## Build and use

Run `make` from the repository root to build `libnn.a`. The default CPU build requires a C11 compiler and `make`.

Include [`nn.h`](nn.h) in your program. For example, save this as `demo.c`:

```c
#include <stdio.h>
#include "nn.h"

int main(void) {
    const float a[] = {1, 2, 3};
    const float b[] = {4, 5, 6};
    printf("%g\n", nn_dot_f32(a, b, 3));
    return 0;
}
```

```sh
cc -I. demo.c libnn.a -lm -pthread -o demo
./demo  # prints 32
```

## Optional GPU build

On Apple silicon with [MLX C](https://github.com/ml-explore/mlx-c) installed, `make gpu` builds the `libnn_mlx.a` archive. The Makefile looks for Homebrew's `mlx-c` by default; set `MLXC_PREFIX` to use another installation.  GPU programs include [`nn_mlx.h`](nn_mlx.h) and link against `libnn_mlx.a` and MLX C.

```sh
MLXC_PREFIX=$(brew --prefix mlx-c)
cc -I. -I"$MLXC_PREFIX/include" demo.c libnn_mlx.a \
    -L"$MLXC_PREFIX/lib" -lmlxc -lm -pthread -o demo
```
