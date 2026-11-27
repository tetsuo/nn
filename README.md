# nn

`nn` provides neural network inference primitives in C, with SIMD kernels for x86 and Arm CPUs and optional MLX kernels for Apple silicon GPUs.

## Build

From the repository root, build the CPU library with a C11 compiler and CMake 3.23 or newer:

```sh
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu
```

This produces `build/cpu/libnn.a`. The default build uses Accelerate on macOS or OpenBLAS on Linux. To build without BLAS, add `-DNN_USE_BLAS=OFF` to the configure command.

### Optional GPU build

On Apple silicon, you can include both CPU and GPU kernels in `libnn.a`. With [MLX C](https://github.com/ml-explore/mlx-c) installed through Homebrew:

```sh
cmake -S . -B build/mlx -DCMAKE_BUILD_TYPE=Release \
  -DNN_MLX=ON -DCMAKE_PREFIX_PATH="$(brew --prefix mlx-c)"
cmake --build build/mlx
```

For an MLX-only build, add `-DNN_CPU=OFF` to the configure command.

## Using the CPU API

With `NN_CPU=ON`, link `nn::nn` and include [`nn.h`](nn.h). This program prints `32`:

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
