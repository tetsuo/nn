cmake_minimum_required(VERSION 3.23)
project(nn LANGUAGES C)

option(NN_CPU "Build the native CPU kernels" ON)
option(NN_MLX "Build the MLX kernels on macOS" OFF)
option(NN_USE_BLAS "Use Accelerate or OpenBLAS for the native CPU kernels" ON)
set(NN_MLXC_TARGET mlxc CACHE STRING "CMake target providing the MLX-C library and headers")
set(NN_ARCH_FLAGS "-march=native" CACHE STRING "C architecture flags; empty for compiler defaults")

if(NOT NN_CPU AND NOT NN_MLX)
  message(FATAL_ERROR "Enable NN_CPU or NN_MLX")
endif()
if(NN_MLX AND NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(FATAL_ERROR "NN_MLX requires macOS with Metal support")
endif()

include(GNUInstallDirs)
add_library(nn vec.c cosine.c)
add_library(nn::nn ALIAS nn)
target_include_directories(nn PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_compile_features(nn PUBLIC c_std_11)
set_target_properties(nn PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
  C_VISIBILITY_PRESET hidden POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(nn PRIVATE _DEFAULT_SOURCE)
if(APPLE)
  set_target_properties(nn PROPERTIES INSTALL_RPATH "@loader_path")
elseif(UNIX)
  target_link_libraries(nn PRIVATE m)
endif()

# Normalization and cosine are host operations even in an MLX-only build.
# Keep the numerical flags on the same sources as the embedding kernels.
set(nn_numerical_sources vec.c)
if(NN_CPU)
  set(nn_cpu_sources act.c attn.c avx.c gemm.c generic.c layer_norm.c maxsim.c
    neon.c norm.c pool.c rope.c threadpool.c)
  target_sources(nn PRIVATE ${nn_cpu_sources})
  list(APPEND nn_numerical_sources ${nn_cpu_sources} cosine.c)
  find_package(Threads REQUIRED)
  target_link_libraries(nn PRIVATE Threads::Threads)
  if(NN_USE_BLAS)
    target_compile_definitions(nn PRIVATE USE_BLAS)
    if(APPLE)
      find_library(NN_ACCELERATE_LIBRARY Accelerate REQUIRED)
      target_link_libraries(nn PRIVATE "${NN_ACCELERATE_LIBRARY}")
      target_compile_definitions(nn PRIVATE ACCELERATE_NEW_LAPACK)
    else()
      find_path(NN_OPENBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas REQUIRED)
      find_library(NN_OPENBLAS_LIBRARY openblas REQUIRED)
      target_include_directories(nn SYSTEM PRIVATE "${NN_OPENBLAS_INCLUDE_DIR}")
      target_link_libraries(nn PRIVATE "${NN_OPENBLAS_LIBRARY}")
      target_compile_definitions(nn PRIVATE USE_OPENBLAS)
    endif()
  endif()
endif()

if(NN_MLX)
  if(NN_MLXC_TARGET STREQUAL "mlxc" AND NOT TARGET mlxc)
    find_package(MLXC CONFIG REQUIRED)
  endif()
  if(NOT TARGET "${NN_MLXC_TARGET}")
    message(FATAL_ERROR "NN_MLXC_TARGET must name an existing MLX-C target (got '${NN_MLXC_TARGET}')")
  endif()
  set(nn_mlx_sources mlx_act.c mlx_attention.c mlx_gdn.c mlx_linear.c mlx_matmul.c mlx_norm.c)
  target_sources(nn PRIVATE ${nn_mlx_sources})
  list(APPEND nn_numerical_sources ${nn_mlx_sources})
  # nn_mlx.h exposes MLX-C types, so consumers also need its headers and library.
  target_link_libraries(nn PUBLIC "${NN_MLXC_TARGET}")
  install(FILES nn_mlx.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif()

separate_arguments(nn_arch_options NATIVE_COMMAND "${NN_ARCH_FLAGS}")
target_compile_options(nn PRIVATE ${nn_arch_options})
if(CMAKE_C_COMPILER_ID MATCHES "^(AppleClang|Clang|GNU)$")
  target_compile_options(nn PRIVATE -Wall -Wextra -Werror=implicit-function-declaration -UNDEBUG)
  if(NN_CPU AND APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    target_compile_options(nn PRIVATE -falign-functions=32)
  endif()
  set_property(SOURCE ${nn_numerical_sources} APPEND PROPERTY COMPILE_OPTIONS
    -fno-math-errno -ffp-contract=fast -fno-trapping-math -fno-signed-zeros
    -fassociative-math -freciprocal-math)
endif()

install(TARGETS nn ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES nn.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
