CC ?= cc
AR ?= ar
ARFLAGS := rcs

TARGET := libnn.a

# CPU implementation. The architecture-specific files are safe to compile on
# every platform; their contents are enabled only when the compiler exposes
# the matching SIMD ISA.
SRCS := \
	act.c \
	attn.c \
	avx.c \
	cosine.c \
	gemm.c \
	generic.c \
	layer_norm.c \
	maxsim.c \
	neon.c \
	norm.c \
	pool.c \
	rope.c \
	threadpool.c \
	vec.c
OBJS := $(SRCS:.c=.o)

MLX_SRCS := mlx_act.c mlx_attention.c mlx_gdn.c mlx_linear.c mlx_matmul.c mlx_norm.c
MLX_OBJS := $(MLX_SRCS:.c=.o)
MLXC_PREFIX ?= $(shell brew --prefix mlx-c 2>/dev/null)

DEPS := $(OBJS:.o=.d) $(MLX_OBJS:.o=.d)

CPPFLAGS += -I.
CFLAGS ?= -O3 -std=c11
CFLAGS += -fPIC

.PHONY: all libnn gpu clean

all: libnn

libnn: $(TARGET)

gpu: libnn_mlx.a

$(TARGET): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

libnn_mlx.a: $(MLX_OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(MLX_OBJS): CPPFLAGS += -DUSE_GPU -I$(MLXC_PREFIX)/include

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

-include $(DEPS)

clean:
	$(RM) $(OBJS) $(MLX_OBJS) $(DEPS) $(TARGET) libnn_mlx.a
