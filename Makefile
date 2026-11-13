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
DEPS := $(OBJS:.o=.d)

CPPFLAGS += -I.
CFLAGS ?= -O3 -std=c11
CFLAGS += -fPIC

.PHONY: all libnn clean

all: libnn

libnn: $(TARGET)

$(TARGET): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

-include $(DEPS)

clean:
	$(RM) $(OBJS) $(DEPS) $(TARGET)
