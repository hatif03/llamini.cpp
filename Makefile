# Compiler
CC = gcc

# Compile flags: warnings, optimization, C99, math + pthread libraries.
# -O3 -march=native -fassociative-math -fno-signed-zeros -fno-trapping-math
# let gcc vectorize linear()'s dot-product reduction (plain -O2/-O3 keep it
# scalar, since FP reassociation needs explicit permission) without going as
# far as -ffast-math, which also assumes no NaN/Inf -- see STDLIB.md.
CFLAGS = -Wall -O3 -march=native -fassociative-math -fno-signed-zeros -fno-trapping-math -pthread -std=c99 -lm

# All source files
SRC = main.c tensor.c model.c gguf.c kv_cache.c quant.c tokenizer.c generate.c gpt2.c

# Convert .c to .o
OBJ = $(SRC:.c=.o)

# Output program name
TARGET = llamini

# Build the final program
all: $(TARGET)

# Link object files into executable
$(TARGET): $(OBJ)
	$(CC) $(OBJ) $(CFLAGS) -o $@

# Clean build files
clean:
	rm -f $(OBJ) $(TARGET)