# Compiler
CC = gcc

# Compile flags: warnings, optimization, C99, math library
CFLAGS = -Wall -O2 -std=c99 -lm

# All source files
SRC = main.c tensor.c model.c gguf.c kv_cache.c quant.c

# Convert .c to .o
OBJ = $(SRC:.c=.o)

# Output program name
TARGET = mini_llama

# Build the final program
all: $(TARGET)

# Link object files into executable
$(TARGET): $(OBJ)
	$(CC) $(OBJ) $(CFLAGS) -o $@

# Clean build files
clean:
	rm -f $(OBJ) $(TARGET)