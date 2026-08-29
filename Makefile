# Compiler and compile flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -lm

# Source file list (add quant.c from this section)
SRC = main.c tensor.c kv_cache.c quant.c
OBJ = $(SRC:.c=.o)
TARGET = mini_llama

# Build executable binary
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(CFLAGS)

# Compile single c file to object file
%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)

# Clean all compiled objects and binary
clean:
	rm -f $(OBJ) $(TARGET)