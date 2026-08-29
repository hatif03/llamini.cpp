CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -lm

SRC = main.c tensor.c kv_cache.c quant.c gguf.c model.c tokenizer.c generate.c
OBJ = $(SRC:.c=.o)
TARGET = mini_llama

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(CFLAGS)

%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)

clean:
	rm -f $(OBJ) $(TARGET)