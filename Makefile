CC = gcc
CFLAGS = -Wall -Werror
LIBRARIES = -lm -pthread

SRC = net.c neuron.c neural_functions.c serial_net.c easy_net.c mnist.c

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
OUTPUT_DIR = lib

SRCS = $(wildcard $(SRC_DIR)/*.c)

OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

TARGET = $(OUTPUT_DIR)/c_ai.a

all:
	@echo $(SRCS)
	@echo $(OBJS)

$(TARGET): $(OBJS)
	ar rcs $@ $^

./obj/%.o : SRC_DIR/%.c
	$(CC) -c -o $@ $<
