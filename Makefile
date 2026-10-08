CC = gcc
CFLAGS = -Wall -Werror -Iinclude
LIBRARIES = -lm -pthread

SRC = net.c neuron.c neural_functions.c serial_net.c easy_net.c mnist.c

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
OUTPUT_DIR = lib

SRCS = $(wildcard $(SRC_DIR)/*.c)

OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

INCLUDES = $(wildcard $(INC_DIR)/*.h)

TARGET = $(OUTPUT_DIR)/libc_ai.a


INSTALL_DIR = /usr/lib
INSTALL_INC_DIR = /usr/include

INSTALL_INC_LIST = $(patsubst $(INC_DIR)/%, $(INSTALL_INC_DIR)/%, $(INCLUDES))

all: $(TARGET)

$(TARGET): $(OBJS)
	mkdir -p $(OUTPUT_DIR)
	ar rcs $@ $^

./$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	mkdir -p $(OBJ_DIR)
	$(CC) -c -o $@ $< $(CFLAGS) $(LIBRARIES)

install: $(TARGET)
	cp $(TARGET) $(INSTALL_DIR)
	cp $(INCLUDES) $(INSTALL_INC_DIR)

uninstall:
	rm $(INSTALL_DIR)/libc_ai.a
	rm $(INSTALL_INC_LIST)
