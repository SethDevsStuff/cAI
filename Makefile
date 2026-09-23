CC = gcc
CFLAGS = -lm -pthread

SRC = net.c neuron.c neural_functions.c serial_net.c easy_net.c mnist.c

TARGET = mnist_create mnist_train mnist_test

all: $(TARGET)


% : $(SRC) %.c
	$(CC) -o $@ $^ $(CFLAGS)

./%.o : ./%.c
	$(CC) -c -o $@ $<
