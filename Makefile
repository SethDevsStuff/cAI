CC = gcc
CFLAGS = -lm -pthread

SRC = net.c neuron.c neural_functions.c easy_net.c

TARGET = a.out

all: byte_reverse_easy


% : $(SRC) %.c
	$(CC) $(CFLAGS) -o $@ $^

$(TARGET) : $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

./%.o : ./%.c
	$(CC) -c -o $@ $<
