CC = gcc
CFLAGS = -lm

SRC = net.c neuron.c neural_functions.c

TARGET = a.out


% : $(SRC) %.c
	$(CC) $(CFLAGS) -o $@ $^

$(TARGET) : $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

./%.o : ./%.c
	$(CC) -c -o $@ $<
