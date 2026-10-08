#include "neuron.h"
#include "net.h"
#include "neural_functions.h"
#include "easy_net.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TRAINING_PATH "./mnist_train.csv"
#define TEST_PATH "./mnist_test.csv"

void create_expected(int number, float expected[10]) {
  for (int i = 0; i < 10; i++) {
    if (i == number) {
      expected[i] = 1.0;
    }
    else {
      expected[i] = 0.0;
    }
  }
}

int create_output(float output[10]) {
  for (int i = 0; i < 10; i++) {
    if (output[i] > 0.9) return i;
  }
  return -1;
}

/*
  * reads the correct number into number and all the pixel values
  * into arr
*/

void read_mnist_line(FILE *file_ptr, float arr[784], int *number) {
  fscanf(file_ptr, "%d,", number);

  for (int i = 0; i < 784; i++) {
    int n = 0;
    fscanf(file_ptr, "%d,", &n);
    arr[i] = n / 255.0;
  }

  fscanf(file_ptr, "%*c%*c"); // delete \r\n from line
}

/*
int main() {
  FILE *training_ptr = fopen(TRAINING_PATH, "r");
  if (!training_ptr) goto cleanup;
  FILE *test_ptr = fopen(TEST_PATH, "r");
  if (!test_ptr) goto cleanup;

  int epochs = 4;
  int batch_size = 50;
  int thread_count = 5;
  int training_size = 5000;

  easy_net_t easy_net = { 0 };

  int layer_count = 3;
  int layer_sizes[] = {1024, 512, 10};
  int input_layer_size = 784;
  activation_e activations[] = {RELU, RELU, IDENTITY};

  create_easy_net(&easy_net, layer_count, layer_sizes, input_layer_size,
                  activations, SOFTMAX_N, ARGMAX_N, IDENTITY,
                  CROSS_ENTROPY, 0.16, 0);
  init_bias_easy(&easy_net, 0.1);
  init_weights_easy(&easy_net);

  create_easy_batch(&easy_net);
  prepare_threads_easy(&easy_net, thread_count);

  int number = 0;
  float **inputs = create_batch_arr(batch_size, input_layer_size);
  float **expecteds = create_batch_arr(batch_size, 10);

  int trained_inputs = 0;

  // -------------- training ------------
  for (int i = 0; i < epochs; i++) {

    while (trained_inputs < training_size) {
      for (int j = 0; j < batch_size; j++) {
        read_mnist_line(training_ptr, inputs[j], &number);
        create_expected(number, expecteds[j]);
      }
      train_batch_easy(&easy_net, inputs, expecteds, batch_size);
      trained_inputs += batch_size;
    }
    fseek(training_ptr, 0, SEEK_SET);
    trained_inputs = 0;
  }

  // --------------- testing ------------
  int test_size = 200;
  int correct = 0;
  for (int i = 0; i < test_size; i++) {
    read_mnist_line(test_ptr, inputs[0], &number);

    float *outputs = input_to_output(easy_net.net, inputs[0]);
    int output_num = create_output(outputs);

    printf("expected: %d | output: %d\n", number, output_num);

    if (output_num == number) {
      printf("CORRECT!\n");
      correct++;
    }
    else {
      printf("INCORRECT :((\n");
    }

    free(outputs);
    outputs = NULL;
  }
  printf("You scored %d / %d = %f\n", correct, test_size,
         ((float)correct) / test_size);

cleanup:
  if (training_ptr) fclose(training_ptr);
  if (test_ptr) fclose(test_ptr);
  return 0;
}
*/
