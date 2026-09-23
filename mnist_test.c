#include "neuron.h"
#include "net.h"
#include "neural_functions.h"
#include "easy_net.h"

#include <stdio.h>

#define MODEL_PATH "./mnist.cai"
#define TEST_PATH "./mnist_test.csv"

extern void read_mnist_line(FILE *, float *, int *);
extern int create_output(float *);

int main() {
  FILE *model_ptr = fopen(MODEL_PATH, "rb+");
  if (!model_ptr) goto cleanup;
  FILE *test_ptr = fopen(TEST_PATH, "r");
  if (!test_ptr) goto cleanup;

  easy_net_t easy_net = { 0 };
  read_file_to_easy(model_ptr, &easy_net);

  // --------------- testing ------------
  int test_size = 1000;
  int correct = 0;

  int number = 0;
  float inputs[784] = { 0 };
  float outputs[10] = { 0 };

  for (int i = 0; i < test_size; i++) {
    read_mnist_line(test_ptr, inputs, &number);

    input_to_output(easy_net.net, inputs, outputs);
    int output_num = create_output(outputs);

    //printf("expected: %d | output: %d\n", number, output_num);

    if (output_num == number) {
      //printf("CORRECT!\n");
      correct++;
    }
    else {
      //printf("INCORRECT :((\n");
    }
  }
  printf("You scored %d / %d = %f\n", correct, test_size,
         (correct * 100.0) / test_size);

cleanup:
  if (model_ptr) {
    fclose(model_ptr);
    model_ptr = NULL;
  }
  if (test_ptr) {
    fclose(test_ptr);
    test_ptr = NULL;
  }
}
