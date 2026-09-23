#include "neuron.h"
#include "net.h"
#include "neural_functions.h"
#include "easy_net.h"

#include <stdio.h>

#define MODEL_PATH "./mnist.cai"
#define TRAINING_PATH "./mnist_train.csv"

extern void read_mnist_line(FILE *, float *, int *);
extern int create_output(float *);
extern void create_expected(int, float *);

int main() {
  FILE *model_ptr = fopen(MODEL_PATH, "rb+");
  if (!model_ptr) goto cleanup;
  FILE *training_ptr = fopen(TRAINING_PATH, "r");
  if (!training_ptr) goto cleanup;

  easy_net_t easy_net = { 0 };
  read_file_to_easy(model_ptr, &easy_net);

  int input_layer_size = easy_net.net->input_layer_size;

  int epochs = 1;
  int batch_size = 64;
  int thread_count = 4;
  int training_size = 50000;

  create_easy_batch(&easy_net);
  prepare_threads_easy(&easy_net, thread_count);

  int number = 0;
  float **inputs = create_batch_arr(batch_size, input_layer_size);
  float **expecteds = create_batch_arr(batch_size, 10);

  int trained_inputs = 0;
  int current_batch_size = batch_size;

  // -------------- training ------------
  for (int i = 0; i < epochs; i++) {

    while (trained_inputs < training_size) {
      // handles the last batch of an epoch when training size
      // does not divide by the batch size
      if ((training_size - trained_inputs) < batch_size) {
        current_batch_size = training_size - trained_inputs;
      }
      for (int j = 0; j < current_batch_size; j++) {
        read_mnist_line(training_ptr, inputs[j], &number);
        create_expected(number, expecteds[j]);
      }
      train_batch_easy(&easy_net, inputs, expecteds, current_batch_size);
      trained_inputs += current_batch_size;
    }
    fseek(training_ptr, 0, SEEK_SET);
    trained_inputs = 0;
    current_batch_size = batch_size;
  }

  write_easy_to_file(model_ptr, &easy_net);

cleanup:
  if (model_ptr) {
    fclose(model_ptr);
    model_ptr = NULL;
  }
  if (training_ptr) {
    fclose(training_ptr);
    training_ptr = NULL;
  }
}
