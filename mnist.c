#include "neuron.h"
#include "net.h"
#include "neural_functions.h"

#include <stdio.h>
#include <math.h>

#define TRAINING_PATH "./mnist_train.csv"


float normalize_pixel(float n) {
  float max = 255;

  return n / 255;
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
    arr[i] = (float) n;
  }

  fscanf(file_ptr, "%*c%*c"); // delete \r\n from line
}

int main() {
  FILE *training_ptr = fopen(TRAINING_PATH, "r");
  if (!training_ptr) goto cleanup;

  int layer_sizes[] = {1024, 512, 128, 10};
  float (*activations[])(float) = {&sigmoid, &sigmoid, &sigmoid, &identity};
  float (*activations_d[])(float, float) = {&sigmoid_d, &sigmoid_d,
                                            &sigmoid_d, &identity_d};

  float (*weights[])(int, int) = {&get_glorot, &get_glorot, &get_glorot,
                                  &get_glorot};

  net_t *net = create_net(4, layer_sizes, activations, activations_d,
                          &normalize_pixel,
                          784,
                          &softmax, &softmax,
                          &cross_entropy,
                          &cross_softmax_d,
                          0.001,
                          0);
  net_t *batch = create_batch_net(net);
  init_bias(net, 0);
  init_weights(net, weights);

  int correct = 0;
  float inputs[784] = { 0 };
  float expecteds[10] = { 0 };

  for (int i = 0; i < 1; i++) {
    for (int i = 0; i < 1000; i++) {
      read_mnist_line(training_ptr, inputs, &correct);

      // reset expecteds array
      for (int i = 0; i < 10; i++) {
        expecteds[i] = 0.0;
      }
      expecteds[correct] = 1.0;

      input_in_net(net, inputs);
      calculate_hidden(net);
      push_to_output_training(net);
      float loss = calculate_total_loss(net, expecteds);
      full_back_prop(net, expecteds);
      update_net(net);

      printf("%f\n", loss);
    }

    printf("epoch %d\n", i);
  }

cleanup:
  if (training_ptr) fclose(training_ptr);
  if (net) delete_net(net);
  return 0;
}
