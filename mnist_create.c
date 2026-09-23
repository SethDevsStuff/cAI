#include "neuron.h"
#include "net.h"
#include "neural_functions.h"
#include "easy_net.h"

#include <stdio.h>

#define MODEL_PATH "./mnist.cai"

int main() {
  FILE *model_ptr = fopen(MODEL_PATH, "wb");

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

  write_easy_to_file(model_ptr, &easy_net);

cleanup:
  if (model_ptr) {
    fclose(model_ptr);
    model_ptr = NULL;
  }
}
