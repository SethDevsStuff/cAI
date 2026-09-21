#include "neuron.h"
#include "net.h"
#include "neural_functions.h"

#include <stdio.h>
#include <stdlib.h>

int main() {
  int hidden_layers_sizes[] = {128, 1};
  float (*hidden_layers_activations[])(float) = {&leaky_relu,
                                                 &identity};
  float (*hidden_layers_activations_d[])(float, float) = {&leaky_relu_d,
                                                          &identity_d};

  net_t *net = create_net(2, hidden_layers_sizes,
                          hidden_layers_activations,
                          hidden_layers_activations_d,
                          &identity,
                          1,
                          &identity_normal,
                          &identity_normal,
                          &squared_error,
                          &squared_identity_d,
                          0.001,
                          true);

  float (*weight_functions[])(int, int) = {&get_msra, &get_glorot};

  init_bias(net, 0.1);
  init_weights(net, weight_functions);
  float inputs[] = {1.0};
  float expecteds[] = {0.4};

  for (int i = 0; i < 10000; i++) {
    float total_loss = 0;
    for (int j = -10; j < 10; j++) {
      inputs[0] = (float) j;
      expecteds[0] = inputs[0] * inputs[0] / 100;
      inputs[0] /= 100;

      input_in_net(net, inputs);
      calculate_hidden(net);
      push_to_output_training(net);
      total_loss += calculate_total_loss(net, expecteds);
      full_back_prop(net, expecteds);
      update_net(net);
    }

    //printf("epoch %d | average loss: %f\n", i, (total_loss / 5));
  }

  inputs[0] = -5;
  expecteds[0] = inputs[0] * inputs[0];
  inputs[0] /= 100;

  float *output = input_to_output(net, inputs);
  output[0] *= 100;
  printf("input: %f | output: %f | expected: %f\ndifference:%f\n",
         inputs[0], output[0], expecteds[0], output[0] - expecteds[0]);

  free(output);

  return 0;
}
