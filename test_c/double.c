#include "neuron.h"
#include "net.h"
#include "neural_functions.h"

#include <stdio.h>
#include <stdlib.h>

int main() {
  int hidden_layers_sizes[] = {10, 10};
  float (*hidden_layers_activations[])(float) = {&identity, &identity};
  float (*hidden_layers_activations_d[])(float, float) = {&identity_d,
                                                          &identity_d};

  net_t *net = create_net(2, hidden_layers_sizes,
                          hidden_layers_activations,
                          hidden_layers_activations_d,
                          1,
                          &identity_normal,
                          &identity_normal,
                          &squared_error,
                          &squared_identity_d,
                          0.0001,
                          true);

  float (*weight_functions[])(int, int) = {&get_msra, &get_msra};

  init_bias(net, 0.1);
  init_weights(net, weight_functions);
  float inputs[] = {1.0};
  float expecteds[] = {0.4};

  for (int i = 0; i < 10000; i++) {
    float total_loss = 0;
    for (int j = -10; j < 10; j++) {
      inputs[0] = (float) j;
      expecteds[0] = 2 * inputs[0];

      input_in_net(net, inputs);
      calculate_hidden(net);
      push_to_output_training(net);
      total_loss += calculate_total_loss(net, expecteds);
      full_back_prop(net, expecteds);
      update_net(net);
    }

    printf("epoch %d | average loss: %f\n", i, (total_loss / 5));
  }

  inputs[0] = 12;
  expecteds[0] = 2 * inputs[0];

  float *output = input_to_output(net, inputs);
  printf("input: %f | output: %f | expected: %f\ndifference:%f",
         inputs[0], output[0], expecteds[0], output[0] - expecteds[0]);

  free(output);

  return 0;
}
