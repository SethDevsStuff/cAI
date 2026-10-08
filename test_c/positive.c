#include "neuron.h"
#include "net.h"
#include "neural_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  int hidden_layers_sizes[] = {4, 4, 1};
  float (*activations[])(float) = {&tanhf, &relu, &sigmoid};
  float (*activations_d[])(float, float) = {&tanhf_d, &relu_d,
                                            &sigmoid_d};
  float (*weights[])(int, int) = {&get_glorot, &get_msra, &get_glorot};
  
  net_t *net = create_net(3, hidden_layers_sizes,
                          activations, activations_d,
                          &tanhf,
                          1,
                          &identity_normal, &identity_normal,
                          &squared_error,
                          &squared_identity_d,
                          0.001,
                          1);


  init_bias(net, 0);
  init_weights(net, weights);

  float inputs[] = {0.0};
  float expecteds[] = {0.0};

  srand(time(NULL));

  for (int i = 0; i < 1000; i++) {
    float epoch_cost = 0;
    for (int j = 0; j < 100; j++) {
      inputs[0] = random_f(-10, 10);

      if (inputs[0] < 0) {
        expecteds[0] = 0.0;
      }
      else {
        expecteds[0] = 1.0;
      }

      input_in_net(net, inputs);
      calculate_hidden(net);
      push_to_output_training(net);
      epoch_cost += calculate_total_loss(net, expecteds);
      full_back_prop(net, expecteds);
      update_net(net);

    }
    printf("epoch %d | cost: %f\n", i, epoch_cost / 100);
  }

  inputs[0] = -12;

  expecteds[0] = 0.0;

  float *outputs = input_to_output(net, inputs);

  outputs[0] = (int) (outputs[0] + 0.5);

  if (outputs[0] == 0) {
    printf("net predicts: NEGATIVE\n");
  }
  else {
    printf("net predicts: POSITIVE\n");
  }

  if (inputs[0] < 0) {
    printf("%f is NEGATIVE\n", inputs[0]);
  }
  else {
    printf("%f is POSITIVE\n", inputs[0]);
  }
  return 0;
}
