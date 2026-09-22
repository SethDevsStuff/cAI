#include "neuron.h"
#include "net.h"
#include "neural_functions.h"
#include "serial_net.h"

#include <math.h>
#include <stdio.h>

void compiled_to_serial_net(serial_net_compiled_t *s_net_c,
                            serial_net_t *s_net) {
  s_net->final_normal_training = s_net_c->final_normal_training;
  s_net->final_normal_answer = s_net_c->final_normal_answer;

  s_net->start_normal = s_net_c->start_normal;

  s_net->loss = s_net_c->loss;

  s_net->hidden_layers_count = s_net_c->hidden_layers_count;

  s_net->average_across_nodes = s_net_c->average_across_nodes;
  s_net->learning_rate = s_net_c->learning_rate;
}

void write_compiled_to_file(FILE *file_ptr, serial_net_compiled_t *s_net_c) {
  serial_net_t s_net = { 0 };
  compiled_to_serial_net(s_net_c, &s_net);

  // write header (serial_net_t)
  fwrite(&s_net, sizeof(serial_net_t), 1, file_ptr);

  // write layer sizes (arr of ints)
  fwrite(s_net_c->hidden_layers_sizes, sizeof(int),
         s_net.hidden_layers_count, file_ptr);

  // write layer activation enums(arr)
  fwrite(s_net_c->hidden_layers_activations, sizeof(activation_e),
         s_net.hidden_layers_count, file_ptr);

  // write layer biases
  for (int i = 0; i < s_net.hidden_layers_count; i++) {
    neuron_t *layer = s_net_c->hidden_layers[i];
    for (int j = 0; j < s_net_c->hidden_layers_sizes[i]; j++) {
      fwrite(&(layer[j].bias), sizeof(float), 1, file_ptr);
    }
  }

  // write layer weights
  for (int i = 0; i < s_net.hidden_layers_count; i++) {
    neuron_t *layer = s_net_c->hidden_layers[i];
    for (int j = 0; j < s_net_c->hidden_layers_sizes[i]; j++) {
      fwrite(layer[j].weights, sizeof(float),
             layer[j].previous_count, file_ptr);
    }
  }
}
