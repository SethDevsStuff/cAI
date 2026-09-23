#include "neuron.h"
#include "net.h"
#include "neural_functions.h"
#include "serial_net.h"

#include <math.h>
#include <stdio.h>

void compiled_to_serial_net(net_compiled_t *net_c,
                            serial_net_t *s_net) {
  s_net->final_normal_training = net_c->final_normal_training;
  s_net->final_normal_answer = net_c->final_normal_answer;

  s_net->start_normal = net_c->start_normal;

  s_net->loss = net_c->loss;

  s_net->hidden_layers_count = net_c->hidden_layers_count;

  s_net->input_layer_size = net_c->input_layer_size;

  s_net->average_across_nodes = net_c->average_across_nodes;
  s_net->learning_rate = net_c->learning_rate;
}

void serial_to_compiled(serial_net_t *s_net,
                        net_compiled_t *net_c) {
  net_c->final_normal_training = s_net->final_normal_training;
  net_c->final_normal_answer = s_net->final_normal_answer;

  net_c->start_normal = s_net->start_normal;

  net_c->loss = s_net->loss;

  net_c->hidden_layers_count = s_net->hidden_layers_count;

  net_c->input_layer_size = s_net->input_layer_size;

  net_c->average_across_nodes = s_net->average_across_nodes;
  net_c->learning_rate = s_net->learning_rate;
}

void write_compiled_to_file(FILE *file_ptr, net_compiled_t *net_c) {
  serial_net_t s_net = { 0 };
  compiled_to_serial_net(net_c, &s_net);

  // write header (serial_net_t)
  fwrite(&s_net, sizeof(serial_net_t), 1, file_ptr);

  // write layer sizes (arr of ints)
  fwrite(net_c->hidden_layers_sizes, sizeof(int),
         s_net.hidden_layers_count, file_ptr);

  // write layer activation enums(arr)
  fwrite(net_c->hidden_layers_activations, sizeof(activation_e),
         s_net.hidden_layers_count, file_ptr);

  // write layer biases
  for (int i = 0; i < s_net.hidden_layers_count; i++) {
    neuron_t *layer = net_c->hidden_layers[i];
    for (int j = 0; j < net_c->hidden_layers_sizes[i]; j++) {
      fwrite(&(layer[j].bias), sizeof(float), 1, file_ptr);
    }
  }

  // write layer weights
  for (int i = 0; i < s_net.hidden_layers_count; i++) {
    neuron_t *layer = net_c->hidden_layers[i];
    for (int j = 0; j < net_c->hidden_layers_sizes[i]; j++) {
      fwrite(layer[j].weights, sizeof(float),
             layer[j].previous_count, file_ptr);
    }
  }
}
