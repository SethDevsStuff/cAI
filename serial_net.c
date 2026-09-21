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

normal_function_t get_normal_function(normal_e n) {
  switch(n) {
    case IDENTITY_N:
      return &identity_normal;
      break;

    case SIGMOID_N:
      return &sigmoid_normal;
      break;

    case SOFTMAX_N:
      return &softmax;
      break;

    case ARGMAX_N:
      return &argmax;
      break;

    default:
      return NULL;
  }
}

loss_function_t get_loss_function(loss_e l) {
  switch (l) {
    case SQUARED_ERROR:
      return &squared_error;
      break;

    case CROSS_ENTROPY:
      return &cross_entropy;
      break;

    case BINARY_CROSS_ENTROPY:
      return &binary_cross_entropy;
      break;

    default:
      return NULL;
  }
}

loss_normal_combined_d_t get_loss_normal_combined_d(loss_e l, normal_e n) {
  if (l == BINARY_CROSS_ENTROPY && n == SIGMOID_N)
    return &binary_cross_sigmoid_d;
  else if (l == CROSS_ENTROPY && n == SOFTMAX_N)
    return &cross_softmax_d;
  else if (l == SQUARED_ERROR && n == IDENTITY_N)
    return &squared_identity_d;
  else
    return NULL;
}

activation_function_t get_activation_function(activation_e a) {
  switch (a) {
    case IDENTITY:
      return &identity;
      break;

    case SIGMOID:
      return &sigmoid;
      break;

    case TAN_H:
      return &tanhf;
      break;

    case RELU:
      return &relu;
      break;

    case LEAKY_RELU:
      return &leaky_relu;
      break;

    default:
      return NULL;
  }
}

activation_function_d_t get_activation_function_d(activation_e a) {
  switch (a) {
    case IDENTITY:
      return &identity_d;
      break;

    case SIGMOID:
      return &sigmoid_d;
      break;

    case TAN_H:
      return &tanhf_d;
      break;

    case RELU:
      return &relu_d;
      break;

    case LEAKY_RELU:
      return &leaky_relu_d;
      break;

    default:
      return NULL;
  }
}

weight_init_function_t get_weight_init_function(activation_e a) {
  switch (a) {
    case IDENTITY:
    case SIGMOID:
    case TAN_H:
      return &get_glorot;
      break;

    case RELU:
    case LEAKY_RELU:
      return &get_msra;
      break;

    default:
      return NULL;
  }
}
