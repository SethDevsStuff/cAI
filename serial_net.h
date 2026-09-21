#include "neuron.h"
#include "net.h"
#include "neural_functions.h"

#include <math.h>
#include <stdio.h>

#ifndef SERIAL_NET_H
#define SERIAL_NET_H

typedef enum normal {
  IDENTITY_N,
  SIGMOID_N,
  SOFTMAX_N,
  ARGMAX_N
} normal_e;

typedef enum loss {
  SQUARED_ERROR,
  CROSS_ENTROPY,
  BINARY_CROSS_ENTROPY
} loss_e;

typedef enum activation {
  IDENTITY,
  SIGMOID,
  TAN_H,
  RELU,
  LEAKY_RELU
} activation_e;

/*
 * this is all the data needed to save to a file, it does
 * not itself get saved to the file however.
 */

typedef struct serial_net_compiled {
  normal_e final_normal_training;
  normal_e final_normal_answer;

  activation_e start_normal;

  loss_e loss;

  activation_e *hidden_layers_activations;

  neuron_t **hidden_layers;
  int *hidden_layers_sizes;
  int hidden_layers_count;

  int average_across_nodes;
  float learning_rate;
} serial_net_compiled_t;

/*
 * this is the header that actually gets saved to the
 * file.
 */

typedef struct serial_net {
  normal_e final_normal_training;
  normal_e final_normal_answer;

  activation_e start_normal;

  loss_e loss;

  int hidden_layers_count;

  int average_across_nodes;
  float learning_rate;
} serial_net_t;

/*
 * data saved to a file is structured as the following:
 * // serial_net_t (header) //
 * ---------------------------
 * // hidden layers sizes(ints) // 
 * int, int, int, int, etc.
 * ---------------------------
 * // hidden layers activations (activation_e)
 * ---------------------------
 * // hidden layers biases //
 * ---------------------------
 * // hidden layers weights //
 * ---------------------------
 */

void compiled_to_serial_net(serial_net_compiled_t *,
                            serial_net_t *);
void write_compiled_to_file(FILE *, serial_net_compiled_t *);
normal_function_t get_normal_function(normal_e);
loss_function_t get_loss_function(loss_e);
loss_normal_combined_d_t get_loss_normal_combined_t(loss_e, normal_e);
activation_function_t get_activation_function(activation_e);
activation_function_d_t get_activation_function_d(activation_e);
weight_init_function_t get_weight_init_function(activation_e);

#endif
