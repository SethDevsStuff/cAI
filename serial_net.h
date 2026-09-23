#include "neuron.h"
#include "net.h"
#include "neural_functions.h"

#include <math.h>
#include <stdio.h>

#ifndef SERIAL_NET_H
#define SERIAL_NET_H

/*
 * this is all the data needed to save to a file, it does
 * not itself get saved to the file however.
 */

typedef struct net_compiled {
  normal_e final_normal_training;
  normal_e final_normal_answer;

  activation_e start_normal;

  loss_e loss;

  activation_e *hidden_layers_activations;

  neuron_t **hidden_layers;
  int *hidden_layers_sizes;
  int hidden_layers_count;

  int input_layer_size;

  int average_across_nodes;
  float learning_rate;
} net_compiled_t;

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

  int input_layer_size;

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

extern void compiled_to_serial_net(net_compiled_t *,
                            serial_net_t *);
extern void serial_to_compiled(serial_net_t *, net_compiled_t *);
extern void write_compiled_to_file(FILE *, net_compiled_t *);

#endif
