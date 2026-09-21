#include "neuron.h"
#include "net.h"
#include "neural_functions.h"
#include "serial_net.h"

#ifndef EASY_NET_H
#define EASY_NET_H


typedef struct easy_net {
  net_t *net;
  net_t *batch;

  normal_e final_normal_training;
  normal_e final_normal_answer;

  activation_e start_normal;

  loss_e loss;

  activation_e *hidden_layers_activations;
} easy_net_t;


#endif
