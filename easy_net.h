#include "neuron.h"
#include "net.h"
#include "neural_functions.h"
#include "serial_net.h"

#include <pthread.h>

#ifndef EASY_NET_H
#define EASY_NET_H


typedef struct easy_net {
  net_t *net;
  net_t *batch;

  thread_wrapper_t *t_wrappers;
  int thread_count;

  normal_e final_normal_training;
  normal_e final_normal_answer;

  activation_e start_normal;

  loss_e loss;

  activation_e *hidden_layers_activations;
} easy_net_t;

extern void create_easy_net(easy_net_t *, int, int *, int,
                     activation_e *, normal_e, normal_e,
                     activation_e, loss_e, float, int);
extern void init_bias_easy(easy_net_t *, float);
extern void init_weights_easy(easy_net_t *);
extern void create_easy_batch(easy_net_t *);
extern void delete_easy_batch(easy_net_t *);
extern void prepare_threads_easy(easy_net_t *, int);
extern void update_mirrors_easy(easy_net_t *);
extern void train_batch_easy(easy_net_t *, float **, float **, int);


#endif
