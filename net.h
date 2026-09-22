#include "neuron.h"
#include "neural_functions.h"

#include <pthread.h>

#ifndef NET_H
#define NET_H

typedef struct net {
  /* These are the normalization functions of the outputs.
   * They take two arrays of floats, one input, one output.
   * a common example would be softmax for training and
   * argmax for final answer. The derivative of the training
   * function is needed for backpropagation.
   * You can always just use the identity function :)
   */

  void (*final_normal_training)(float *, float *, int);

  void (*final_normal_answer)(float *, float *, int);

  float (*start_normal)(float);

  float (*loss)(float, float); // takes expected and outputted and returns
                               // the cost
  /* The combined derivative for the loss function and
   * the normalization function. Allows for nicer math
   * with things like softmax and cross entropy eg.
   * First float is raw value, second is after normalizaiton
   */
  float (*loss_normal_combined_d)(float, float);

  int average_across_nodes; // flag for whether to include 1/N in the
                            // total loss or gradient descent.
                            // Use true for transgressional or
                            // multi-label, false for
                            // categorization

  neuron_t **hidden_layers; // array of layers
  int *hidden_layers_sizes; // array of layer sizes
  int hidden_layers_count; // # of hidden layers

  float (**hidden_layers_activations)(float);
  float (**hidden_layers_activations_d)(float, float);

  neuron_t *input_layer;
  int input_layer_size;

  neuron_t *output_layer;
  int output_layer_size;

  float learning_rate;

  pthread_mutex_t lock;
} net_t;

typedef struct thread_wrapper {
  net_t *mirror;
  net_t *batch;
  float **inputs;
  float **expecteds;
  pthread_t thread;
  int inputs_size;
} thread_wrapper_t;

extern int *duplicate_int_array(int *, int);
extern activation_function_t *duplicate_activations_array(
    activation_function_t *, int);
extern activation_function_d_t *duplicate_activations_d_array(
    activation_function_d_t *, int);
extern net_t *create_net(int, int *, float (**)(float),
                         float (**)(float, float),
                         float (*)(float), int,
                         void (*)(float *, float *, int),
                         void (*)(float *, float *, int),
                         float (*)(float, float),
                         float (*)(float, float),
                         float, int);
extern void delete_net(net_t *);
extern net_t *create_batch_net(net_t *);
extern void reset_batch_values(net_t *);
extern void add_gradient_to_batch(net_t *, net_t *);
extern void add_batch_to_batch(net_t *, net_t *);
extern void init_bias(net_t *, float);
extern void init_weights(net_t *, float (**)(int, int));
extern void push_to_output_training(net_t *);
extern void push_to_output_answer(net_t *);
extern float calculate_loss(net_t *, float, float);
extern float calculate_total_loss(net_t *, float *);
extern void input_in_net(net_t *, float *);
extern void calculate_hidden(net_t *);
extern float *input_to_output(net_t *, float *);
extern void first_back_prop(net_t *, float *);
extern void layer_back_prop(neuron_t *, int, neuron_t *, int);
extern void full_back_prop(net_t *net, float *);
extern void update_net(net_t *);
extern void update_net_from_batch(net_t *, net_t *, int);
extern net_t *create_mirror_net(net_t *);
extern void delete_mirror_net(net_t *);
extern void update_mirror(net_t *, net_t *);
extern void *thread_function(void *);
extern float **create_batch_arr(int, int);

#endif
