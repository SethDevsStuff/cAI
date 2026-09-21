#include "neuron.h"
#include "net.h"
#include "neural_functions.h"
#include "serial_net.h"
#include "easy_net.h"

#include <pthread.h>
#include <stdlib.h>

void create_easy_net(easy_net_t *easy_net,
                     int hidden_layers_count,
                     int *hidden_layers_sizes,
                     int input_layer_size,
                     activation_e *hidden_layers_activations_e,
                     normal_e final_normal_training_e,
                     normal_e final_normal_answer_e,
                     activation_e start_normal_e,
                     loss_e loss_f_e,
                     float learning_rate,
                     int average_across_nodes) {
  float (**hidden_layers_activations)(float) = { 0 };
  float (**hidden_layers_activations_d)(float, float) = { 0 };

  for (int i = 0; i < hidden_layers_count; i++) {
    hidden_layers_activations[i] = get_activation_function(
      hidden_layers_activations_e[i]
    );
    hidden_layers_activations_d[i] = get_activation_function_d(
      hidden_layers_activations_e[i]
    );
  }

  float (*start_normal)(float) = get_activation_function(start_normal_e);

  void (*final_normal_training)(float *, float *, int) = 
    get_normal_function(final_normal_training_e);
  void (*final_normal_answer)(float *, float *, int) = 
    get_normal_function(final_normal_answer_e);

  float (*loss)(float, float) = get_loss_function(loss_f_e);
  float (*loss_normal_combined_d)(float, float) = 
    get_loss_normal_combined_d(loss_f_e, final_normal_answer_e);

  easy_net->net = create_net(hidden_layers_count, hidden_layers_sizes,
                             hidden_layers_activations,
                             hidden_layers_activations_d,
                             start_normal,
                             input_layer_size,
                             final_normal_training,
                             final_normal_answer,
                             loss,
                             loss_normal_combined_d,
                             learning_rate,
                             average_across_nodes);
  if (!easy_net->net) return;

  easy_net->batches = NULL;
  easy_net->batches_count = -1;

  easy_net->final_normal_training = final_normal_training_e;
  easy_net->final_normal_answer = final_normal_answer_e;

  easy_net->start_normal = start_normal_e;

  easy_net->loss = loss_f_e;
  
  easy_net->hidden_layers_activations = malloc(sizeof(activation_e) *
                                               hidden_layers_count);
  if (!easy_net->hidden_layers_activations) return;
  for (int i = 0; i < hidden_layers_count; i++) {
    easy_net->hidden_layers_activations[i] =
      hidden_layers_activations_e[i];
  }
}

void create_easy_batch(easy_net_t *easy_net) {
  easy_net->batch = create_batch_net(easy_net->net);
}

void delete_easy_batch(easy_net_t *easy_net) {
  free(easy_net->batch);
  easy_net->batch = NULL;
}
