#include "neuron.h"
#include "net.h"
#include "neural_functions.h"
#include "serial_net.h"
#include "easy_net.h"

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

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
  float (**hidden_layers_activations)(float) =
    calloc(hidden_layers_count,
           sizeof(activation_function_t));
  float (**hidden_layers_activations_d)(float, float) =
    calloc(hidden_layers_count,
           sizeof(activation_function_d_t));

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

  free(hidden_layers_activations);
  free(hidden_layers_activations_d);

  easy_net->batch = NULL;

  easy_net->t_wrappers = NULL;
  easy_net->thread_count = 0;

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

void init_bias_easy(easy_net_t *easy_net, float n) {
  init_bias(easy_net->net, n);
}

void init_weights_easy(easy_net_t *easy_net) {
  float (**weights)(int, int) = malloc(easy_net->net->hidden_layers_count *
                                       sizeof(weight_init_function_t));

  for (int i = 0; i < easy_net->net->hidden_layers_count; i++) {
    weights[i] =
      get_weight_init_function(easy_net->hidden_layers_activations[i]);
  }

  init_weights(easy_net->net, weights);

  free(weights);
}

void create_easy_batch(easy_net_t *easy_net) {
  easy_net->batch = create_batch_net(easy_net->net);
}

void delete_easy_batch(easy_net_t *easy_net) {
  free(easy_net->batch);
  easy_net->batch = NULL;
}

void prepare_threads_easy(easy_net_t *easy_net, int thread_count) {
  easy_net->t_wrappers = calloc(thread_count, sizeof(thread_wrapper_t));
  if (!easy_net->t_wrappers) return;

  easy_net->thread_count = thread_count;

  for (int i = 0; i < thread_count; i++) {
    thread_wrapper_t *i_wrapper = easy_net->t_wrappers + i;
    i_wrapper->mirror = create_mirror_net(easy_net->net);
    if (!i_wrapper->mirror) goto cleanup;

    i_wrapper->batch = easy_net->batch;
  }

  return;

cleanup:
  easy_net->thread_count = 0;
  if (easy_net->t_wrappers) {
    for (int i = 0; i < thread_count; i++) {
      thread_wrapper_t *i_wrapper = easy_net->t_wrappers + i;
      if (i_wrapper->mirror) {
        delete_mirror_net(i_wrapper->mirror);
      }
      else {
        break;
      }
    }
    free(easy_net->t_wrappers);
  }
}

void update_mirrors_easy(easy_net_t *easy_net) {
  for (int i = 0; i < easy_net->thread_count; i++) {
    thread_wrapper_t *tw = easy_net->t_wrappers + i;

    update_mirror(easy_net->net, tw->mirror);
  }
}

void train_batch_easy(easy_net_t *easy_net, float **inputs,
                      float **expecteds, int batch_size) {
  reset_batch_values(easy_net->batch);
  update_mirrors_easy(easy_net);

  int inputs_per_thread = batch_size / easy_net->thread_count;
  int remainder = batch_size % easy_net->thread_count;

  float **inputs_head = inputs;
  float **expecteds_head = expecteds;

  thread_wrapper_t *tw = NULL;
  for (int i = 0; i < easy_net->thread_count; i++) {
    tw = easy_net->t_wrappers + i;

    tw->inputs_size = inputs_per_thread;

    if (remainder) {
      tw->inputs_size += 1;
      remainder -= 1;
    }

    tw->inputs = inputs_head;
    tw->expecteds = expecteds_head;

    inputs_head = inputs_head + tw->inputs_size;
    expecteds_head = expecteds_head + tw->inputs_size;

    pthread_create(&tw->thread, NULL, &thread_function, tw);
  }

  for (int i = 0; i < easy_net->thread_count; i++) {
    tw = easy_net->t_wrappers + i;

    pthread_join(tw->thread, NULL);
  }
  /*
  printf("g_batch layer 0 neuron 0 grad: %f\n", 
         easy_net->batch->hidden_layers[0][0].weights[0]);
  */
  update_net_from_batch(easy_net->net, easy_net->batch, batch_size);
}
