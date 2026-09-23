#include "neuron.h"
#include "net.h"
#include "neural_functions.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int *duplicate_int_array(int *arr, int n){
  int *arr_copy = malloc(n * sizeof(int));
  if (!arr_copy) return NULL;

  for (int i = 0; i < n; i++) {
    arr_copy[i] = arr[i];
  }

  return arr_copy;
}

activation_function_t *duplicate_activations_array(
  activation_function_t *arr,
  int n){

  activation_function_t *arr_copy = malloc(n * sizeof(*arr_copy));
  if (!arr_copy) return NULL;

  for (int i = 0; i < n; i++) {
    arr_copy[i] = arr[i];
  }

  return arr_copy;
}

activation_function_d_t *duplicate_activations_d_array(
  activation_function_d_t *arr,
  int n){

  activation_function_d_t *arr_copy = malloc(n * sizeof(*arr_copy));
  if (!arr_copy) return NULL;

  for (int i = 0; i < n; i++) {
    arr_copy[i] = arr[i];
  }

  return arr_copy;
}

net_t *create_net(int hidden_layers_count, int *hidden_layers_sizes,
                  float (**hidden_layers_activations)(float),
                  float (**hidden_layers_activations_d)(float, float),
                  float (*start_normal)(float),
                  int input_layer_size,
                  void (*final_normal_training)(float *, float *, int),
                  void (*final_normal_answer)(float *, float *, int),
                  float (*loss)(float, float),
                  float (*loss_normal_combined_d)(float, float),
                  float learning_rate,
                  int average_across_nodes) {
  if (hidden_layers_count <= 0) return NULL;

  // allocate memory for net struct
  net_t *net = calloc(1, sizeof(net_t));
  if (!net) goto cleanup;

  net->hidden_layers_count = hidden_layers_count;

  net->hidden_layers_sizes = duplicate_int_array(
    hidden_layers_sizes, hidden_layers_count
  );
  if ( !(net->hidden_layers_sizes) ) goto cleanup;

  // allocate memory for input & output layers
  net->input_layer = create_neurons(input_layer_size);
  net->input_layer_size = input_layer_size;
  if ( !(net->input_layer) ) goto cleanup;

  // # of neurons in output layer == # of neurons in last hidden layer
  net->output_layer = create_neurons(
    hidden_layers_sizes[hidden_layers_count - 1]
  );
  net->output_layer_size = hidden_layers_sizes[hidden_layers_count - 1];
  if ( !(net->output_layer) ) goto cleanup;

  // allocate memory for array of ptrs to layers
  net->hidden_layers = calloc(hidden_layers_count, sizeof(neuron_t *));
  if ( !(net->hidden_layers) ) goto cleanup;

  // allocate memory for each individual layer
  for (int i = 0; i < hidden_layers_count; i++) {
    net->hidden_layers[i] = create_neurons(hidden_layers_sizes[i]);
    if ( !(net->hidden_layers[i]) ) goto cleanup;

    // give values to each individual neuron in the layer
    for (int j = 0; j < hidden_layers_sizes[i]; j++) {
      neuron_t *neuron = net->hidden_layers[i] + j;

      neuron->activation = hidden_layers_activations[i];
      neuron->activation_d = hidden_layers_activations_d[i];

      if (i == 0) {
        // first hidden layer has input_layer_size weights
        neuron->weights = create_weights(input_layer_size);

        neuron->previous = net->input_layer;
        neuron->previous_count = net->input_layer_size;
      }
      else {
        // number of weights of a node is equal to the number of nodes
        // in the previous layer
        neuron->weights = create_weights(hidden_layers_sizes[i - 1]);

        neuron->previous = net->hidden_layers[i - 1];
        neuron->previous_count = hidden_layers_sizes[i - 1];
      }
      if ( !(neuron->weights) ) goto cleanup;

      if (i == hidden_layers_count - 1) {
        // layer is last hidden layer
        neuron->next = net->output_layer;
        neuron->next_count = net->output_layer_size;
      }
      else {
        neuron->next = net->hidden_layers[i + 1];
        neuron->next_count = hidden_layers_sizes[i + 1];
      }
    }
  }

  net->final_normal_training = final_normal_training;
  net->final_normal_answer = final_normal_answer;
  net->loss_normal_combined_d = loss_normal_combined_d;

  net->start_normal = start_normal;

  net->loss = loss;

  net->average_across_nodes = average_across_nodes;
  net->learning_rate = learning_rate;

  net->hidden_layers_activations = duplicate_activations_array(
    hidden_layers_activations,
    hidden_layers_count
  );
  net->hidden_layers_activations_d = duplicate_activations_d_array(
    hidden_layers_activations_d,
    hidden_layers_count
  );

  return net;


cleanup:
  if (net) {
    delete_neurons(net->input_layer, net->input_layer_size);
    delete_neurons(net->output_layer, net->output_layer_size);
    if (net->hidden_layers_activations)
      free(net->hidden_layers_activations);
    if (net->hidden_layers_activations_d)
      free(net->hidden_layers_activations_d);
    if (net->hidden_layers) goto clean_layers;
  }
  free(net);

  return NULL;

clean_layers:
  for (int i = 0; i < hidden_layers_count; i++) {
    delete_neurons(net->hidden_layers[i], net->hidden_layers_sizes[i]);
    net->hidden_layers[i] = NULL;
  }
  free(net->hidden_layers_sizes);
  net->hidden_layers_sizes = NULL;
  free(net->hidden_layers);
  net->hidden_layers = NULL;

  free(net);

  return NULL;
}

void delete_net(net_t *net) {
  delete_neurons(net->input_layer, net->input_layer_size);
  delete_neurons(net->output_layer, net->output_layer_size);
  for (int i = 0; i < net->hidden_layers_count; i++) {
    delete_neurons(net->hidden_layers[i], net->hidden_layers_sizes[i]);
    net->hidden_layers[i] = NULL;
  }
  free(net->hidden_layers_sizes);
  net->hidden_layers_sizes = NULL;

  free(net->hidden_layers);
  net->hidden_layers = NULL;

  free(net->hidden_layers_activations);
  net->hidden_layers_activations = NULL;

  free(net->hidden_layers_activations_d);
  net->hidden_layers_activations_d = NULL;
  

  free(net);
}

net_t *create_batch_net(net_t *net) {
  net_t *net_copy = create_net(net->hidden_layers_count,
                               net->hidden_layers_sizes,
                               net->hidden_layers_activations,
                               net->hidden_layers_activations_d,
                               net->start_normal,
                               net->input_layer_size,
                               net->final_normal_training,
                               net->final_normal_answer,
                               net->loss,
                               net->loss_normal_combined_d,
                               net->learning_rate,
                               net->average_across_nodes);

  reset_batch_values(net_copy);

  if (pthread_mutex_init(&net_copy->lock, NULL)) {
    delete_net(net_copy);
  }

  return net_copy;
}

void reset_batch_values(net_t *net) {
  for (int i = 0; i < net->hidden_layers_count; i++) {
    neuron_t *layer = net->hidden_layers[i];

    for (int j = 0; j < net->hidden_layers_sizes[i]; j++) {
      neuron_t *neuron = layer + j;

      neuron->local_gradient = 0.0;

      for (int k = 0; k < neuron->previous_count; k++) {
        neuron->weights[k] = 0.0;
      }
    }
  }
}

/*
 * adds the local gradient from one net, to the other.
 * only used for batch/mini-batch training. to should be
 * your batch neural net. to MUST be a batch
 */

void add_gradient_to_batch(net_t *from, net_t *to) {
  pthread_mutex_lock(&to->lock);
  for (int i = 0; i < from->hidden_layers_count; i++) {
    neuron_t *from_layer = from->hidden_layers[i];
    neuron_t *to_layer = to->hidden_layers[i];

    for (int j = 0; j < from->hidden_layers_sizes[i]; j++) {
      neuron_t *to_neuron = to_layer + j;

      to_neuron->local_gradient += from_layer[j].local_gradient;

      for (int k = 0; k < to_neuron->previous_count; k++) {
        to_neuron->weights[k] += from_layer[j].previous[k].y *
          from_layer[j].local_gradient;
      }
    }
  }
  pthread_mutex_unlock(&to->lock);
}

void add_batch_to_batch(net_t *from, net_t *to) {
  pthread_mutex_lock(&to->lock);
  for (int i = 0; i < from->hidden_layers_count; i++) {
    neuron_t *from_layer = from->hidden_layers[i];
    neuron_t *to_layer = to->hidden_layers[i];

    for (int j = 0; j < from->hidden_layers_sizes[i]; j++) {
      neuron_t *to_neuron = to_layer + j;
      //printf("%f\n", to_neuron->local_gradient);

      to_neuron->local_gradient += from_layer[j].local_gradient;

      for (int k = 0; k < to_neuron->previous_count; k++) {
        //printf("%f\n", from_layer[j].previous[k].y);
        to_neuron->weights[k] += from_layer[j].weights[k];
      }
    }
  }
  pthread_mutex_unlock(&to->lock);
}

void init_bias(net_t *net, float val) {
  for (int i = 0; i < net->hidden_layers_count; i++) {
    neuron_t *layer = net->hidden_layers[i];

    for (int j = 0; j < net->hidden_layers_sizes[i]; j++) {
      neuron_t *neuron = layer + j;

      neuron->bias = val;
    }
  }
}

void init_weights(net_t *net, float (**get_weight)(int, int)) {
  for (int i = 0; i < net->hidden_layers_count; i++) {
    neuron_t *layer = net->hidden_layers[i];

    for (int j = 0; j < net->hidden_layers_sizes[i]; j++) {
      neuron_t *neuron = layer + j;

      for (int k = 0; k < neuron->previous_count; k++) {
        neuron->weights[k] = (get_weight[i])(
          neuron->previous_count, neuron->next_count
        );
      }
    }
  }
}

/*
  * puts the values in the final hidden layer through the
  * normalization function and puts those values into the
  * output layer neurons.
*/

void push_to_output_training(net_t *net) {
  int n = net->output_layer_size;

  float *y_arr = malloc(n * sizeof(float));
  float *normal_arr = malloc(n * sizeof(float));

  neuron_t *last_layer = net->hidden_layers[net->hidden_layers_count - 1];

  for (int i = 0; i < n; i++) {
    y_arr[i] = last_layer[i].y;
  }

  net->final_normal_training(y_arr, normal_arr, n);

  for (int i = 0; i < n; i++) {
    net->output_layer[i].x = normal_arr[i];
    net->output_layer[i].y = normal_arr[i];
  }

cleanup:
  free(y_arr);
  free(normal_arr);
}

void push_to_output_answer(net_t *net) {
  int n = net->output_layer_size;

  float *y_arr = malloc(n * sizeof(float));
  float *normal_arr = malloc(n * sizeof(float));
  if (!y_arr || !normal_arr) goto cleanup;

  neuron_t *last_layer = net->hidden_layers[net->hidden_layers_count - 1];

  for (int i = 0; i < n; i++) {
    y_arr[i] = last_layer[i].y;
  }

  net->final_normal_answer(y_arr, normal_arr, n);

  for (int i = 0; i < n; i++) {
    net->output_layer[i].x = normal_arr[i];
    net->output_layer[i].y = normal_arr[i];
  }

cleanup:
  free(y_arr);
  free(normal_arr);
}

float calculate_loss(net_t *net, float expected, float actual) {
  return (net->loss(expected, actual));
}

float calculate_total_loss(net_t *net, float *expecteds) {
  float total = 0;
  int n = net->output_layer_size;

  neuron_t *output = net->output_layer;

  //printf("---------\n");
  for (int i = 0; i < n; i++) {
    float sub = calculate_loss(net, expecteds[i], output[i].y);
    total += sub;
    //printf("%f %f\n", total, sub);
  }

  if (net->average_across_nodes) {
    total /= n;
  }

  return total;
}

void input_in_net(net_t *net, float *inputs) {
  int n = net->input_layer_size;

  neuron_t *input = net->input_layer;

  for (int i = 0; i < n; i++) {
    input[i].x = net->start_normal(inputs[i]);
    input[i].y = input[i].x;
  }
}

void calculate_hidden(net_t *net) {
  int n = net->hidden_layers_count;

  for (int i = 0; i < n; i++) {
    calculate_layer_x(net->hidden_layers[i], net->hidden_layers_sizes[i]);
    run_layer_activation(net->hidden_layers[i],
                               net->hidden_layers_sizes[i]);
  }
}

void input_to_output(net_t *net, float *inputs, float *outputs) {
  input_in_net(net, inputs);
  calculate_hidden(net);
  push_to_output_answer(net);

  for (int i = 0; i < net->output_layer_size; i++) {
    outputs[i] = net->output_layer[i].y;
  }
}

void first_back_prop(net_t *net, float *expecteds) {
  int last_hidden_i = net->hidden_layers_count - 1;
  int last_hidden_size = net->hidden_layers_sizes[last_hidden_i];

  neuron_t *last_hidden = net->hidden_layers[last_hidden_i];
  neuron_t *output_layer = net->output_layer;

  // looping through every neuron in the last hidden layer
  for (int i = 0; i < last_hidden_size; i++){
    neuron_t *neuron = last_hidden + i;

    float dC_dy = net->loss_normal_combined_d(expecteds[i],
                                              output_layer[i].x);

    float dy_dx = neuron->activation_d(neuron->x, neuron->y);

    neuron->local_gradient = dC_dy * dy_dx;

    if (net->average_across_nodes) {
      neuron->local_gradient /= last_hidden_size;
    }
  }
}

/*
  * layer_previous is is the next of all neurons in layer,
  * we use previous here because we are moving backwards.
*/

void layer_back_prop(neuron_t *layer, int layer_n,
                     neuron_t *layer_previous, int layer_previous_n) {
  // loop through every neuron in current layer
  for (int i = 0; i < layer_n; i++) {
    neuron_t *neuron = layer + i;

    neuron->local_gradient = 0;

    // loop through every neuron in the previous layer
    for (int j = 0; j < layer_previous_n; j++) {
      neuron_t *previous_neuron = layer_previous + j;
      float part_sum = previous_neuron->weights[i] *
                        previous_neuron->local_gradient;
      neuron->local_gradient += part_sum;
    }

    neuron->local_gradient *= neuron->activation_d(neuron->x, neuron->y);
  }
}

/*
  * every single local_gradient is set. does NOT adjust weights
  * or biases.
*/

void full_back_prop(net_t *net, float *expecteds) {
  first_back_prop(net, expecteds);

  int layers_n = net->hidden_layers_count;

  // loops through every hidden layer except the last one
  // in reverse order
  for (int i = layers_n - 2; i >= 0; i--) {
    layer_back_prop(net->hidden_layers[i], net->hidden_layers_sizes[i],
                    net->hidden_layers[i + 1],
                    net->hidden_layers_sizes[i + 1]);
  }
}

void update_net(net_t *net) {
  int layers_n = net->hidden_layers_count;

  for (int i = 0; i < layers_n; i++) {
    neuron_t *layer = net->hidden_layers[i];
    int layer_n = net->hidden_layers_sizes[i];

    for (int j = 0; j < layer_n; j++) {
      neuron_t *neuron = layer + j;

      float gradient_rate = net->learning_rate * neuron->local_gradient;

      neuron->bias -= gradient_rate;

      for (int k = 0; k < neuron->previous_count; k++) {
        neuron->weights[k] -= gradient_rate * neuron->previous[k].y;
      }
    }
  }
}

/*
  * uses the local gradients from batch divided by count to
  * calculate the updated weights and biases
*/
void update_net_from_batch(net_t *net, net_t *batch, int count) {
  int layers_n = net->hidden_layers_count;

  for (int i = 0; i < layers_n; i++) {
    neuron_t *net_layer = net->hidden_layers[i];
    neuron_t *batch_layer = batch->hidden_layers[i];
    int layer_n = net->hidden_layers_sizes[i];
    
    for (int j = 0; j < layer_n; j++) {
      neuron_t *neuron = net_layer + j;

      float gradient_rate = net->learning_rate *
        batch_layer[j].local_gradient / count;

      neuron->bias -= gradient_rate;

      for (int k = 0; k < neuron->previous_count; k++) {
        neuron->weights[k] -= net->learning_rate * 
          batch_layer[j].weights[k] / count;
      }
    }
  }
}

net_t *create_mirror_net(net_t *net) {
  // allocate memory for net struct
  net_t *mirror_net = calloc(1, sizeof(net_t));
  if (!mirror_net) goto cleanup;

  *mirror_net = *net;

  // prevents layers in net from being freed if mirror_net
  // layers fail to allocate
  mirror_net->input_layer = NULL;
  mirror_net->output_layer = NULL;
  mirror_net->hidden_layers = NULL;

  //mirror_net->hidden_layers_sizes = net->hidden_layers_sizes;

  // allocate memory for input & output layers
  mirror_net->input_layer = create_neurons(net->input_layer_size);
  //mirror_net->input_layer_size = net->input_layer_size;
  if ( !(mirror_net->input_layer) ) goto cleanup;

  // # of neurons in output layer == # of neurons in last hidden layer
  mirror_net->output_layer = create_neurons(
    net->hidden_layers_sizes[net->hidden_layers_count - 1]
  );
  mirror_net->output_layer_size =
    net->hidden_layers_sizes[net->hidden_layers_count - 1];
  if ( !(mirror_net->output_layer) ) goto cleanup;

  // allocate memory for array of ptrs to layers
  mirror_net->hidden_layers = calloc(net->hidden_layers_count,
                                     sizeof(neuron_t *));
  if ( !(mirror_net->hidden_layers) ) goto cleanup;

  // allocate memory for each individual layer
  for (int i = 0; i < net->hidden_layers_count; i++) {
    mirror_net->hidden_layers[i] =
      create_neurons(net->hidden_layers_sizes[i]);
    if ( !(mirror_net->hidden_layers[i]) ) goto cleanup;

    // give values to each individual neuron in the layer
    for (int j = 0; j < net->hidden_layers_sizes[i]; j++) {
      neuron_t *neuron = mirror_net->hidden_layers[i] + j;
      neuron_t *net_neuron = net->hidden_layers[i] + j;

      *neuron = *net_neuron;

      if (i == 0) {
        // first hidden layer has input_layer_size weights
        neuron->previous = mirror_net->input_layer;
      }
      else {
        // number of weights of a node is equal to the number of nodes
        // in the previous layer
        neuron->previous = mirror_net->hidden_layers[i - 1];
      }

      if (i == net->hidden_layers_count - 1) {
        // layer is last hidden layer
        neuron->next = mirror_net->output_layer;
      }
      else {
        neuron->next = mirror_net->hidden_layers[i + 1];
      }
    }
  }

  //mirror_net->hidden_layers_activations = net->hidden_layers_activations;
  //mirror_net->hidden_layers_activations_d = net->hidden_layers_activations_d;

  return mirror_net;


cleanup:
  if (mirror_net) {
    delete_neurons(mirror_net->input_layer, mirror_net->input_layer_size);
    delete_neurons(mirror_net->output_layer, mirror_net->output_layer_size);
    if (mirror_net->hidden_layers) goto clean_layers;
  }
  free(mirror_net);

  return NULL;

clean_layers:
  for (int i = 0; i < net->hidden_layers_count; i++) {
    delete_neurons(mirror_net->hidden_layers[i],
                   mirror_net->hidden_layers_sizes[i]);
    mirror_net->hidden_layers[i] = NULL;
  }
  //free(mirror_net->hidden_layers_sizes);
  //mirror_net->hidden_layers_sizes = NULL;
  free(mirror_net->hidden_layers);
  mirror_net->hidden_layers = NULL;

  free(mirror_net);

  return NULL;
}

void delete_mirror_net(net_t *mirror) {
  delete_neurons(mirror->input_layer, mirror->input_layer_size);
  delete_neurons(mirror->output_layer, mirror->output_layer_size);

  for (int i = 0; i < mirror->hidden_layers_count; i++) {
    delete_mirror_neurons(mirror->hidden_layers[i],
                   mirror->hidden_layers_sizes[i]);
    mirror->hidden_layers[i] = NULL;
  }

  free(mirror->hidden_layers);

  free(mirror);
}

void update_mirror(net_t *net, net_t *mirror) {
  for (int i = 0; i < net->hidden_layers_count; i++) {
    for (int j = 0; j < net->hidden_layers_sizes[i]; j++) {
      neuron_t *mirror_neuron = mirror->hidden_layers[i] + j;
      neuron_t *net_neuron = net->hidden_layers[i] + j;

      mirror_neuron->bias = net_neuron->bias;
    }
  }
}

/*
 * this is the function that gets run on a thread
*/

/*
void *thread_function(void *ptr) {
  thread_wrapper_t *tw = (thread_wrapper_t *) ptr;
  net_t *mirror = tw->mirror;
  net_t *batch = tw->batch;
  float **inputs = tw->inputs;
  float **expecteds = tw->expecteds;
  int inputs_size = tw->inputs_size;

  // loop through inputs
  for (int i = 0; i < inputs_size; i++) {
    input_in_net(mirror, inputs[i]);
    calculate_hidden(mirror);
    push_to_output_training(mirror);
    float loss = calculate_total_loss(mirror, expecteds[i]);
    full_back_prop(mirror, expecteds[i]);
    add_gradient_to_batch(mirror, batch);
  }

  return NULL;
}
*/

void *thread_function(void *ptr) {
  thread_wrapper_t *tw = (thread_wrapper_t *) ptr;
  net_t *mirror = tw->mirror;
  net_t *g_batch = tw->batch;
  net_t *batch = create_batch_net(g_batch);
  float **inputs = tw->inputs;
  float **expecteds = tw->expecteds;
  int inputs_size = tw->inputs_size;

  // loop through inputs
  for (int i = 0; i < inputs_size; i++) {
    input_in_net(mirror, inputs[i]);
    calculate_hidden(mirror);
    push_to_output_training(mirror);
    //float loss = calculate_total_loss(mirror, expecteds[i]);
    //printf("%f\n", loss);
    full_back_prop(mirror, expecteds[i]);
    add_gradient_to_batch(mirror, batch);
  }

  add_batch_to_batch(batch, g_batch);

  delete_net(batch);

  return NULL;
}

/*
 * count is the # of inputs / outputs, neurons
 * is the number of neurons in either the
 * input or output layer, depending on which
 * you are using the array for.
*/
float **create_batch_arr(int count, int neurons) {
  float **arr = calloc(count, sizeof(float *));
  if (!arr) return NULL;
  for (int i = 0; i < count; i++) {
    arr[i] = calloc(neurons, sizeof(float));
    if (!(arr[i])) goto cleanup;
  }

  return arr;

cleanup:
  if (arr) {
    for (int i = 0; i < count; i++) {
      if (arr[i]) {
        free(arr[i]);
        arr[i] = NULL;
      }
      else {
        break;
      }
    }
    free(arr);
  }
  return NULL;
}
