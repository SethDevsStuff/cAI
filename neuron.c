#include "neuron.h"

#include <stdlib.h>
#include <stdio.h>

neuron_t *create_neurons(int count) {
  if (count <= 0) return NULL;

  neuron_t *neurons = calloc(count, sizeof(neuron_t));
  if (!neurons) return NULL;

  return neurons;
}

float *create_weights(int count) {
  if (count <= 0 ) return NULL;

  float *weights = malloc(count * sizeof(float));
  if (!weights) return NULL;

  return weights;
}

void delete_neurons(neuron_t *neurons, int neuron_count) {
  if (!neurons) return;
  for (int i = 0; i < neuron_count; i++) {
    // ensures no null dereference due to failed malloc of neuron
    if (neurons + i) {
      free(neurons[i].weights);
      neurons[i].weights = NULL;
    }
  }
  free(neurons);
}

void run_activation(neuron_t *n) {
  n->y = n->activation(n->x);
}

void run_layer_activation(neuron_t *layer, int n) {
  for (int i = 0; i < n; i++) {
    run_activation(layer + i);
  }
}

/*
  * given a neuron n, calculate x using the previous layer y's
*/
void calculate_x(neuron_t *n) {
  n->x = n->bias;

  for (int i = 0; i < n->previous_count; i++) {
    neuron_t *neuron_i = n->previous + i;
    n->x += n->weights[i] * neuron_i->y;
  }
}

void calculate_layer_x(neuron_t *layer, int n) {
  for (int i = 0; i < n; i++) {
    calculate_x(layer + i);
  }
}
