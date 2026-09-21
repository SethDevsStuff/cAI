#ifndef NEURON_H
#define NEURON_H


typedef struct neuron {
  float (*activation)(float); // activation function
  float (*activation_d)(float, float); // derivative of activation function

  struct neuron *previous; // pointer to previous layer neurons(array)
  struct neuron *next; // pointer to next layer neurons(array)

  float *weights; // array of weights, capacity == previous_count

  int previous_count;
  int next_count;

  float bias;

  float x; // input
  float y; // output

  float local_gradient; // see README for more details on the math
} neuron_t;

extern neuron_t *create_neurons(int);
extern float *create_weights(int);
extern void delete_neurons(neuron_t *, int);
extern void run_activation(neuron_t *);
extern void run_layer_activation(neuron_t *, int);
extern void calculate_x(neuron_t *);
extern void calculate_layer_x(neuron_t *, int);

#endif
