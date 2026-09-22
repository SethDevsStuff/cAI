#include "neural_functions.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

float identity(float x) {
  return x;
}

float identity_d(float x, float y) {
  return 1;
}

float sigmoid(float x) {
  return (1 / (1 + expf(-1 * x) ));
}

float sigmoid_d(float x, float y) {
  return (y * (1 - y));
}

float tanhf_d(float x, float y) {
  return 1 - (y * y);
}

float relu(float x) {
  if (x > 0) return x;

  return 0;
}

float relu_d(float x, float y) {
  if (x >= 0) return 1;

  return 0;
}

float leaky_relu(float x) {
  if (x > 0) return x;

  return 0.01 * x;
}

float leaky_relu_d(float x, float y) {
  if (x >= 0) return 1;

  return 0.01;
}

void identity_normal(float *a, float *b, int n) {
  for (int i = 0; i < n; i++) {
    b[i] = a[i];
  }
}

void sigmoid_normal(float *a, float *b, int n) {
  for (int i = 0; i < n; i++) {
    b[i] = sigmoid(a[i]);
  }
}

void softmax(float *a, float *b, int n) {
  float denom = 0;
  for (int i = 0; i < n; i++) {
    denom += expf(a[i]);
  }

  for (int i = 0; i < n; i++) {
    b[i] = expf(a[i]) / denom;
  }
}

void argmax(float *a, float *b, int n) {
  int biggest = 0;
  for (int i = 1; i < n; i++) {
    if (a[i] > a[biggest]) biggest = i;
  }

  for (int i = 0; i < n; i++) {
    if (i == biggest) {
      b[i] = 1.0;
    }
    else {
      b[i] = 0.0;
    }
  }
}

float squared_error(float expected, float actual) {
  float diff = expected - actual;
  return (diff * diff);
}

float cross_entropy(float expected, float actual) {
  float n = -1 * expected * logf(actual);

  return n;
}

float binary_cross_entropy(float expected, float actual) {
  if (expected == actual) return 0.0;
  if (expected == (1 - actual)) return 5.0;
  float n = expected * logf(actual);
  n += (1 - expected) * logf(1 - actual);
  n *= -1;

  return n;
}

float binary_cross_sigmoid_d(float expected, float actual) {
  return actual - expected;
}

float cross_softmax_d(float expected, float actual) {
  return actual - expected;
}

float squared_identity_d(float expected, float actual) {
  return 2 * (actual - expected);
}

float random_f(float min, float max) {
  return ( (float) rand() / ((float) RAND_MAX)) * (max - min) + min;
}

float get_glorot(int input, int output) {
  float r = sqrtf(6.0 / (input + output));

  return ( ( (float)rand() ) / ( (float) RAND_MAX ) ) * 2 * r - r;
}

float get_msra(int input, int output) {
  float r = sqrtf(6.0 / input);

  return ( ( (float)rand() ) / ( (float) RAND_MAX ) ) * 2 * r - r;
}

normal_function_t get_normal_function(normal_e n) {
  switch(n) {
    case IDENTITY_N:
      return &identity_normal;
      break;

    case SIGMOID_N:
      return &sigmoid_normal;
      break;

    case SOFTMAX_N:
      return &softmax;
      break;

    case ARGMAX_N:
      return &argmax;
      break;

    default:
      return NULL;
  }
}

loss_function_t get_loss_function(loss_e l) {
  switch (l) {
    case SQUARED_ERROR:
      return &squared_error;
      break;

    case CROSS_ENTROPY:
      return &cross_entropy;
      break;

    case BINARY_CROSS_ENTROPY:
      return &binary_cross_entropy;
      break;

    default:
      return NULL;
  }
}

loss_normal_combined_d_t get_loss_normal_combined_d(loss_e l, normal_e n) {
  if (l == BINARY_CROSS_ENTROPY && n == SIGMOID_N)
    return &binary_cross_sigmoid_d;
  else if (l == CROSS_ENTROPY && n == SOFTMAX_N)
    return &cross_softmax_d;
  else if (l == SQUARED_ERROR && n == IDENTITY_N)
    return &squared_identity_d;
  else
    return NULL;
}

activation_function_t get_activation_function(activation_e a) {
  switch (a) {
    case IDENTITY:
      return &identity;
      break;

    case SIGMOID:
      return &sigmoid;
      break;

    case TAN_H:
      return &tanhf;
      break;

    case RELU:
      return &relu;
      break;

    case LEAKY_RELU:
      return &leaky_relu;
      break;

    default:
      return NULL;
  }
}

activation_function_d_t get_activation_function_d(activation_e a) {
  switch (a) {
    case IDENTITY:
      return &identity_d;
      break;

    case SIGMOID:
      return &sigmoid_d;
      break;

    case TAN_H:
      return &tanhf_d;
      break;

    case RELU:
      return &relu_d;
      break;

    case LEAKY_RELU:
      return &leaky_relu_d;
      break;

    default:
      return NULL;
  }
}

weight_init_function_t get_weight_init_function(activation_e a) {
  switch (a) {
    case IDENTITY:
    case SIGMOID:
    case TAN_H:
      return &get_glorot;
      break;

    case RELU:
    case LEAKY_RELU:
      return &get_msra;
      break;

    default:
      return NULL;
  }
}
