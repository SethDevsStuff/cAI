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
