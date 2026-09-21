#include <math.h>

#ifndef NEURAL_FUNCTIONS_H
#define NEURAL_FUNCTIONS_H

typedef float (*activation_function_t)(float);
typedef float (*activation_function_d_t)(float, float);

typedef float (*loss_function_t)(float, float);

typedef void (*normal_function_t)(float *, float *, int);
typedef float (*loss_normal_combined_d_t)(float, float);

typedef float (*weight_init_function_t)(int, int);

extern float identity(float);
extern float identity_d(float, float);

extern float sigmoid(float);
extern float sigmoid_d(float, float);

//extern float tanhf(float); // included in math.h
extern float tanhf_d(float, float);

extern float relu(float);
extern float relu_d(float, float);

extern float leaky_relu(float);
extern float leaky_relu_d(float, float);

extern void identity_normal(float *, float *, int);
extern void sigmoid_normal(float *, float *, int);
extern void softmax(float *, float *, int);
extern void argmax(float *, float *, int);

extern float squared_error(float, float);
extern float cross_entropy(float, float);
extern float binary_cross_entropy(float, float);


extern float binary_cross_sigmoid_d(float, float);
extern float cross_softmax_d(float, float);
extern float squared_identity_d(float, float);

extern float random_f(float, float);

extern float get_glorot(int, int); // use linear, tanh, sigmoid
extern float get_msra(int, int); // use ReLU, LeakyReLU

#endif
