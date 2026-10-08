#include "neural_functions.h"
#include "easy_net.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void normalize_float_bits(float *arr) {
  for (int i = 0; i < 8; i++) {
    if (arr[i] < 0.5) arr[i] = 0;
    else arr[i] = 1;
  }
}

void char_to_float_bits(unsigned char c, float *dest_arr) {
    for (int i = 0; i < 8; i++) {
        // Extract the i-th bit (0 or 1) from MSB to LSB
        unsigned char bit = (c >> (7 - i)) & 1U;
        dest_arr[i] = (float)bit;
    }
}

unsigned char float_bits_to_char(const float *src_arr) {
    unsigned char result = 0;
    for (int i = 0; i < 8; i++) {
        // Round to the nearest integer to protect against precision loss
        int bit = (int)(src_arr[i] + 0.5f); 

        // Pack the bit back into its correct position
        result |= (bit & 1) << (7 - i);
    }
    return result;
}

unsigned char reverse_bits(unsigned char b) {
    unsigned char reversed = 0;
    for (int i = 0; i < 8; i++) {
        reversed = (reversed << 1) | (b & 1);
        b >>= 1;
    }
    return reversed;
}


/*
  * assume arr is normalized
*/

void print_float_arr_binary(float arr[8]) {
  for (int i = 0; i < 8; i++) {
    if (arr[i] != 0) {
      printf("1");
    }
    else {
      printf("0");
    }
  }
  printf("\n");
}

void print_byte_binary(unsigned char a) {
  float arr[8] = { 0 };
  char_to_float_bits(a, arr);
  print_float_arr_binary(arr);
}

int main() {
  srand(time(NULL));
  easy_net_t easy_net = { 0 };
  int layer_count = 3;
  int layer_sizes[] = {32, 32, 8};
  int input_layer_size = 8;
  activation_e activations[] = {LEAKY_RELU, LEAKY_RELU, IDENTITY};

  int epochs = 1000;
  int batch_size = 16;
  int thread_count = 4;

  create_easy_net(&easy_net, layer_count, layer_sizes, input_layer_size,
                  activations, SIGMOID_N, SIGMOID_N, IDENTITY,
                  BINARY_CROSS_ENTROPY, 0.08, 1);
  init_bias_easy(&easy_net, 0.1);
  init_weights_easy(&easy_net);

  create_easy_batch(&easy_net);
  prepare_threads_easy(&easy_net, thread_count);

  float **inputs = create_batch_arr(batch_size, input_layer_size);
  float **expecteds = create_batch_arr(batch_size, 8);

  for (int i = 0; i < epochs; i++) {
    for (int j = 0; j < batch_size; j++) {
      unsigned char c = (unsigned char) rand();
      unsigned char reverse = reverse_bits(c);

      char_to_float_bits(c, inputs[j]);
      char_to_float_bits(reverse, expecteds[j]);
    }
    train_batch_easy(&easy_net, inputs, expecteds, batch_size);
  }

  
  // -------------- testing ------------------
  int correct = 0;
  float test_inputs[8] = { 0 };

  for (int i = 0; i < 256; i++) {
    unsigned char c = (unsigned char) i;
    unsigned char reverse = reverse_bits(c);

    char_to_float_bits(c, test_inputs);

    float *outputs = input_to_output(easy_net.net, test_inputs);

    normalize_float_bits(outputs);

    unsigned char output_c = float_bits_to_char(outputs);

    printf("input: ");
    print_byte_binary(c);

    printf("expected output: ");
    print_byte_binary(reverse);

    printf("actual output: ");
    print_byte_binary(output_c);

    if (output_c == reverse) {
      printf("CORRECT!\n");
      correct++;
    }
    else {
      printf("INCORRECT\n");
    }
    free(outputs);
    outputs = NULL;
  }

  printf("scored %d / 256 = %f\n", correct, 100.0 * correct / 256.0);
  return 0;
}
