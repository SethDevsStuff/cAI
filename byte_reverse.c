#include "neuron.h"
#include "net.h"
#include "neural_functions.h"

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
  int layer_sizes[] = {32, 32, 8};
  float (*activations[])(float) = {&leaky_relu, &leaky_relu, &identity};
  float (*activations_d[])(float, float) = {&leaky_relu_d, &leaky_relu_d,
                                            &identity_d};

  float (*weights[])(int, int) = {&get_msra, &get_msra, &get_glorot};

  net_t *net = create_net(3, layer_sizes, activations, activations_d,
                          &identity,
                          8,
                          &sigmoid_normal, &sigmoid_normal,
                          &binary_cross_entropy,
                          &binary_cross_sigmoid_d,
                          0.05,
                          1);
  net_t *batch = create_batch_net(net);
  int batch_size = 4;

  init_bias(net, 0.1);
  init_weights(net, weights);

  float inputs[8] = { 0 };
  float expecteds[8] = { 0 };

  for (int i = 0; i < 30000; i++) {
    for (int j = 0; j < batch_size; j++) {
      unsigned char c = (unsigned char) rand();
      unsigned char reverse = reverse_bits(c);

      char_to_float_bits(c, inputs);
      char_to_float_bits(reverse, expecteds);


      input_in_net(net, inputs);
      calculate_hidden(net);
      push_to_output_training(net);
      float loss = calculate_total_loss(net, expecteds);
      full_back_prop(net, expecteds);
      //update_net(net);
      printf("loss: %f\n", loss);
      add_gradient_from(net, batch);
    }
    update_net_from_batch(net, batch, batch_size);
    reset_batch_values(batch);
  }


  // -------------- testing ------------------
  int correct = 0;

  for (int i = 0; i < 256; i++) {
    unsigned char c = (unsigned char) i;
    unsigned char reverse = reverse_bits(c);

    char_to_float_bits(c, inputs);

    float *outputs = input_to_output(net, inputs);

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
  }

  printf("scored %d / 256 = %f\n", correct, 100.0 * correct / 256.0);
  return 0;
}
