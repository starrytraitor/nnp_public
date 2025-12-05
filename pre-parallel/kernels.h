#include "config.h"

// Kernel function prototypes
__global__ void forward_layer(const float*, const float*, const float*, float*, int, int);
__global__ void relu_layer(float*, int);
__global__ void compute_delta(const float*, const float*, const float*, float*, int, int);
__global__ void update_weights(float*, const float*, const float*, float, int, int);
__global__ void update_bias(float*, const float*, float, int);

