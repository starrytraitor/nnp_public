#include "config.h"

// Kernel function prototypes
__global__ void matMul_forward(const float*, const float*, const float*, float*, int, int);
__global__ void relu_k(float*, int);
__global__ void compute_delta(const float*, const float*, const float*, float*, int, int);
__global__ void update_weights(float*, const float*, const float*, float, int, int);
__global__ void update_bias(float*, const float*, float, int);


