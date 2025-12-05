#include "kernels.h"
/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */


/*
__global__ void layer1_kernel(float* w1, float* x, float* b1, float* h1a){
    int j = blockIdx.x * blockDim.x + threadIdx.x; // neuron index
    if (j >= H1) return;

    float sum = b1[j];

    // matches: model->W1[i*H1 + j]
    for (int i = 0; i < SIZE; i++)
        sum += x[i] * w1[i * H1 + j];


    h1a[j] = sum > 0 ? sum : 0;
}
*/


/*
__global__ void matMul_forward(
    const float* w1,  // SIZE * H1
    const float* w2,  // H1 * H2
    const float* w3,  // H2 * CLASSES
    const float* b1,  // H1
    const float* b2,  // H2
    const float* b3,  // CLASSES
    const float* x,   // SIZE
    float* out        // CLASSES output
) {
    extern __shared__ float shmem[];

    float* h1_a = shmem;           // size H1
    float* h2_a = &shmem[H1];      // size H2

    int tid = threadIdx.x;

    // --- Step 1: compute h1a[j] for j < H1 ---
    if (tid < H1) {
        float sum = b1[tid];
        for (int i = 0; i < SIZE; i++) {
            sum += x[i] * w1[i * H1 + tid];
        }
        h1_a[tid] = sum > 0 ? sum : 0;
    }
    __syncthreads();

    // --- Step 2: compute h2a[j] for j < H2 ---
    if (tid < H2) {
        float sum = b2[tid];
        for (int i = 0; i < H1; i++) {
            sum += h1_a[i] * w2[i * H2 + tid];
        }
        h2_a[tid] = sum > 0 ? sum : 0;
    }
    __syncthreads();

    // --- Step 3: compute out[k] for k < CLASSES ---
    if (tid < CLASSES) {
        float sum = b3[tid];
        for (int j = 0; j < H2; j++) {
            sum += h2_a[j] * w3[j * CLASSES + tid];
        }
        out[tid] = sum;   // no relu on final layer
    }
}
*/


__global__ void forward_layer(
    const float* __restrict__ W,  // weight matrix [in_size * out_size]
    const float* __restrict__ x,  // input vector [in_size]
    const float* __restrict__ b,  // bias vector [out_size]
    float* __restrict__ h,        // output vector [out_size]
    int in_size,
    int out_size
){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j < out_size){
        float sum = b[j];
        for(int i = 0; i < in_size; i++){
            sum += x[i] * W[i*out_size + j];
        }
        h[j] = sum;
    }
}

__global__ void relu_layer(float* h, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) h[idx] = h[idx] > 0 ? h[idx] : 0.0f;
}


__global__ void compute_delta(
    const float* __restrict__ W,      // weight matrix of next layer [in_size*out_size]
    const float* __restrict__ delta_next, // delta of next layer [out_size]
    const float* __restrict__ h,      // activation of current layer [in_size]
    float* __restrict__ delta,        // output delta [in_size]
    int in_size,
    int out_size
){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j < in_size){
        float sum = 0.0f;
        for(int k=0; k<out_size; k++){
            sum += delta_next[k] * W[j*out_size + k];
        }
        delta[j] = (h[j] > 0 ? 1.0f : 0.0f) * sum;  // ReLU derivative
    }
}


__global__ void update_weights(
    float* __restrict__ W,
    const float* __restrict__ delta,
    const float* __restrict__ h_prev,
    float lr,
    int in_size,
    int out_size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = in_size*out_size;
    if(idx < total){
        int i = idx / out_size;
        int j = idx % out_size;
        W[i*out_size + j] += lr * delta[j] * h_prev[i];
    }
}


__global__ void update_bias(float* b, const float* delta, float lr, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        b[idx] += lr * delta[idx];
    }
}

