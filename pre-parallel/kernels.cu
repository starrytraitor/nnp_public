#include "kernels.h"
#include <stdio.h>

__global__ void matMul_forward(
    const float*  W,  // weights matrix
    const float* x,  // input vector
    const float* b,  // bias vector
    float* h,        // output vector
    int in_size,
    int out_size
){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("%.4f", in_size);
    if(j < out_size){
        float sum = b[j];
	//printf("%.4f", b[j]);
        for(int i = 0; i < in_size; i++){
            sum += x[i] * W[i*out_size + j];
        }
        h[j] = sum;
    }
}

__global__ void relu_k(float* h, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) h[idx] = h[idx] > 0 ? h[idx] : 0.0f;
}


__global__ void compute_delta(
    const float* W,      // weight matrix
    const float* delta_next, // delta of next layer
    const float* h,      // activation of current layer
    float* delta,        // output delta
    int in_size,
    int out_size
){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j < in_size){
        float sum = 0.0f;
        for(int k=0; k<out_size; k++){
            sum += delta_next[k] * W[j*out_size + k];
        }
        delta[j] = (h[j] > 0 ? 1.0f : 0.0f) * sum;  // drelu without the drelu
    }
}


__global__ void update_weights(
    float* W,
    const float* delta,
    const float* h_prev,
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


