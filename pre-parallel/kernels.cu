#include "kernels.h"
#include <stdio.h>

__global__ void forward_layer(
    const float*  W,  // weight matrix [in_size * out_size]
    const float* x,  // input vector [in_size]
    const float* b,  // bias vector [out_size]
    float* h,        // output vector [out_size]
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


