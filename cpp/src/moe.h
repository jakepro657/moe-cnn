// moe.h

#ifndef MOE_H
#define MOE_H

#include <CL/cl.hpp>
#include <vector>
#include <string>

// Function to initialize OpenCL context, queue, and program
void initOpenCL(cl::Context &context, cl::CommandQueue &queue, cl::Program &program, const std::string &kernelSource);

// Function to execute the convolution layer kernel
void executeConvLayer(cl::CommandQueue &queue, cl::Kernel &kernel, cl::Buffer &inputBuffer, cl::Buffer &weightsBuffer,
                      cl::Buffer &outputBuffer, int in_channels, int out_channels, int input_size, int output_size);

// Function to execute the attention layer kernel
void executeAttentionLayer(cl::CommandQueue &queue, cl::Kernel &kernel, cl::Buffer &inputBuffer, cl::Buffer &queryWeightsBuffer,
                           cl::Buffer &keyWeightsBuffer, cl::Buffer &valueWeightsBuffer, cl::Buffer &outputBuffer,
                           int batch_size, int channels, int width, int height);

// Function to execute the pooling layer kernel
void executePoolingLayer(cl::CommandQueue &queue, cl::Kernel &kernel, cl::Buffer &inputBuffer, cl::Buffer &outputBuffer,
                         int input_size, int output_size);

#endif // MOE_H