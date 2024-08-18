#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include "moe.h"

// Utility function to load kernel source code
std::string loadKernelSource(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open kernel source file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

int main()
{
    // Load the kernel source code
    std::string kernelSource = loadKernelSource("moe.cl");

    // Initialize OpenCL context, command queue, and program
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    initOpenCL(context, queue, program, kernelSource);

    // Define model parameters
    const int in_channels = 3;
    const int out_channels = 64;
    const int input_size = 32;  // CIFAR-10 image size
    const int output_size = 30; // Example output size after convolution
    const int batch_size = 32;  // Example batch size

    // Create buffers for input, weights, and output
    cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * in_channels * input_size * input_size);
    cl::Buffer weightsBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * out_channels * in_channels * 9); // 3x3 kernel
    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * out_channels * output_size * output_size);

    // Initialize data (example data, replace with actual data loading)
    std::vector<float> inputData(in_channels * input_size * input_size, 1.0f);
    std::vector<float> weightsData(out_channels * in_channels * 9, 0.5f);
    std::vector<float> outputData(out_channels * output_size * output_size);

    // Write data to device buffers
    queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, sizeof(float) * inputData.size(), inputData.data());
    queue.enqueueWriteBuffer(weightsBuffer, CL_TRUE, 0, sizeof(float) * weightsData.size(), weightsData.data());

    // Set up and execute the convolution layer kernel
    cl::Kernel convKernel(program, "conv_layer");
    executeConvLayer(queue, convKernel, inputBuffer, weightsBuffer, outputBuffer, in_channels, out_channels, input_size, output_size);

    // Read back the results
    queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float) * outputData.size(), outputData.data());

    // Output the results (for verification)
    for (int i = 0; i < 10; ++i)
    { // Print first 10 elements as an example
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "FPGA MoE Model Execution Completed" << std::endl;
    return 0;
}