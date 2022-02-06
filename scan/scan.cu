#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "CycleTimer.h"

extern float toBW(int bytes, float sec);

__global__ void upsweep_kernel(int N, int* data, int twod, int twod1) {
    // Compute overall index from position of thread in current block,
    // and given the block we are in
    int index = (blockIdx.x * blockDim.x + threadIdx.x)*twod1;

    if (index < N){
        data[index+twod1-1] += data[index+twod-1];
    }else{}
}

__global__ void downsweep_kernel(int N, int* data, int twod, int twod1) {
    // Compute overall index from position of thread in current block,
    // and given the block we are in
    int index = (blockIdx.x * blockDim.x + threadIdx.x)*twod1;

    if (index < N){
        int t = data[index+twod-1];
        data[index+twod-1] = data[index+twod1-1];

        // change twod1 below to twod to reverse prefix sum.
        data[index+twod1-1] += t;
    }else{}
}

__global__ void smol_kernel(int N, int *data) {
    // Compute overall index from position of thread in current block,
    // and given the block we are in
    data[N-1] = 0;
}

__global__ void peaks_kernel(int N, int *array, int *result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index > 0 && index < (N-1)) {
        if (array[index - 1] < array[index] && array[index] > array[index + 1]) {
            result[index] = 1;
        }
        else {
            result[index] = 0;
        }
    }else {
        result[index]=0;
    }
}

__global__ void result_kernel(int N, int *array, int *result ){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > 0 && index <(N-1)){
        if (array[index]!=array[index+1]){
            result[array[index]]=index;
        }
    }
}

/* Helper function to round up to a power of 2.
 */
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

void exclusive_scan(int *device_data, int length) {
    int N = nextPow2(length);

    // Compute number of blocks and threads per block
    const int threadsPerBlock = 512; 
    // upsweep phase.
    for (int twod = 1; twod < N; twod*=2) {
        int twod1 = twod*2;
        int blocks = ((N/twod1)+threadsPerBlock-1)/threadsPerBlock;
        upsweep_kernel<<<blocks, threadsPerBlock>>>(N, device_data, twod, twod1);  
        cudaDeviceSynchronize();          
    }

    smol_kernel<<<1,1>>>(N, device_data);
    cudaDeviceSynchronize();

    // downsweep phase.
    for (int twod = N/2; twod >= 1; twod /= 2) {
        int twod1 = twod*2;
        int blocks = ((N/twod1)+threadsPerBlock-1)/threadsPerBlock;
        downsweep_kernel<<<blocks, threadsPerBlock>>>(N, device_data, twod, twod1);
        cudaDeviceSynchronize();
    }
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int *inarray, int *end, int *resultarray) {
    int *device_data;
    // We round the array size up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness.
    // You may have an easier time in your implementation if you assume the
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_data, sizeof(int) * rounded_length);

    cudaMemcpy(device_data, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_data, end - inarray);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_data, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);
    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int *inarray, int *end, int *resultarray) {
    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

int find_peaks(int *device_input, int length, int *device_output) {
    /* TODO:
     * Finds all elements in the list that are greater than the elements before and after,
     * storing the index of the element into device_result.
     * Returns the number of peak elements found.
     * By definition, neither element 0 nor element length-1 is a peak.
     *
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if
     * it requires that. However, you must ensure that the results of
     * find_peaks are correct given the original length.
     */
    // Compute number of blocks and threads per block
    int rounded_length = nextPow2(length);
    const int threadsPerBlock = 512;
    const int blocks = (length+threadsPerBlock-1)/threadsPerBlock;

    int *result;
    cudaMalloc((void **)&result, rounded_length * sizeof(int));

    peaks_kernel<<<blocks, threadsPerBlock>>>(length, device_input, result);
    cudaDeviceSynchronize();

    
    exclusive_scan(result, length);
    cudaDeviceSynchronize();
    int totalCount;

    cudaMemcpy(&totalCount, &(result[length-1]), sizeof(int), cudaMemcpyDeviceToHost); 

    result_kernel<<<blocks, threadsPerBlock>>>(length,result,device_output);
    cudaDeviceSynchronize();
    cudaFree(result); 
    return totalCount;
}

/* Timing wrapper around find_peaks. You should not modify this function.
 */
double cudaFindPeaks(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);

    // std::cout << "Orignal array";
    // for (size_t i = 0; i < length; i++) {
    //     std::cout << input[i];
    // }std::cout << "\n";

    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    int result = find_peaks(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << "Output array";
    // for (size_t i = 0; i < result; i++) {
    //     std::cout << output[i];
    // }std::cout << "\n";
    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}

void printCudaInfo() {
    // For fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
