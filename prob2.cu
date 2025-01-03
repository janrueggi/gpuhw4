#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define DataType float
#define TPB 128

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + offset;
    out[i] = in1[i] + in2[i];
}

void classic(int power, int S_seg){
    int inputLength = 2 ** power;

    DataType *hostInput1;
    DataType *hostInput2;
    DataType *hostOutput;
    DataType *resultRef;
    DataType *deviceInput1;
    DataType *deviceInput2;
    DataType *deviceOutput;
    
    hostInput1 = (DataType*) malloc(inputLength*sizeof(DataType));
    hostInput2 = (DataType*) malloc(inputLength*sizeof(DataType));
    hostOutput = (DataType*) malloc(inputLength*sizeof(DataType));
    cudaMalloc(&deviceInput1, inputLength*sizeof(DataType));
    cudaMalloc(&deviceInput2, inputLength*sizeof(DataType));
    cudaMalloc(&deviceOutput, inputLength*sizeof(DataType));

    srand(time(NULL));
    for(int i = 0; i < inputLength; i++){
        hostInput1[i] = (DataType) rand() / RAND_MAX;
        hostInput2[i] = (DataType) rand() / RAND_MAX;
    }

    double iStart = cpuSecond();

    cudaMemcpy(deviceInput1, hostInput1, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);
    vecAdd<<<(inputLength+TPB-1)/TPB,TPB>>>(deviceInput1, deviceInput2, deviceOutput, 0);
    cudaMemcpy(hostOutput, deviceOutput, inputLength*sizeof(DataType), cudaMemcpyDeviceToHost);

    double diff = cpuSecond() - iStart;
    printf("Power %02d: %f\n", power, diff);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
}

void improved(int power, int S_seg){
    int inputLength = 2 ** power;

    int nStreams = 4;

    DataType *hostInput1;
    DataType *hostInput2;
    DataType *hostOutput;
    DataType *deviceInput1;
    DataType *deviceInput2;
    DataType *deviceOutput;

    cudaMallocHost((void **)&hostInput1, inputLength*sizeof(DataType));
    cudaMallocHost((void **)&hostInput2, inputLength*sizeof(DataType));
    cudaMallocHost((void **)&hostOutput, inputLength*sizeof(DataType));
    cudaMalloc(&deviceInput1, inputLength*sizeof(DataType));
    cudaMalloc(&deviceInput2, inputLength*sizeof(DataType));
    cudaMalloc(&deviceOutput, inputLength*sizeof(DataType));

    srand(time(NULL));
    for (size_t i = 0; i < inputLength; i++) {
        hostInput1[i] = (float)rand() / RAND_MAX;
        hostInput2[i] = (float)rand() / RAND_MAX;
    }
    
    double iStart = cpuSecond();

    int N_seg = (inputLength + S_seg - 1) / S_seg;

    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; i++)
    {
        cudaStreamCreate(&stream[i]);
    }

    for (int i = 0; i < N_seg; i++)
    {
        int offset = i * S_seg;
        int idx = i % nStreams;
        cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], S_seg*sizeof(DataType), cudaMemcpyHostToDevice, stream[idx]);
        cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], S_seg*sizeof(DataType), cudaMemcpyHostToDevice, stream[idx]);
        vecAdd<<<S_seg/TPB, TPB, 0, stream[idx]>>>(deviceInput1, deviceInput2, deviceOutput, offset);
        cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], S_seg*sizeof(DataType), cudaMemcpyDeviceToHost, stream[idx]);
    }
    
    for (int i = 0; i < nStreams; i++)
    {
        cudaStreamDestroy(&stream[i]);
    }
    cudaDeviceSynchronize()
    double diff = cpuSecond() - iStart;
    printf("Power %02d, S_seg %05d: %f\n", power, S_seg, diff);

    cudaFreeHost(hostInput1);
    cudaFreeHost(hostInput2);
    cudaFreeHost(hostOutput);
    cudaFree(deviceInput1);
    cudaFree(deviceInput1);
    cudaFree(deviceOutput);
}

int main(int argc, char **argv) {

    int lengths[5] = {4, 8, 12, 16, 20};
    int S_seg = 2 ** 4;

    printf("CLassic:\n")
    for (int i = 0; i < 5; i++)
    {
        classic(lengths[i], S_seg);
    }
    printf("\n---\n")
    printf("Improved:")
    for (int i = 0; i < 5; i++)
    {
        improved(lengths[i], S_seg);
    }
}