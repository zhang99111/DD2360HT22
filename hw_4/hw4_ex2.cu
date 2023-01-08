#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <random>
#include <cuda_profiler_api.h>
#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  int idx=0; 
  idx=threadIdx.x+blockDim.x * blockIdx.x;
  if(idx<len){
    out[idx]=in1[idx]+in2[idx];
  }

}
double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
//@@ Insert code to implement timer start

//@@ Insert code to implement timer stop

double stopTimer(double starttime) {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + ((double)tp.tv_usec*1.e-6)-starttime);
}
int main(int argc, char **argv) {
  
  int inputLength;

  int S_seg;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  
  inputLength=atoi(argv[1]);
  printf("The input length is %d\n", inputLength);
  S_seg=atoi(argv[2]);
  printf("the input segnebt lengrh is %d\n",S_seg);
  int seg_num=ceil(inputLength/S_seg);
  printf("the number of the segment is %d\n",seg_num);
  
  //@@ Insert code blow to allocate Host memory for input and output
  
  hostInput1 = (DataType*)malloc(inputLength*sizeof(*hostInput1));
  hostInput2 = (DataType*)malloc(inputLength*sizeof(*hostInput2));
  hostOutput = (DataType*)malloc(inputLength*sizeof(*hostOutput));
  resultRef = (DataType*)malloc(inputLength*sizeof(DataType));
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  std::uniform_real_distribution<DataType> distribution(0.0, 1.0);
  std::default_random_engine gen(1145);
  for (DataType *ptr : {hostInput1, hostInput2}) {
      for (int i=0; i<inputLength; ++i){
          ptr[i] = distribution(gen);
      }
  }
  
  double starttime=cpuSecond();
  cudaStream_t streams[seg_num]; 
  //cudaProfilerStart();
  for(int i = 0; i < seg_num; i++) {
    cudaStreamCreate(&streams[i]);
  }
  for(int i=0; i<inputLength; i++){
    resultRef[i]=hostInput1[i]+hostInput2[i];
  }
  double stoptime_cpu=stopTimer(starttime);

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength*sizeof(hostInput1));
  cudaMalloc(&deviceInput2, inputLength*sizeof(hostInput2));
  cudaMalloc(&deviceOutput, inputLength*sizeof(hostOutput));

  //@@ Insert code to below to Copy memory to the GPU here
  double starttime_memory=cpuSecond();
  //cudaMemcpy(deviceInput1, hostInput1, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);
  //cudaMemcpy(deviceInput2, hostInput2, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);

  //cudaDeviceSynchronize();
  double stoptime_memory=stopTimer(starttime_memory);

   int Db =1024;
   int Dg = S_seg / Db;
  //@@ Initialize the 1D grid and block dimensions here
  for(int i = 0; i < seg_num; i++) {
        int offset = i * S_seg;
        cudaMemcpyAsync(deviceInput1 + offset, hostInput1 + offset, S_seg * sizeof(DataType), 
                        cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(deviceInput2 + offset, hostInput2 + offset, S_seg * sizeof(DataType), 
                        cudaMemcpyHostToDevice, streams[i]); 
       
            vecAdd<<<Dg, Db, 0, streams[i]>>>(deviceInput1 + offset, deviceInput2 + offset, deviceOutput + offset, S_seg);
        cudaMemcpyAsync(hostOutput + offset, deviceOutput + offset, S_seg * sizeof(DataType), 
                        cudaMemcpyDeviceToHost, streams[i]);
         cudaStreamSynchronize(streams[i]);
    }
  
  //@@ Launch the GPU Kernel here
  double starttime_gpu=cpuSecond();
   for(int i = 0; i < seg_num; i++) {
        cudaStreamDestroy(streams[i]);
    }
  cudaDeviceSynchronize();
  double stoptime_gpu=stopTimer(starttime_gpu);
  double starttime_memory_2=cpuSecond();
  //cudaProfilerStop();
  //@@ Copy the GPU memory back to the CPU here
 //cudaMemcpy(hostOutput, deviceOutput, inputLength*sizeof(DataType), cudaMemcpyDeviceToHost); 
  cudaDeviceSynchronize();
  double stoptimer_memory_2=stopTimer(starttime_memory_2);

  
  //@@ Insert code below to compare the output with the reference
  for(int i=0;i<inputLength;i++)
  {
    if(resultRef[i] != hostOutput[i]&& abs(resultRef[i]-hostOutput[i])>0.0001){
      printf("error counted:%f",abs(resultRef[i]-hostOutput[i]));
      return 0;
    }
  }
  //printf("cost Host->Device: %f - cost Device->Host: %f\n",stoptime_memory,stoptimer_memory_2);
  //printf("CPU cost: %f - GPU cost: %f\n",stoptime_cpu,stoptime_gpu);
  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);


  //@@ Free the CPU memory here

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);
  return 0;



}