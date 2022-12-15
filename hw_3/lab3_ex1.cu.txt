#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <random>
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
  
  //@@ Insert code below to allocate Host memory for input and output
  
  hostInput1 = (DataType*)malloc(inputLength*sizeof(*hostInput1));
  hostInput2 = (DataType*)malloc(inputLength*sizeof(*hostInput2));
  hostOutput = (DataType*)malloc(inputLength*sizeof(*hostOutput));
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  std::uniform_real_distribution<DataType> distribution(0.0, 1.0);
  std::default_random_engine gen(1145);
  for (DataType *ptr : {hostInput1, hostInput2}) {
      for (int i=0; i<inputLength; ++i){
          ptr[i] = distribution(gen);
      }
  }
  resultRef = (DataType*)malloc(inputLength*sizeof(DataType));
  double starttime=cpuSecond();
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
  cudaMemcpy(deviceInput1, hostInput1, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  double stoptime_memory=stopTimer(starttime_memory);

   int Db =1024;
   int Dg = (inputLength + Db - 1) / Db;
  //@@ Initialize the 1D grid and block dimensions here
 
  
  //@@ Launch the GPU Kernel here
  double starttime_gpu=cpuSecond();
  vecAdd<<<Dg,Db>>>(deviceInput1,deviceInput2,deviceOutput,inputLength);
  cudaDeviceSynchronize();
  double stoptime_gpu=stopTimer(starttime_gpu);
  double starttime_memory_2=cpuSecond();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, inputLength*sizeof(DataType), cudaMemcpyDeviceToHost);
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
  printf("cost Host->Device: %f - cost Device->Host: %f\n",stoptime_memory,stoptimer_memory_2);
  printf("CPU cost: %f - GPU cost: %f\n",stoptime_cpu,stoptime_gpu);
  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);


  //@@ Free the CPU memory here

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;


  return 0;
}