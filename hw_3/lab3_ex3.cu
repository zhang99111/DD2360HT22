#include <stdio.h>
#include <sys/time.h>
#include <random>
#define DataType double
#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics
__shared__ unsigned int sharedbins[NUM_BINS];
int idx = blockDim.x * blockIdx.x + threadIdx.x;
for (int i=threadIdx.x; i<num_bins; i+=blockDim.x) {
    if (i < num_bins) {
      sharedbins[i]=0;
    }
  } 
__syncthreads();
if (idx < num_elements) {
        atomicAdd(&(sharedbins[input[idx]]), 1);
  }
   __syncthreads(); 
for (int j=threadIdx.x; j<num_bins; j+=blockDim.x) {
    if (j < num_bins) {
        atomicAdd(&(bins[j]), sharedbins[j]);
      }
  }
  __syncthreads();
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127
const int bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin >= num_bins) return;

    if (bins[bin] > 127) {
        bins[bin] = 127;
    }

}


int main(int argc, char **argv) {
  
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  inputLength=atoi(argv[1]);
  printf("The input length is %d\n", inputLength);
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (unsigned int*)malloc(inputLength*sizeof(unsigned int));
  hostBins = (unsigned int*)malloc(NUM_BINS*sizeof(unsigned int));
  resultRef = (unsigned int*)malloc(NUM_BINS*sizeof(unsigned int));
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  std::uniform_real_distribution<DataType> distribution(0.0, NUM_BINS - 1);
  std::default_random_engine gen(1145);
  
    for (int i=0; i<inputLength; i++) {
       hostInput[i] = distribution(gen);
        //printf("the random in put is:%d ",hostInput[i]);
    }
  

  //@@ Insert code below to create reference result in CPU
  for (int i=0; i<NUM_BINS; i++) resultRef[i]=0;

    for (int i=0; i<inputLength; i++) {
        if(resultRef[hostInput[i]]<127){
            resultRef[hostInput[i]]++;
        }
    }

  //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
    cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

  //@@ Insert code to Copy memory to the GPU here
   cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));
  //@@ Initialize the grid and block dimensions here
   int Db_1 = 64;
   int Dg_1 = (inputLength + Db_1 - 1) / Db_1;

  //@@ Launch the GPU Kernel here
  histogram_kernel<<<Dg_1, Db_1>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

  //@@ Initialize the second grid and block dimensions here
   int Db_2 = 64;
   int Dg_2 = (NUM_BINS + Db_2 - 1) / Db_2;
  //@@ Launch the second GPU Kernel here
   convert_kernel<<<Dg_2, Db_2>>>(deviceBins, NUM_BINS);

  //@@ Copy the GPU memory back to the CPU here
   cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < NUM_BINS; ++i) {
        
        printf("resultRef[%d] = %d\n", i, resultRef[i]);
        printf("hostBins[%d] =  %d\n", i, hostBins[i]);
        
        if (hostBins[i] != resultRef[i]) {
            printf("CPU and GPU results are not equal.\n");
            continue;
            //break;
        }
        else{
          printf("CPU and GPU results are  equal.\n");
            continue;
        }
    }
    

   FILE *fptr;
    fptr = fopen("./histogram.txt","w+");
    if (fptr == NULL) {
        printf("Error!");   
        exit(1);             
    }
    for (int i = 0; i < NUM_BINS; ++i) {
        fprintf(fptr, "%d\n", hostBins[i]);
    }
    fclose(fptr);

   FILE *fp;

   fp = fopen("./test.txt", "w+");
   fprintf(fp, "This is testing for fprintf...\n");
   fprintf(fp, "This is testing for fprintf... %d \n", 10);
   fputs("This is testing for fputs...\n", fp);
   fclose(fp);

  //@@ Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceBins);

  //@@ Free the CPU memory here
   free(hostInput);
    free(hostBins);
    free(resultRef);

  return 0;
}