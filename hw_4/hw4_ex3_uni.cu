#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <random>
#define DataType double
// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  if ((col >= numBColumns) || (row >= numARows)) return;
  double sum=0.0;
  for (int k = 0; k < numAColumns; k++) {
        sum += A[row*numAColumns + k] * B[k*numBColumns + col];
    }
    C[row*numBColumns+col]=sum;
}

int main(int argc, char **argv) {
    
  DataType *uniA; // The A matrix
  DataType *uniB; // The B matrix
  DataType *uniC; // The output C matrix
  DataType *resultRef; // The reference result
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
   numARows = atoi(argv[1]);
   numAColumns = atoi(argv[2]);
   numBRows = atoi(argv[3]);
   numBColumns = atoi(argv[4]);
   numCRows = numARows;
   numCColumns = numBColumns;
  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaMallocManaged((void**)&uniA, sizeof(DataType) * numARows * numAColumns);
  cudaMallocManaged((void**)&uniB, sizeof(DataType) * numBRows * numBColumns);
  cudaMallocManaged((void**)&uniC, sizeof(DataType) * numCRows * numCColumns);
  resultRef = (DataType*) malloc(numCRows * numCColumns * sizeof(DataType));
  if (numAColumns != numBRows) {
        printf("ERROR: the matrix could not be multiplied" );
        return 0;
    }

  //@@ Insert code below to allocate Host memory for input and output
  //hostA = (DataType*)malloc(numARows * numAColumns * sizeof(DataType));
  //hostB = (DataType*)malloc(numBRows * numBColumns * sizeof(DataType));
  //hostC = (DataType*)malloc(numCRows * numCColumns * sizeof(DataType));
  //resultRef = (DataType*)malloc(numCRows * numCColumns * sizeof(DataType));
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  std::uniform_real_distribution<DataType> distribution(0.0, 1.0);
  std::default_random_engine gen(1000);
  for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numAColumns; j++) {
            DataType randomNumber = distribution(gen);
            hostA[i*numAColumns + j] = randomNumber;
            
        }
    }
    for (int i = 0; i < numBRows; i++) {
        for (int j = 0; j < numBColumns; j++) {
            DataType randomNumber = distribution(gen); 
            hostB[i*numBColumns + j] = randomNumber;
            
        }
    }
  ;
  for(int i=0; i<numARows;i++){
    for(int j=0;j<numBColumns;j++){
       resultRef[i*numBColumns+j]=0.0;
      for(int k=0;k<numBRows;k++){
        resultRef[i*numBColumns+j]+=hostA[k+numBRows*i]*hostB[j+numBColumns*k];
      }
      
    }
  }
  //@@ Insert code below to allocate GPU memory here
  //cudaMalloc(&deviceA, numARows * numAColumns * sizeof(DataType));
  //cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(DataType));
  //cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(DataType));
  
  //@@ Insert code to below to Copy memory to the GPU here
  
  //cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  //cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  
  
  int Dg_x=(numCColumns+32-1)/32;
  int Dg_y=(numCRows+32-1)/32;
  //@@ Initialize the grid and block dimensions here
  dim3 Dg(Dg_x,Dg_y,1);
  dim3 Db(32,32,1);

  //@@ Launch the GPU Kernel here
 
  gemm<<<Dg,Db>>>(deviceA,deviceB,deviceC,numARows,numAColumns,numBRows,numBColumns);
  cudaDeviceSynchronize();
  

  //@@ Copy the GPU memory back to the CPU here
  
  //cudaMemcpy(hostC, deviceC,  numCRows * numCColumns *sizeof(DataType), cudaMemcpyDeviceToHost);
  
  //@@ Insert code below to compare the output with the reference
  bool judge = 1;
  for(int i=0;i<numCRows * numCColumns;i++){
    if(hostC[i]!=resultRef[i] && abs(resultRef[i]-hostC[i])>0.01 ){
      
      judge=0;
      break;
    }
  }
  judge?printf("the commdan is correct"):printf("error");;
  //@@ Free the GPU memory here
  cudaFree(uniA);
  cudaFree(uniB);
  cudaFree(uniC);

  //@@ Free the CPU memory here
  //free(hostA);
  //free(hostB);
  //free(hostC);
  free(resultRef);
  return 0;
}