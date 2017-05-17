#include "book.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// A(m*n) * B(n*l)
const int m = 1024;
const int n = 1024;
const int l = 1024;
const int TILE_WIDTH = 32;

const int sizeA = m*n*sizeof(int);
const int sizeB = n*l*sizeof(int);
const int sizeC = m*l*sizeof(int);

dim3 blocksPerGrid(m/TILE_WIDTH, l/TILE_WIDTH);
dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);

//dim3 blocksPerGrid((l+threadsPerBlock.x-1)/threadsPerBlock.x, 
//					(m+threadsPerBlock.y-1)/threadsPerBlock.y );

// define function
void PrintArray(int *A, int l);
void PrintMatrix(int *M, int row, int col);
int *MallocMatrix(int row, int col, int attr);
int *Transport(int* B, int row, int col);
int *MetrixMult(int *A, int *B);

__global__ void CudaMetrixMult(int *A, int *B, int *C)
{
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	int threadRow = threadIdx.y;
	int threadCol = threadIdx.x;

	__shared__ int subA_element[TILE_WIDTH][TILE_WIDTH];
	__shared__ int subB_element[TILE_WIDTH][TILE_WIDTH];

	int *subC = C + blockRow*m*TILE_WIDTH + blockCol*TILE_WIDTH;
	int sum = 0;

	for (int i = 0; i < n/TILE_WIDTH; ++i)
	{
		int *subA = A + blockRow*m*TILE_WIDTH + i*TILE_WIDTH;
		int *subB = B + i*m*TILE_WIDTH + blockCol*TILE_WIDTH;
		subA_element[threadRow][threadCol] = *(subA + threadRow*n + threadCol);
		subB_element[threadRow][threadCol] = *(subB + threadRow*n + threadCol);
		__syncthreads();

		for (int j = 0; j < TILE_WIDTH; ++j)
		{

			//int Aelement = *(subA + threadRow*n + j);
			//int Belement = *(subB + j*n + threadCol);
			//sum += Aelement*Belement;
			sum += subA_element[threadRow][j]*subB_element[j][threadCol];
		}
		__syncthreads();
	}
	*(subC + threadRow*n + threadCol) = sum;
}

int main(void)
{
	int *A, *B, *BT, *C, *CP;
	int *A_cuda, *B_cuda, *C_cuda;
	int i,check;
	double diff;
	struct timespec t_start, t_end;

	// 1 = random matrix, 0 = all zero matrix, -1 = only allocate memory
	A = MallocMatrix(m, n, 1);
	B = MallocMatrix(n, l, 1);
	CP = MallocMatrix(m, l, 0);

	// allocate device memory
	HANDLE_ERROR(cudaMalloc((void**)&A_cuda, sizeA));
	HANDLE_ERROR(cudaMalloc((void**)&B_cuda, sizeB));
	HANDLE_ERROR(cudaMalloc((void**)&C_cuda, sizeC));

	// multiply in CPU============================================================
	clock_gettime( CLOCK_REALTIME, &t_start);
	printf("Metrix multiply in sequential:\n");

	//C = MetrixMult(A, B);
	//PrintMatrix(C, m, l);

	clock_gettime( CLOCK_REALTIME, &t_end);
	diff = (t_end.tv_sec - t_start.tv_sec) * 1000.0;   
	diff += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;   
	printf("matrix mult in sequential elapsedTime: %lf ms\n", diff);

	// multiply in cuda shared memory==============================================
	clock_gettime( CLOCK_REALTIME, &t_start);
	printf("Metrix multiply with CUDA:\n");

	// copy A1 and B1 datas to device
	HANDLE_ERROR(cudaMemcpy(A_cuda, A, sizeA,
							cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(B_cuda, B, sizeB,
							cudaMemcpyHostToDevice)); 		 

	// kernal function
	CudaMetrixMult<<<blocksPerGrid, threadsPerBlock>>>(A_cuda, B_cuda, C_cuda);

	// copy datas back to host from device
	HANDLE_ERROR(cudaMemcpy(CP, C_cuda, sizeC,
							cudaMemcpyDeviceToHost)); 

	clock_gettime( CLOCK_REALTIME, &t_end);
	diff = (t_end.tv_sec - t_start.tv_sec) * 1000.0;   
	diff += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;   
	printf("matrix mult with CUDA elapsedTime: %lf ms\n", diff);

	// free device's memory
	cudaFree(A_cuda);  cudaFree(B_cuda);  cudaFree(C_cuda);

	//PrintMatrix(CP, m, l);
	
	/*
	check = 0;
	for (i = 0; i < m*l; ++i)
	{
		if (C[i] != CP[i])
		{
			printf("err = CP[%d]\t", i);
			printf("correct = %d\t", C[i]);
			printf("err = %d\n", CP[i]);
			check = 1;
			//break;
		}
	}

	
	if (check == 1)
	{
		printf("Fail!!!\n");
	}
	else
	{
		printf("Sucess!!!\n");
	}
	*/

	return 0;
}

void PrintArray(int *A, int l)
{
	int i;
	for (i = 0; i < l; ++i)
	{
		printf("%d\t", A[i]);
	}
}

void PrintMatrix(int *M, int row, int col)
{
	int i,j;
	for (i = 0; i < row; ++i)
	{
		for (j = 0; j < col; ++j)
		{
			printf("%d\t", M[i*col + j]);
		}
		printf("\n");
	}
	printf("\n");
}

// attr = 1:random; 0:all zero; -1:only allocate memory;
int* MallocMatrix(int row, int col, int attr)
{
	int *A;
	int i,j;

	A = (int*)malloc( row*col*sizeof(int) );

	if (attr == 1)
	{
		for (i = 0; i < row; ++i)
		{
			for (j = 0; j < col; ++j)
			{
				A[i*col + j] = rand()%10;
				//A[i*col + j] = 1;
			}
		}
	}
	else if (attr == 0)
	{
		for (i = 0; i < row; ++i)
		{
			for (j = 0; j < col; ++j)
			{
				A[i*col + j] = 0;
			}
		}
	}
	return A;
}

// matrix transport
int *Transport(int* B, int row, int col)
{
	int *BT;
	int i,j;

	BT = MallocMatrix(col, row, -1);

	for (i = 0; i < row; ++i)
	{
		for (j = 0; j < col; ++j)
		{
			BT[j*row + i] = B[i*col + j];
		}
	}
	return BT;
}

int* MetrixMult(int *A, int *B)
{
	int *C;
	int i,j,k;

	C = MallocMatrix(m, l, 0);

	for (i = 0; i < m; ++i)
	{
		for (j = 0; j < l; ++j)
		{
			for (k = 0; k < n; ++k)
			{
				C[i*l + j] += A[i*m + k] * B[k*l + j];
			}
		}
	}
	return C;
}

