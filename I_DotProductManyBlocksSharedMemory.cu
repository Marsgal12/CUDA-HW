// Name: Marshall Galbreath
// Vector Dot product on many block and useing shared memory
// nvcc I_DotProductManyBlocksSharedMemory.cu -o temp
/*
 What to do:
 This code computes the dot product of vectors smaller than the block size.

 Your tasks:
 - Extend the code to launch as many blocks as needed based on a fixed thread count and the vector length.
 - Use **shared memory** within each block to speed up the computation.
 - Pad the input with zeros to fill the last block, if necessary.
 - Perform the final reduction (summing partial results) on the **CPU**.
 - Set the thread count (block size) to 256.
 - Test your code by setting N to different values.
*/

/*
 Purpose:
 To understand that blocks do **not** synchronize with each other during a kernel call.
 In other words, you can't detect when **all blocks** are finished from inside the kernel.
 You can work around this by exiting the kernel, which ensures all blocks have completed.
 Also to learn how to use shared memory to speed up your code.
*/

/*
 Explain what you did to fix the code:
 
 I changed the block size from 200 threads to 256 threads as required.
 I changed the grid size so that the program calculates how many blocks
 are needed to process the entire vector. This allows the program to work
 with vectors larger than one block.

 I added padding so that the last block is filled with zeros when N is not
 a multiple of 256. The extra values do not affect the dot product because
 multiplying by zero gives zero.

 I changed the GPU kernel to use shared memory. Each thread calculates one
 multiplication and stores its result in shared memory. The threads in each
 block then reduce those values until only one value remains.

 Each block stores its partial dot product in C_GPU. Since blocks cannot
 synchronize with each other inside the kernel, the kernel only calculates
 one partial result per block. After the kernel finishes, the CPU copies
 all of the partial results back and adds them together to get the final
 GPU dot product.

 I also removed the old check that stopped the program when N was larger
 than the block size. The program can now use multiple blocks.

 The code can now be tested with different values of N, including values
 smaller than 256, equal to 256, larger than 256, and values that are not
 evenly divisible by 256.
*/

/*
 Original comments were kept.
 Lines or sections that were changed have a comment explaining the change.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 25 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void CleanUp(); // CHANGED: Fixed capitalization so it matches the CleanUp function below.

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = 256; // CHANGED: The assignment requires a block size of 256 threads.
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	// CHANGED: Calculate how many blocks are needed to process all N values.
	// The + BlockSize.x - 1 makes sure that a partial last block gets included.
	GridSize.x = (N + BlockSize.x - 1) / BlockSize.x;
	GridSize.y = 1;
	GridSize.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// CHANGED: Calculate the number of elements needed after padding.
	// This makes the vector large enough to fill every block.
	int paddedN = GridSize.x * BlockSize.x;

	// Host "CPU" memory.				
	A_CPU = (float*)malloc(paddedN*sizeof(float)); // CHANGED: Allocate padded memory.
	B_CPU = (float*)malloc(paddedN*sizeof(float)); // CHANGED: Allocate padded memory.
	C_CPU = (float*)malloc(paddedN*sizeof(float)); // CHANGED: Allocate enough space for partial results.
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,paddedN*sizeof(float)); // CHANGED: Allocate padded GPU memory.
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,paddedN*sizeof(float)); // CHANGED: Allocate padded GPU memory.
	cudaErrorCheck(__FILE__, __LINE__);
	
	// CHANGED: C_GPU only needs one result for each block.
	cudaMalloc(&C_GPU,GridSize.x*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	int paddedN = GridSize.x * BlockSize.x; // CHANGED: Find the padded vector size.

	for(int i = 0; i < paddedN; i++) // CHANGED: Initialize the padded vector as well.
	{		
		if(i < N) // CHANGED: Put the original values in the vector.
		{
			A_CPU[i] = (float)i;	
			B_CPU[i] = (float)(3*i);
		}
		else
		{
			// CHANGED: Pad the rest of the vector with zeros.
			// These values will not change the dot product.
			A_CPU[i] = 0.0f;
			B_CPU[i] = 0.0f;
		}
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void dotProductCPU(float *a, float *b, float *C_CPU, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		C_CPU[id] = a[id] * b[id];
	}
	
	for(int id = 1; id < n; id++)
	{ 
		C_CPU[0] += C_CPU[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void dotProductGPU(float *a, float *b, float *c, int n)
{
	// CHANGED: Find the thread's position inside its block.
	int id = threadIdx.x;

	// CHANGED: Find the thread's position in the entire grid.
	int globalID = blockIdx.x * blockDim.x + threadIdx.x;

	// CHANGED: Create shared memory for all threads in the block.
	extern __shared__ float sharedData[];

	// CHANGED: Calculate the multiplication for this thread.
	// If this is a padded location, store zero instead.
	if(globalID < n)
	{
		sharedData[id] = a[globalID] * b[globalID];
	}
	else
	{
		sharedData[id] = 0.0f;
	}

	// CHANGED: Make sure every thread has finished writing to shared memory
	// before any thread starts reading from it.
	__syncthreads();

	// CHANGED: Reduce the values stored in shared memory.
	// Each step cuts the number of active values in half.
	int fold = blockDim.x;

	while(fold > 1)
	{
		int half = fold / 2;

		if(id < half)
		{
			sharedData[id] += sharedData[id + half];
		}

		// CHANGED: If the number of elements is odd, add the extra value.
		if((fold % 2 != 0) && id == 0)
		{
			sharedData[0] += sharedData[fold - 1];
		}

		// CHANGED: Synchronize before the next reduction step.
		__syncthreads();

		fold = half;
	}

	// CHANGED: Thread 0 stores this block's partial dot product.
	// Each block writes to a different location in C_GPU.
	if(id == 0)
	{
		c[blockIdx.x] = sharedData[0];
	}
}

// Checking to see if anything went wrong in the dot product.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	percentError = abs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\n percent error = %lf\n", percentError);
	
	if(percentError < Tolerance) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	//float localC_CPU, localC_GPU;
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// CHANGED: Removed the old if statement that stopped the program
	// when N was larger than the block size.
	// The program now supports multiple blocks.
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	int paddedN = GridSize.x * BlockSize.x; // CHANGED: Get the padded vector size.

	cudaMemcpyAsync(A_GPU, A_CPU, paddedN*sizeof(float), cudaMemcpyHostToDevice); // CHANGED: Copy padded vector.
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, paddedN*sizeof(float), cudaMemcpyHostToDevice); // CHANGED: Copy padded vector.
	cudaErrorCheck(__FILE__, __LINE__);
	
	// CHANGED: Launch enough blocks to process the entire vector.
	// Shared memory size is BlockSize.x floats.
	dotProductGPU<<<GridSize, BlockSize, BlockSize.x*sizeof(float)>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	// CHANGED: Copy one partial result from each block back to the CPU.
	cudaMemcpyAsync(C_CPU, C_GPU, GridSize.x*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// CHANGED: Make sure the GPU has finished before using the copied values.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

	// CHANGED: Perform the final reduction on the CPU.
	// Each element of C_CPU contains the partial result from one block.
	DotGPU = 0.0f;
	for(int i = 0; i < GridSize.x; i++)
	{
		DotGPU += C_CPU[i];
	}
	
	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU dot product.\n");
	}
	else
	{
		printf("\n\n You did a dot product correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}
