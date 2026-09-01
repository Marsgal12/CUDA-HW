// Name: Marshall Galbreath
// Vector addition on the GPU, with many block
// nvcc C_VectorAddGPUManyBlocks.cu -o temp

/*
 What to do:
 This code adds two vectors on the GPU using a single block.
 But why use just one block?
 We have thousands of CUDA cores, so we should use many blocks to keep the SMs (Streaming Multiprocessors) on the GPU busy.

 Extend this code so that, given a block size, it will set the grid size to handle "almost" any vector addition.
 I say "almost" because there is a limit to how many blocks you can use, but this number is very large. 
 We will address this limitation in the next HW.

 Hard-code the block size to 256.
 
 CUDA error checking has been added to your code go through it and understand it. 
 Then, when you encounter issues like those in your last HW, CUDA will provide clues about what went wrong.
*/

/*
 Purpose:
 To learn how to use more than one block.
*/

/*
 Explain what you did to fix the code:
 
 I changed the block size to 256 threads.
 I changed the grid size so that the number of blocks is calculated
 based on the vector size and block size.
 I also changed the thread ID in the GPU kernel so that every thread
 gets a unique ID across all of the blocks.
 Finally, I added a check to make sure threads do not access elements
 outside the size of the vector.
*/


// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 11503 // Length of the vector

// Global variables
// These point to memory on the CPU (host).
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
// These point to memory on the GPU (device)
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
// Stores the number of threads in each block.
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
// Stores the number of blocks in the grid.
dim3 GridSize; //This variable will hold the Dimensions of your grid
// Maximum amount of error we will allow when checking the answer
float Tolerance = 0.01;

// Function prototypes
// Checks for CUDA errors.
void cudaErrorCheck(const char *, int);
// Sets the block and grid sizes.
void setUpDevices();
// Allocates CPU and GPU memory.
void allocateMemory();
// Puts values into the input vectors.
void innitialize();
// Adds two vectors together on the CPU.
void addVectorsCPU(float*, float*, float*, int);

// CHANGED The original prototype had float instead of float*.
// The kernel uses pointers because A, B, and C are arrays.
// Adds two vectors together on the GPU.
__global__ void addVectorsGPU(float*, float*, float*, int);
// Checks whether the GPU answer is correct.
bool  check(float*, int);
// Calculates elapsed time.
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

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
	// CHANGED: The assignment says to hard-code the block size to 256.
	// Each block will contain 256 threads.
	BlockSize.x = 256;

	// These remain 1 because we are only using the x dimension.
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	// CHANGED: Calculate the number of blocks needed to process
	// all N elements of the vector.
	//
	// (N + BlockSize.x - 1) / BlockSize.x rounds up.
	//
	// For this assignment:
	// (11503 + 256 - 1) / 256 = 45 blocks.
	//
	// 45 blocks * 256 threads = 11520 threads.
	// This gives us enough threads for all 11503 elements.
	GridSize.x = (N + BlockSize.x - 1) / BlockSize.x;

	// These remain 1 because we are only using the x dimension.
	GridSize.y = 1;
	GridSize.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.			
// Allocate enough memory for N floating-point numbers.
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaMalloc(&C_GPU,N*sizeof(float));

}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		c[id] = a[id] + b[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
{
	// CHANGED: threadIdx.x only gives the thread's ID inside
	// its current block.
	//
	// blockIdx.x tells us which block the thread belongs to.
	// blockDim.x tells us how many threads are in each block.
	//
	// This calculation gives every thread in the entire grid
	// a unique global ID.
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	// CHANGED: Since we rounded the number of blocks up,
	// there are a few extra threads.
	//
	// 45 blocks * 256 threads = 11520 threads
	// N = 11503 elements
	//
	// Therefore, the last 17 threads would be outside the
	// vector. This check prevents them from accessing
	// memory outside the array.
	if(id < n)
	{
		c[id] = a[id] + b[id];
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float *c, int n, float tolerence)
{
	int id;
	double myAnswer;
	double trueAnswer;
	double percentError;
	double m = n-1; // Needed the -1 because we start at 0.
	
	myAnswer = 0.0;
	for(id = 0; id < n; id++)
	{ 
		myAnswer += c[id];
	}
	
	trueAnswer = 3.0*(m*(m+1))/2.0;
	
	percentError = abs((myAnswer - trueAnswer)/trueAnswer)*100.0;
	
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
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Zeroing out the C_CPU vector just to be safe because right now it has the correct answer in it.
	for(int id = 0; id < N; id++)
	{ 
		C_CPU[id] = 0.0;
	}
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// CHANGED: The kernel now launches multiple blocks.
	// GridSize.x contains 45 blocks and BlockSize.x contains
	// 256 threads per block.
	addVectorsGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU ,C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the GPU and CPU wait until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU vector addition\n");
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}
