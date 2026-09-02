// Name: Marshall Galbreath
// Vector addition on the GPU of any size with fixed block and grid size also adding pragma unroll for speed up.
// nvcc D_VectorAddFixedBlocksAndGrid.cu -o temp
/*
 What to do:
 This code works well for adding vectors with fixed-size blocks. 
 Given the size of the vector it needs to add, it takes a set block size, determines how 
 many blocks are needed, and creates a grid large enough to complete the task. Cool, cool!
 
 But—and this is a big but—this can get you into trouble because there is a limited number 
 of blocks you can use. Though large, it is still finite. Therefore, we need to write the 
 code in such a way that we don't have to worry about this limit. Additionally, some block 
 and grid sizes work better than others, which we will explore when we look at the streaming 
 multiprocessors.
 
 Extend this code so that, given a block size and a grid size, it can handle any vector addition. 
 Start by hard-coding the block size to 256 and the grid size to 64. Then, experiment with different 
 block and grid sizes to see if you can achieve any speedup. Set the vector size to a very large value 
 for time testing.

 You’ve probably already noticed that the GPU doesn’t significantly outperform the CPU. This is because 
 we’re not asking the GPU to do much work, and the overhead of setting up the GPU eliminates much of the 
 potential speedup. 
 
 To address this, modify the computation so that:
 c = sqrt(cos(a)*cos(a) + a*a + sin(a)*sin(a) - 1.0) + sqrt(cos(b)*cos(b) + b*b + sin(b)*sin(b) - 1.0)
 Hopefully, this is just a convoluted and computationally expensive way to calculate a + b.
 If the compiler doesn't recognize the simplification and optimize away all the unnecessary work, this 
 should create enough computational workload for the GPU to outperform the CPU.

 Write the loop as a for loop rather than a while loop. This will allow you to also use #pragma unroll 
 to explore whether it provides any speedup. Make sure to include an if (id < n) condition in your code 
 to ensure safety. Finally, be prepared to discuss the impact of #pragma unroll and whether it helped 
 improve performance.
*/

/*
 Purpose:
 1. To learn how to stride through a vector of any size and add it on the GPU.
 2. To learn how to use #pragma unroll
*/

/*
 Explain what you did to fix the code:

 I changed the grid size so that it is fixed at 64 blocks instead of
 calculating the number of blocks from the size of the vector.

 Since the grid is fixed, there may not be enough threads for every
 element of a very large vector. To solve this, I use a grid-stride
 loop. Each thread processes multiple elements, and the distance
 between elements is the total number of threads in the grid.

 I also changed the calculation so that it performs the more
 computationally expensive sin(), cos(), and sqrt() operations
 requested in the assignment.

 I changed the loop to a for loop and added #pragma unroll 4.
 This gives the compiler permission to unroll the loop four times,
 which may reduce loop overhead and potentially improve performance.

 The if (id < n) condition makes sure that a thread never attempts
 to access an element outside of the vector.

 I also changed the CPU calculation to use the same expensive
 calculation as the GPU so that the CPU and GPU are performing
 equivalent work.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Defines

// CHANGED The vector is made very large so there is enough work to make
// performance testing between the CPU and GPU meaningful.
#define N 100000000

// CHANGED The block size is fixed at 256 as required by the assignment.
#define BLOCK_SIZE 256

// CHANGED The grid size is fixed at 64 blocks as required by the assignment.
#define GRID_SIZE 64

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void addVectorsCPU(float*, float*, float*, int);

// CHANGED The original prototype had the wrong arguments.
// The kernel actually receives pointers to A, B, C and n.
__global__ void addVectorsGPU(float*, float*, float*, int);

bool check(float*, int, float);
long elaspedTime(struct timeval, struct timeval);
void CleanUp();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t error;

	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n",
		       cudaGetErrorString(error), file, line);
		exit(0);
	}
}

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;

	// CHANGED The grid is now fixed at 64 blocks.
	// Originally, the number of blocks was calculated from N.
	// That meant a very large vector could require an extremely
	// large number of blocks.
	// By fixing the grid size, we avoid depending on the maximum
	// number of blocks that can be launched.
	GridSize.x = GRID_SIZE;

	GridSize.y = 1;
	GridSize.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory()
{
	// Host "CPU" memory.
	// These arrays store the vectors while they are on the CPU.
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));

	// Device "GPU" Memory
	// These arrays store the vectors while they are on the GPU.
	cudaMalloc((void**)&A_GPU, N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	cudaMalloc((void**)&B_GPU, N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	cudaMalloc((void**)&C_GPU, N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	// Fill A and B with predictable values.
	// A[i] = i
	// B[i] = 2*i
	// Therefore, the expected answer is:
	// C[i] = A[i] + B[i]
	//      = i + 2*i
	//      = 3*i

	for(int i = 0; i < N; i++)
	{
		A_CPU[i] = (float)i;
		B_CPU[i] = (float)(2*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	// CHANGED The CPU performs the same computationally expensive
	// calculation as the GPU.
	// This gives us a fairer CPU vs. GPU comparison because
	// both processors are doing the same mathematical work.

	for(int id = 0; id < n; id++)
	{
		c[id] =
			sqrtf(
				cosf(a[id]) * cosf(a[id])
				+ a[id] * a[id]
				+ sinf(a[id]) * sinf(a[id])
				- 1.0f
			)
			+
			sqrtf(
				cosf(b[id]) * cosf(b[id])
				+ b[id] * b[id]
				+ sinf(b[id]) * sinf(b[id])
				- 1.0f
			);
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
{
	// Calculate the global ID of the current thread.
	// blockIdx.x tells us which block this thread belongs to.
	// blockDim.x tells us how many threads are in each block.
	// threadIdx.x tells us the thread's position inside its block.
	// For example, with 256 threads per block:
	// Block 0 handles IDs 0-255.
	// Block 1 handles IDs 256-511.
	// etc.
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// CHANGED Calculate the total number of threads in the entire grid.
	// With:
	// 64 blocks * 256 threads = 16,384 threads
	// Since our vector can contain millions of elements, each
	// thread will need to process more than one element.
	int stride = blockDim.x * gridDim.x;

	// CHANGED The grid-stride loop allows the fixed number of GPU threads
	// to process a vector of any size.
	// Instead of each thread processing only one element, a thread
	// processes:
	// id
	// id + stride
	// id + 2*stride
	// id + 3*stride
	// ...
	// This means that even though we only have 16,384 threads,
	// they can process a vector containing 100,000,000 elements.
	// #pragma unroll 4 tells the compiler that it can unroll this
	// loop by a factor of 4. This can reduce the overhead caused
	// by repeatedly checking and incrementing the loop.
	#pragma unroll 4
	for(int id = blockIdx.x * blockDim.x + threadIdx.x;
	    id < n;
	    id += stride)
	{
		// Make sure the current thread is accessing a valid
		// element of the vector.
		// The loop condition already guarantees this, but the
		// assignment specifically asks for an if (id < n)
		// condition for safety.
		if(id < n)
		{
			// Calculate the expensive mathematical expression
			// for vector A.
			// Mathematically:
			// cos²(a) + sin²(a) = 1
			// Therefore:
			// sqrt(cos²(a) + a² + sin²(a) - 1)
			// = sqrt(a²)
			// = |a|
			// Since our values are non-negative, this equals a.
			float aResult =
				sqrtf(
					cosf(a[id]) * cosf(a[id])
					+ a[id] * a[id]
					+ sinf(a[id]) * sinf(a[id])
					- 1.0f
				);

			// Calculate the same expensive mathematical expression
			// for vector B.
			float bResult =
				sqrtf(
					cosf(b[id]) * cosf(b[id])
					+ b[id] * b[id]
					+ sinf(b[id]) * sinf(b[id])
					- 1.0f
				);

			// Add the two calculated values and store the result
			// in the output vector C.
			c[id] = aResult + bResult;
		}
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float *c, int n, float tolerence)
{
	int id;
	double myAnswer;
	double trueAnswer;
	double percentError;

	// Needed the -1 because we start at 0.
	double m = n-1;

	myAnswer = 0.0;

	// Add all the values in C together.
	// This gives us one value that we can compare against
	// the mathematically expected answer.
	for(id = 0; id < n; id++)
	{
		myAnswer += c[id];
	}

	// Since:
	// A[i] = i
	// B[i] = 2*i
	// The expected result is:
	// C[i] = 3*i
	// The sum of 3*i from 0 to n-1 is:
	// 3 * (n-1)n / 2
	trueAnswer = 3.0*(m*(m+1))/2.0;

	// Calculate the percentage difference between the GPU
	// answer and the expected answer.
	percentError =
		fabs((myAnswer - trueAnswer)/trueAnswer)*100.0;

	// Check whether the error is smaller than our tolerance.
	if(percentError < tolerence)
	{
		return(true);
	}
	else
	{
		printf("\n Percent error = %f%%\n", percentError);
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.

	long startTime =
		start.tv_sec * 1000000L + start.tv_usec; // In microseconds.

	long endTime =
		end.tv_sec * 1000000L + end.tv_usec; // In microseconds

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

	// Free divice "GPU" memory.
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
	// This sets the block size to 256 and the grid size to 64.
	setUpDevices();

	// Print the configuration being used.
	// This makes it easy to see which block and grid sizes
	// were used during performance testing.
	printf("\nVector size: %d\n", N);
	printf("Block size: %d\n", BLOCK_SIZE);
	printf("Grid size: %d blocks\n", GRID_SIZE);
	printf("Total GPU threads: %d\n",
	       BLOCK_SIZE * GRID_SIZE);

	// Allocating the memory you will need.
	allocateMemory();

	// Putting values in the vectors.
	innitialize();

	// Adding on the CPU
	// Start the CPU timer before performing the calculation.
	gettimeofday(&start, NULL);

	addVectorsCPU(A_CPU, B_CPU, C_CPU, N);

	// Stop the CPU timer after the calculation is complete.
	gettimeofday(&end, NULL);

	timeCPU = elaspedTime(start, end);

	// Zeroing out the C_CPU vector just to be safe because right now it has the correct answer in it.
	for(int id = 0; id < N; id++)
	{
		C_CPU[id] = 0.0;
	}

	// Adding on the GPU
	// Start timing before copying the data to the GPU.
	gettimeofday(&start, NULL);

	// Copy Memory from CPU to GPU
	cudaMemcpy(A_GPU,
	           A_CPU,
	           N*sizeof(float),
	           cudaMemcpyHostToDevice);

	cudaErrorCheck(__FILE__, __LINE__);

	cudaMemcpy(B_GPU,
	           B_CPU,
	           N*sizeof(float),
	           cudaMemcpyHostToDevice);

	cudaErrorCheck(__FILE__, __LINE__);

	// CHANGED Launch the kernel using the fixed grid and block sizes.
	// GridSize = 64 blocks
	// BlockSize = 256 threads per block
	// The grid-stride loop inside the kernel allows these
	// 16,384 threads to process the entire vector.
	addVectorsGPU<<<GridSize,BlockSize>>>(
		A_GPU,
		B_GPU,
		C_GPU,
		N
	);

	cudaErrorCheck(__FILE__, __LINE__);

	// Copy Memory from GPU to CPU
	cudaMemcpy(C_CPU,
	           C_GPU,
	           N*sizeof(float),
	           cudaMemcpyDeviceToHost);

	cudaErrorCheck(__FILE__, __LINE__);

	// Making sure the GPU and CPU wait until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

	// Stop the GPU timer after all GPU work and memory transfers
	// have completed.
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

		printf("\n The time it took on the CPU was %ld microseconds",
		       timeCPU);

		printf("\n The time it took on the GPU was %ld microseconds",
		       timeGPU);

		// Calculate how many times faster the CPU or GPU was.
		printf("\n CPU/GPU speedup was %.2fx",
		       (double)timeCPU / (double)timeGPU);
	}

	// Your done so cleanup your room.
	CleanUp();

	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");

	return(0);
}
