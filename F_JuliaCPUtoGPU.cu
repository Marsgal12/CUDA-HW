// Name: Marshall Galbreath
// Simple Julia CPU.
// nvcc F_JuliaCPUtoGPU.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
 Use __device__ for the escapeOrNotColor function
*/

/*
 Purpose:
 To apply your new GPU skills to do  something cool!
*/

/*
 Explain what you did to fix the code:
 
 I changed the program from using the CPU to using the GPU to create the
 Julia fractal. I changed the escapeOrNotColor function to a __device__
 function so that the GPU could use it. I also created a CUDA kernel called
 juliaKernel that allows each GPU thread to calculate the color of one pixel
 at the same time.

 I added GPU memory using cudaMalloc and copied the completed pixel data back
 to the CPU using cudaMemcpy. I also added CUDA blocks and threads so that the
 GPU can calculate many pixels in parallel instead of calculating them one at
 a time with the CPU loops.

 The window is still 1024 by 1024, and the original red, green, and blue pixel
 setup was kept. I also added CUDA error checking and synchronization to make
 sure the GPU finishes its calculations before the pixels are displayed.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>

// CHANGED: Added CUDA runtime functions so we can use GPU memory and kernels.
#include <cuda_runtime.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);

// CHANGED: Added __device__ so the GPU can use this function.
__device__ float escapeOrNotColor(float, float);

// CHANGED: Added GPU kernel to calculate the fractal pixels.
__global__ void juliaKernel(float*, float, float, float, float,
                            float, float, unsigned int, unsigned int);

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

// CHANGED: Added __device__ so this function runs on the GPU.
__device__ float escapeOrNotColor (float x, float y) 
{
	float mag,tempX;
	int count;
	
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	
	count = 0;
	mag = sqrt(x*x + y*y);;
	while (mag < maxMag && count < maxCount) 
	{	
		tempX = x; //We will be changing the x but we need its old value to find y.
		x = x*x - y*y + A;
		y = (2.0 * tempX * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		return(0.0);
	}
	else
	{
		return(1.0);
	}
}

// CHANGED: Added GPU kernel. Each GPU thread calculates one pixel.
__global__ void juliaKernel(float *pixels, float XMin, float XMax,
                            float YMin, float YMax, float stepSizeX,
                            float stepSizeY, unsigned int width,
                            unsigned int height)
{
	// CHANGED: Find which pixel this GPU thread will calculate.
	int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
	int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

	// CHANGED: Make sure the thread does not go outside the image.
	if(pixelX >= width || pixelY >= height)
	{
		return;
	}

	// CHANGED: Convert the pixel location into x and y coordinates.
	float x = XMin + pixelX * stepSizeX;
	float y = YMin + pixelY * stepSizeY;

	// CHANGED: Find the location of this pixel in the RGB array.
	int k = (pixelY * width + pixelX) * 3;

	// CHANGED: Calculate the red color using the GPU.
	pixels[k] = escapeOrNotColor(x, y);

	// CHANGED: Keep green and blue turned off.
	pixels[k+1] = 0.0;
	pixels[k+2] = 0.0;
}

void display(void) 
{ 
	float *pixels; 
	
	// CHANGED: Added pointer for memory on the GPU.
	float *devicePixels;
	
	float stepSizeX, stepSizeY;
	
	//We need the 3 because each pixel has a red, green, and blue value.
	pixels = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	
	stepSizeX = (XMax - XMin)/((float)WindowWidth);
	stepSizeY = (YMax - YMin)/((float)WindowHeight);
	
	// CHANGED: Allocate memory on the GPU for the pixels.
	cudaMalloc((void **)&devicePixels,
	           WindowWidth*WindowHeight*3*sizeof(float));
	
	// CHANGED: Set the number of threads in each GPU block.
	dim3 threadsPerBlock(16, 16);
	
	// CHANGED: Calculate the number of blocks needed for the image.
	dim3 numberOfBlocks(
		(WindowWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(WindowHeight + threadsPerBlock.y - 1) / threadsPerBlock.y
	);
	
	// CHANGED: Run the Julia fractal calculation on the GPU.
	juliaKernel<<<numberOfBlocks, threadsPerBlock>>>(
		devicePixels, XMin, XMax, YMin, YMax,
		stepSizeX, stepSizeY, WindowWidth, WindowHeight
	);
	
	// CHANGED: Check for errors after starting the GPU kernel.
	cudaErrorCheck(__FILE__, __LINE__);
	
	// CHANGED: Wait for the GPU to finish calculating.
	cudaDeviceSynchronize();
	
	// CHANGED: Copy the pixels from GPU memory back to CPU memory.
	cudaMemcpy(pixels, devicePixels,
	           WindowWidth*WindowHeight*3*sizeof(float),
	           cudaMemcpyDeviceToHost);
	
	// CHANGED: Check for errors after copying the pixels.
	cudaErrorCheck(__FILE__, __LINE__);

	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixels); 
	glFlush(); 
	
	// CHANGED: Free the GPU memory.
	cudaFree(devicePixels);
	
	// CHANGED: Free the CPU memory.
	free(pixels);
}

int main(int argc, char** argv)
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();
}
