// Name: Marshall Galbreath
// Not simple Julia Set on the GPU
// nvcc G_JuliaExtended.cu -o temp -lglut -lGL

/*
 What to do:
 This code displays a simple Julia set fractal using the GPU.
 However, it currently only runs on a 1024x1024 window.

 Your tasks:
 - Modify the code so it works on any given window size. 
   I will pick these on the fly unsigned int WindowWidth, WindowHeight; 
   float XMin, XMax, YMin, YMax; and your code should work. You will be graded on this.
   
 - But you can set these values to whatever you want for the art compitition.
 - Add color to the fractal — be creative! You will be judged on your artistic flair.
 - Don't cut off your ear or anything, but try to make Vincent wish he'd had a GPU.
 - This is a competition with a prize!!!
*/

/*
 Purpose:
 To have some fun with your new GPU skills!
*/

/*
 Explain what you did to fix the code:
 
 I changed the GPU code so that it uses a 2D grid of threads instead of
 assuming that the window is always 1024x1024. Each GPU thread is now
 responsible for one pixel using both its x and y coordinates.

 I also changed the block and grid sizes so that the program calculates
 how many blocks are needed for whatever WindowWidth and WindowHeight
 values are being used. This allows the program to work with different
 window sizes instead of limiting the width to 1024 pixels.

 I changed the pixel location calculation so that the x and y coordinates
 are based on the current window size and the XMin, XMax, YMin, and YMax
 values.

 I also added color to the Julia fractal. The number of iterations is used
 to create different red, green, and blue values, which gives the fractal
 multiple colors instead of only red.
 
 // CHANGED: Explanation updated to describe the GPU and color changes.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>

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
__device__ float escapeOrNotColor (float, float);
__global__ void colorPixels(float*, float, float, float, float, int, int); // CHANGED: Added width and height arguments.
void display(void);

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
	
	// CHANGED: Instead of only returning 0 or 1, the iteration count
	// is returned so that the GPU can use it to create different colors.
	return((float)count);
}

__global__ void colorPixels(float *pixels, float xMin, float yMin, float dx, float dy, int width, int height) // CHANGED: Added width and height.
{
	float x,y;
	int id;
	int pixelX, pixelY; // ADDED: Variables for the pixel's x and y position.
	float count; // ADDED: Stores the number of iterations.
	float color; // ADDED: Used to create the color values.
	
	// ADDED: Each thread now gets both an x and y pixel coordinate.
	pixelX = threadIdx.x + blockDim.x * blockIdx.x;
	pixelY = threadIdx.y + blockDim.y * blockIdx.y;
	
	// ADDED: Threads outside the actual window are ignored.
	if(pixelX >= width || pixelY >= height)
	{
		return;
	}
	
	//Getting the offset into the pixel buffer. 
	//We need the 3 because each pixel has a red, green, and blue value.
	id = 3*(pixelX + width*pixelY); // CHANGED: Uses both x and y coordinates and the actual window width.
	
	//Asigning each thread its x and y value of its pixel.
	x = xMin + dx*pixelX; // CHANGED: Uses pixelX instead of threadIdx.x.
	y = yMin + dy*pixelY; // CHANGED: Uses pixelY instead of blockIdx.x.
	
	// CHANGED: Save the iteration count so it can be used for coloring.
	count = escapeOrNotColor(x, y);
	
	// ADDED: Convert the iteration count into a value between 0 and 1.
	color = count / MAXITERATIONS;
	
	// ADDED: Create multiple colors using the iteration count.
	if(count >= MAXITERATIONS)
	{
		pixels[id] = 0.0;
		pixels[id+1] = 0.0;
		pixels[id+2] = 0.0;
	}
	else
	{
		pixels[id] = color;
		pixels[id+1] = color * color;
		pixels[id+2] = 1.0 - color;
	}
}

void display(void) 
{ 
	dim3 blockSize, gridSize;
	float *pixelsCPU, *pixelsGPU; 
	float stepSizeX, stepSizeY;
	
	//We need the 3 because each pixel has a red, green, and blue value.
	pixelsCPU = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	cudaMalloc(&pixelsGPU,WindowWidth*WindowHeight*3*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	stepSizeX = (XMax - XMin)/((float)WindowWidth);
	stepSizeY = (YMax - YMin)/((float)WindowHeight);
	
	//Threads in a block
	/*
	// CHANGED: The old code used 1024 threads in one block.
	// A CUDA block cannot always be used to represent an entire row,
	// and this limited the program to a maximum width of 1024.
	if(WindowWidth > 1024)
	{
	 	printf("The window width is too large to run with this program\n");
	 	printf("The window width must be less than 1024.\n");
	 	printf("Good Bye and have a nice day!\n");
	 	exit(0);
	}
	blockSize.x = 1024; //WindowWidth;
	blockSize.y = 1;
	blockSize.z = 1;
	*/
	
	// CHANGED: Use a smaller 2D block so the program can handle
	// window widths and heights of different sizes.
	blockSize.x = 16;
	blockSize.y = 16;
	blockSize.z = 1;
	
	//Blocks in a grid
	
	// CHANGED: Calculate the number of blocks needed for the
	// selected WindowWidth and WindowHeight.
	gridSize.x = (WindowWidth + blockSize.x - 1) / blockSize.x;
	gridSize.y = (WindowHeight + blockSize.y - 1) / blockSize.y;
	gridSize.z = 1;
	
	// CHANGED: Added WindowWidth and WindowHeight to the kernel call.
	colorPixels<<<gridSize, blockSize>>>(pixelsGPU, XMin, YMin, stepSizeX, stepSizeY, WindowWidth, WindowHeight);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Copying the pixels that we just colored back to the CPU.
	cudaMemcpyAsync(pixelsCPU, pixelsGPU, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixelsCPU); 
	glFlush(); 
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
