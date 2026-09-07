// Name: Marshall Galbreath
// Device query
// nvcc E_DeviceQuery.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

/*
 Purpose:
 To learn how to find out what is on the GPU(s) in your machine and if you even have a GPU.
*/

/*
 Explain what you did to fix the code:
 
 I extended the program so that it displays additional information stored in
 the cudaDeviceProp structure. The program now displays information about
 the GPU's memory, multiprocessors, threads, texture limits, surface limits,
 clock speeds, cache, PCI information, and other CUDA capabilities.

 I also added the CUDA runtime header so that CUDA types and functions are
 properly defined. I changed the memory output to use %zu because the
 cudaDeviceProp memory fields use the size_t data type.

 Error checking was also kept after the CUDA function calls so that the
 program reports an error if CUDA cannot find a device or if another CUDA
 runtime error occurs.
*/

// Include files

// stdio.h provides printf(), which is used to display information.
#include <stdio.h>

// stdlib.h provides exit() and EXIT_FAILURE for ending the program
// if a CUDA error occurs.
#include <stdlib.h>

// cuda_runtime.h provides the CUDA Runtime API, including
// cudaDeviceProp, cudaGetDeviceCount(), and cudaGetDeviceProperties().
#include <cuda_runtime.h>

// Defines

// This macro makes it easier to call the CUDA error checking function.
// __FILE__ contains the current source filename.
// __LINE__ contains the current line number.
#define CUDA_CHECK() cudaErrorCheck(__FILE__, __LINE__)

// Global variables

// Function prototypes

// This function checks whether the most recent CUDA operation
// generated an error.
void cudaErrorCheck(const char*, int);


// This function checks for CUDA errors.
// file = the name of the source file where the error occurred.
// line = the line number where the error occurred.
// cudaGetLastError() returns the most recent CUDA error.
// If an error occurred, the program prints the error message
// and terminates.
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t error;

	// Get the most recent CUDA error.
	error = cudaGetLastError();

	// Compare the error against cudaSuccess.
	// If it is not successful, display information about the error.
	if(error != cudaSuccess)
	{
		printf("\nCUDA ERROR: message = %s, File = %s, Line = %d\n",
		       cudaGetErrorString(error), file, line);

		// Stop the program because a CUDA error occurred.
		exit(EXIT_FAILURE);
	}
}


int main()
{
	// cudaDeviceProp is a CUDA structure containing information
	// about a CUDA-capable GPU.
	cudaDeviceProp prop;

	// count will store the number of CUDA-capable GPUs
	// found in the computer.
	int count;


	// FIND THE NUMBER OF CUDA-CAPABLE GPUs

	// cudaGetDeviceCount() asks CUDA how many CUDA-capable
	// devices are available in the computer.
	cudaGetDeviceCount(&count);

	// Check whether the CUDA function generated an error.
	CUDA_CHECK();

	// Display the number of GPUs found.
	printf("You have %d GPUs in this machine\n\n", count);


	// HANDLE THE CASE WHERE NO GPU WAS FOUND

	// If count is zero, there are no CUDA-capable GPUs available.
	if (count == 0)
	{
		printf("No CUDA-capable GPU was found.\n");
		return 0;
	}


	// LOOP THROUGH EVERY GPU

	// Computers can have more than one GPU.
	// This loop runs once for every CUDA-capable GPU.
	for (int i = 0; i < count; i++)
	{
		// cudaGetDeviceProperties() fills the cudaDeviceProp
		// structure with information about GPU number i.
		cudaGetDeviceProperties(&prop, i);

		// Check whether getting the device properties succeeded.
		CUDA_CHECK();


		// GENERAL INFORMATION

		printf(" ---General Information for device %d ---\n", i);

		// prop.name contains the name/model of the GPU.
		printf("Name: %s\n", prop.name);

		// major and minor describe the GPU's CUDA compute capability.
		// Compute capability identifies the hardware feature level
		// supported by the GPU.
		printf("Compute capability: %d.%d\n",
		       prop.major, prop.minor);

		// clockRate is the GPU core clock speed in kHz.
		printf("Clock rate: %d kHz\n", prop.clockRate);

		// memoryClockRate is the memory clock speed in kHz.
		printf("Memory clock rate: %d kHz\n",
		       prop.memoryClockRate);

		// memoryBusWidth is the width of the GPU's memory interface
		// in bits.
		printf("Memory bus width: %d bits\n",
		       prop.memoryBusWidth);

		// l2CacheSize is the amount of L2 cache available on the GPU.
		printf("L2 cache size: %d bytes\n",
		       prop.l2CacheSize);

		// deviceOverlap indicates whether the device can overlap
		// memory transfers with kernel execution.
		printf("Device copy overlap: ");
		if (prop.deviceOverlap)
			printf("Enabled\n");
		else
			printf("Disabled\n");

		// kernelExecTimeoutEnabled indicates whether a kernel
		// can be terminated if it takes too long to execute.
		printf("Kernel execution timeout: ");
		if (prop.kernelExecTimeoutEnabled)
			printf("Enabled\n");
		else
			printf("Disabled\n");

		// integrated indicates whether the GPU is integrated
		// rather than a separate/discrete GPU.
		printf("Integrated GPU: ");
		if (prop.integrated)
			printf("Yes\n");
		else
			printf("No\n");

		// canMapHostMemory indicates whether the GPU can map
		// pinned host memory into the GPU address space.
		printf("Can map host memory: ");
		if (prop.canMapHostMemory)
			printf("Yes\n");
		else
			printf("No\n");

		// concurrentKernels indicates whether multiple kernels
		// can execute at the same time when resources permit.
		printf("Concurrent kernels: ");
		if (prop.concurrentKernels)
			printf("Yes\n");
		else
			printf("No\n");

		// ECCEnabled indicates whether error-correcting code
		// memory support is enabled.
		printf("ECC support: ");
		if (prop.ECCEnabled)
			printf("Enabled\n");
		else
			printf("Disabled\n");

		// unifiedAddressing indicates whether the CPU and GPU
		// share a unified virtual address space.
		printf("Unified addressing: ");
		if (prop.unifiedAddressing)
			printf("Supported\n");
		else
			printf("Not supported\n");

		// managedMemory indicates whether the GPU supports
		// CUDA Unified Memory.
		printf("Managed memory: ");
		if (prop.managedMemory)
			printf("Supported\n");
		else
			printf("Not supported\n");

		// pageableMemoryAccess indicates support for accessing
		// pageable host memory from the GPU under supported conditions.
		printf("Pageable memory access: ");
		if (prop.pageableMemoryAccess)
			printf("Supported\n");
		else
			printf("Not supported\n");

		// concurrentManagedAccess indicates whether the device
		// supports concurrent access to managed memory.
		printf("Concurrent managed access: ");
		if (prop.concurrentManagedAccess)
			printf("Supported\n");
		else
			printf("Not supported\n");


		// MEMORY INFORMATION

		printf("\n");
		printf(" ---Memory Information for device %d ---\n", i);

		// totalGlobalMem is the total amount of global GPU memory.
		// It is stored as size_t, so %zu is used for printing it.
		// The second value converts the number of bytes into GB.
		printf("Total global mem: %zu bytes (%.2f GB)\n",
		       prop.totalGlobalMem,
		       (double)prop.totalGlobalMem /
		       (1024.0 * 1024.0 * 1024.0));

		// totalConstMem is the total amount of constant memory
		// available on the device.
		printf("Total constant Mem: %zu bytes\n",
		       prop.totalConstMem);

		// memPitch is the maximum allowed pitch for memory allocations.
		printf("Max mem pitch: %zu bytes\n",
		       prop.memPitch);

		// textureAlignment specifies the required alignment
		// for certain texture memory operations.
		printf("Texture Alignment: %zu bytes\n",
		       prop.textureAlignment);

		// texturePitchAlignment specifies the alignment requirement
		// for texture pitch memory.
		printf("Texture Pitch Alignment: %zu bytes\n",
		       prop.texturePitchAlignment);

		// sharedMemPerBlock is the maximum amount of shared memory
		// available to one thread block.
		printf("Shared memory per block: %zu bytes\n",
		       prop.sharedMemPerBlock);

		// sharedMemPerMultiprocessor is the total shared memory
		// available to a streaming multiprocessor (SM).
		printf("Shared memory per MP: %zu bytes\n",
		       prop.sharedMemPerMultiprocessor);

		// regsPerBlock is the maximum number of registers
		// that can be allocated to a thread block.
		printf("Registers per block: %d\n",
		       prop.regsPerBlock);

		// regsPerMultiprocessor is the number of registers
		// available on an SM.
		printf("Registers per MP: %d\n",
		       prop.regsPerMultiprocessor);


		// MULTIPROCESSOR INFORMATION

		printf("\n");
		printf(" ---MP Information for device %d ---\n", i);

		// multiProcessorCount tells us how many Streaming
		// Multiprocessors (SMs) are on the GPU.
		printf("Multiprocessor count: %d\n",
		       prop.multiProcessorCount);

		// warpSize is the number of threads in one CUDA warp.
		// NVIDIA GPUs normally use a warp size of 32.
		printf("Threads in warp: %d\n",
		       prop.warpSize);

		// maxThreadsPerMultiProcessor is the maximum number
		// of resident threads that can be active on one SM.
		printf("Max threads per multiprocessor: %d\n",
		       prop.maxThreadsPerMultiProcessor);

		// maxThreadsPerBlock is the maximum number of threads
		// allowed in a single CUDA thread block.
		printf("Max threads per block: %d\n",
		       prop.maxThreadsPerBlock);

		// maxThreadsDim contains the maximum x, y, and z dimensions
		// of a thread block.
		printf("Max thread dimensions: (%d, %d, %d)\n",
		       prop.maxThreadsDim[0],
		       prop.maxThreadsDim[1],
		       prop.maxThreadsDim[2]);

		// maxGridSize contains the maximum x, y, and z dimensions
		// of a CUDA grid.
		printf("Max grid dimensions: (%d, %d, %d)\n",
		       prop.maxGridSize[0],
		       prop.maxGridSize[1],
		       prop.maxGridSize[2]);


		// TEXTURE INFORMATION

		printf("\n");
		printf(" ---Texture Information for device %d ---\n", i);

		// Maximum size of a 1D texture.
		printf("Max texture 1D: %d\n",
		       prop.maxTexture1D);

		// Maximum size of a 1D mipmapped texture.
		printf("Max texture 1D mipmapped: %d\n",
		       prop.maxTexture1DMipmap);

		// Maximum size of a 1D linear texture.
		printf("Max texture 1D linear: %d\n",
		       prop.maxTexture1DLinear);

		// Maximum width and height of a 2D texture.
		printf("Max texture 2D: (%d, %d)\n",
		       prop.maxTexture2D[0],
		       prop.maxTexture2D[1]);

		// Maximum width and height of a 2D mipmapped texture.
		printf("Max texture 2D mipmapped: (%d, %d)\n",
		       prop.maxTexture2DMipmap[0],
		       prop.maxTexture2DMipmap[1]);

		// Maximum width and height of a 2D linear texture.
		printf("Max texture 2D linear: (%d, %d)\n",
		       prop.maxTexture2DLinear[0],
		       prop.maxTexture2DLinear[1]);

		// Maximum dimensions of a 2D texture used with gather operations.
		printf("Max texture 2D gather: (%d, %d)\n",
		       prop.maxTexture2DGather[0],
		       prop.maxTexture2DGather[1]);

		// Maximum dimensions of a 3D texture.
		printf("Max texture 3D: (%d, %d, %d)\n",
		       prop.maxTexture3D[0],
		       prop.maxTexture3D[1],
		       prop.maxTexture3D[2]);

		// Maximum size of a cubemap texture.
		printf("Max texture cubemap: %d\n",
		       prop.maxTextureCubemap);


		// SURFACE INFORMATION

		printf("\n");
		printf(" ---Surface Information for device %d ---\n", i);

		// Maximum dimensions for 1D surface memory.
		printf("Max surface 1D: %d\n",
		       prop.maxSurface1D);

		// Maximum width and height for 2D surface memory.
		printf("Max surface 2D: (%d, %d)\n",
		       prop.maxSurface2D[0],
		       prop.maxSurface2D[1]);

		// Maximum width, height, and depth for 3D surface memory.
		printf("Max surface 3D: (%d, %d, %d)\n",
		       prop.maxSurface3D[0],
		       prop.maxSurface3D[1],
		       prop.maxSurface3D[2]);

		// Maximum size of a cubemap surface.
		printf("Max surface cubemap: %d\n",
		       prop.maxSurfaceCubemap);


		// HARDWARE / CUDA INFORMATION

		printf("\n");
		printf(" ---Hardware/CUDA Information for device %d ---\n", i);

		// pciDomainID identifies the PCI domain containing the GPU.
		printf("PCI Domain ID: %d\n",
		       prop.pciDomainID);

		// pciBusID identifies the PCI bus containing the GPU.
		printf("PCI Bus ID: %d\n",
		       prop.pciBusID);

		// pciDeviceID identifies the GPU on the PCI bus.
		printf("PCI Device ID: %d\n",
		       prop.pciDeviceID);

		// tccDriver indicates whether the GPU is using
		// the NVIDIA Tesla Compute Cluster driver.
		printf("TCC Driver: ");
		if (prop.tccDriver)
			printf("Yes\n");
		else
			printf("No\n");

		// asyncEngineCount tells how many asynchronous copy engines
		// the GPU has for transferring data.
		printf("Async engine count: %d\n",
		       prop.asyncEngineCount);

		// streamPrioritiesSupported indicates whether CUDA streams
		// can be assigned different priorities.
		printf("Stream priorities supported: ");
		if (prop.streamPrioritiesSupported)
			printf("Yes\n");
		else
			printf("No\n");

		// globalL1CacheSupported indicates whether global memory
		// can use the L1 cache.
		printf("Global L1 cache supported: ");
		if (prop.globalL1CacheSupported)
			printf("Yes\n");
		else
			printf("No\n");

		// localL1CacheSupported indicates whether local memory
		// can use the L1 cache.
		printf("Local L1 cache supported: ");
		if (prop.localL1CacheSupported)
			printf("Yes\n");
		else
			printf("No\n");

		// hostNativeAtomicSupported indicates whether the device
		// supports native host atomic operations.
		printf("Host native atomic support: ");
		if (prop.hostNativeAtomicSupported)
			printf("Yes\n");
		else
			printf("No\n");

		// cooperativeLaunch indicates whether cooperative kernel
		// launches are supported.
		printf("Cooperative launch: ");
		if (prop.cooperativeLaunch)
			printf("Supported\n");
		else
			printf("Not supported\n");

		// cooperativeMultiDeviceLaunch indicates whether cooperative
		// launches across multiple CUDA devices are supported.
		printf("Cooperative multi-device launch: ");
		if (prop.cooperativeMultiDeviceLaunch)
			printf("Supported\n");
		else
			printf("Not supported\n");

		// memoryPoolsSupported indicates whether the device supports
		// CUDA memory pools.
		printf("Memory pools supported: ");
		if (prop.memoryPoolsSupported)
			printf("Yes\n");
		else
			printf("No\n");

		// gpuDirectRDMASupported indicates whether the GPU supports
		// GPU Direct RDMA.
		printf("GPU Direct RDMA supported: ");
		if (prop.gpuDirectRDMASupported)
			printf("Yes\n");
		else
			printf("No\n");


		// ADDITIONAL INFORMATION

		printf("\n");
		printf(" ---Additional Information for device %d ---\n", i);

		// Compute mode specifies how the CUDA device can be used
		// by CUDA processes.
		printf("Compute mode: %d\n",
		       prop.computeMode);

		// addressableMemory represents the amount of addressable
		// memory supported by the device.
		printf("Addressable memory: %zu bytes\n",
		       prop.addressableMemory);

		// reservedSharedMemPerBlock is the amount of shared memory
		// reserved by the hardware/runtime for each block.
		printf("Reserved shared memory per block: %zu bytes\n",
		       prop.reservedSharedMemPerBlock);


		// Print a blank line before moving to the next GPU.
		printf("\n");
	}


	// The program successfully finished displaying information
	// about all CUDA-capable GPUs.
	return(0);
}
