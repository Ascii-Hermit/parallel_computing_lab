1. Kernel:
    A kernel is a function that you write in CUDA that will run on the GPU. You define a kernel using the `__global__` keyword
    – `func()` will execute on the device
    – `func()` will be called from the host

2. Launching Kernel:
    `<<<...>>>` is known as the kernel launch configuration.
    Code: `kernelFunction<<<numBlocks, numThreadsPerBlock>>>(arguments);`

3. Thread Indexing:
    Each thread in a block has a unique ID.
    `threadIdx`: A built-in variable that contains the thread index within a block.
    `blockIdx`: A built-in variable that contains the block index within the grid.
    `blockDim`: A built-in variable that contains the dimensions (size) of a block.
    `gridDim`: A built-in variable that contains the dimensions (size) of the grid.

4. Memory Allocation
    We need to allocate memory on the GPU
    Simple CUDA API for handling device memory: 
        `cudaMalloc()` :Allocates memory on the GPU.
        `cudaFree()` :Frees allocated memory on the device. 
        `cudaMemcpy()` :Copies data between host and device.

5. Synchronization
        `__syncthreads()`: A barrier synchronization function. It ensures that all threads **in a block** reach the synchronization point.
        `cudaDeviceSynchronize()`: Ensures that the host (CPU) waits for all threads on the device (GPU) to complete before continuing.