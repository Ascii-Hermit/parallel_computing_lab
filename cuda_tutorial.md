# Basic CUDA functions

1. Kernel:<br>
    A kernel is a function that you write in CUDA that will run on the GPU. You define a kernel using the `__global__` keyword
    – `func()` will execute on the device
    – `func()` will be called from the host

2. Launching Kernel:<br>
    `<<<...>>>` is known as the kernel launch configuration.
    Code: `kernelFunction<<<numBlocks, numThreadsPerBlock>>>(arguments);`

3. Thread Indexing:<br>
    Each thread in a block has a unique ID.<br>
    `threadIdx`: A built-in variable that contains the thread index within a block.<br>
    `blockIdx`: A built-in variable that contains the block index within the grid.<br>
    `blockDim`: A built-in variable that contains the dimensions (size) of a block.<br>
    `gridDim`: A built-in variable that contains the dimensions (size) of the grid.<br>

4. Memory Allocation:<br>
    We need to allocate memory on the GPU
    Simple CUDA API for handling device memory: <br>
        `cudaMalloc()` :Allocates memory on the GPU.<br>
        `cudaFree()` :Frees allocated memory on the device. <br>
        `cudaMemcpy()` :Copies data between host and device.<br>

    Note a better variation is using `cudaMallocManaged(void** cpu_pointer, size_t size)`
    This automatically allocates space and conpies content to GPU. 

5. Synchronization:<br>
        `__syncthreads()`: A barrier synchronization function. It ensures that all threads **in a block** reach the synchronization point.<br>
        `cudaDeviceSynchronize()`: Ensures that the host (CPU) waits for all threads on the device (GPU) to complete before continuing.


# Kernel Configurations

### Sample Grid and Block Configuration Breakdown

- **A kernel** is made up of a **grid**.
- That **grid** has **blocks** in it.
- Each **block** has its own set of **threads**.

#### Defining Kernel Configuration

1. **Threads per Block**  <br>
   This command defines the number of threads in each block:
   ```dim3 threadsPerBlock(16, 16);```

2. **Blocks per Grid**<br>
This command defines the number of blocks in a grid. <br>
Note that this will always be written after the threads are defined as we cant have different sized blocks (due to imperfect division of threads among blocks)<br>
`dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);`

3. **Kernel Instantiation**<br>
Finally pass this info to the kernel to be instantiated
`MatAdd<<<numBlocks, threadsPerBlock>>>(d_MatA, d_MatB, d_MatC);`

## 1. 1D Grid with 1D Blocks

### Configuration:
- **Grid**: 1D  
- **Blocks**: 1D (linear array of threads)  
- **Threads per Block**: 1D (linear arrangement of threads in a single row)  

### Thread Indexing:
- **Thread Index**: `idx = threadIdx.x + blockIdx.x * blockDim.x`  
  `idx` is a 1D index representing the position of the thread in the grid.

### Code:
- **Block Size**: `dim3 threadsPerBlock(x)` — all threads are arranged linearly.
- **Grid Size**: `dim3 numBlocks(x)` — the grid is also 1D, covering N elements.

```cpp
dim3 threadsPerBlock(256);
dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
MatAdd<<<numBlocks, threadsPerBlock>>>(MatA, MatB, MatC, N);
```

## 2. 1D Grid with 2D Blocks

### Configuration:
- **Grid**: 1D  
- **Blocks**: 2D (blocks arranged in rows and columns of threads)  
- **Threads per Block**: 2D (2D grid of threads in each block) 

### Thread Indexing:
- **Thread Index in X**: `idx = threadIdx.x + blockIdx.x * blockDim.x`  
- **Thread Index in Y**: `idy = threadIdx.y + blockIdx.y * blockDim.y` 

### Code:
- **Block Size**: `dim3 threadsPerBlock(x,y)` — each block is 2D, i.e., has multiple rows and columns of threads.
- **Grid Size**: `dim3 numBlocks(x)` — the grid is 1D, with a linear array of blocks.

```cpp
dim3 threadsPerBlock(16, 16); 
dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
MatAdd<<<numBlocks, threadsPerBlock>>>(MatA, MatB, MatC, N);
```
## 3. 2D Grid with 1D Blocks

### Configuration:
- **Grid**: 2D (arranged in rows and columns of blocks)
- **Blocks**: 1D (linear array of threads within each block) 
- **Threads per Block**: 1D (single row of threads per block)

### Thread Indexing:
- **Thread Index in X**: `idx = threadIdx.x + blockIdx.x * blockDim.x`  
- **Thread Index in Y**: `idy = threadIdx.y + blockIdx.y * blockDim.y` 

### Code:
- **Block Size**: `dim3 threadsPerBlock(x)` — the threads in each block are 1D.
- **Grid Size**: `dim3 numBlocks(x,y)`— the grid is 2D, with blocks arranged in rows and columns.

```cpp
dim3 threadsPerBlock(256);
dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
               (N + threadsPerBlock.x - 1) / threadsPerBlock.x);
MatAdd<<<numBlocks, threadsPerBlock>>>(MatA, MatB, MatC, N);
```

## 4. 2D Grid with 2D Blocks

### Configuration:
- **Grid**: 2D (arranged in rows and columns of blocks)
- **Blocks**: 2D (each block is a small 2D grid of threads)
- **Threads per Block**: 2D (block has multiple rows and columns of threads)

### Thread Indexing:
- **Thread Index in X**: `idx = threadIdx.x + blockIdx.x * blockDim.x`  
- **Thread Index in Y**: `idy = threadIdx.y + blockIdx.y * blockDim.y` 

### Code:
- **Block Size**: `dim3 threadsPerBlock(x,y)` — each block is 2D.
- **Grid Size**: `dim3 numBlocks(x,y)`— the grid is 2D, with blocks arranged in rows and columns.

```cpp
dim3 threadsPerBlock(16, 16); 
dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
               (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
MatAdd<<<numBlocks, threadsPerBlock>>>(MatA, MatB, MatC, N);

```

