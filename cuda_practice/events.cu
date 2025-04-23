cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start); // Record start time

dim3 gridDim(ROWS);
dim3 blockDim(COLS);
matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C);

cudaEventRecord(stop);  // Record stop time
cudaDeviceSynchronize(); // Wait for the kernel to finish

// Calculate elapsed time
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Time taken for matrix multiplication: %f ms\n", milliseconds);