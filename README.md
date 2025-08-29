# ConvTest: CUDA Convolution Optimization Project

This project demonstrates a progressive optimization of 2D convolution using CUDA, targeting a Tesla T4 GPU in a Google Colab environment.

## Project Structure

The project is divided into five phases, each introducing new optimization techniques.

*   **Phase 1: Environment Setup and Baseline**
    *   `1.1_check_gpu_environment.cu`: Verifies the CUDA environment and displays GPU device capabilities.
    *   `1.2_cpu_baseline_convolution.cu`: Implements a high-performance CPU convolution as a baseline for comparison.
    *   `1.3_cuda_optimization_setup.cu`: Optimizes the CUDA environment and analyzes memory hierarchy.
*   **Phase 2: Basic CUDA Implementation**
    *   `2.0_naive_convolution_complete.cu`: A basic CUDA convolution implementation used to establish a baseline GPU performance.
*   **Phase 3: Shared Memory Optimization**
    *   `3.0_phase3_final_corrected.cu`: Explores different shared memory optimization techniques to improve performance.
*   **Phase 4: Kernel Fusion**
    *   `4.0_phase4_kernel_fusion.cu`: Implements kernel fusion techniques, such as fusing convolution with ReLU and max pooling, to reduce memory traffic and kernel launch overhead.
*   **Phase 5: Final Optimization**
    *   `5.0_phase5_final_optimization.cu`: Includes multi-stream pipelines and other advanced optimization techniques. Focuses on achieving perfect warp efficiency by using a 16x16 block size.
    *   `5.1_phase5_tuned_optimization.cu`: A tuned version of Phase 5, focusing on coalesced loads, shared memory padding, and read-only cache hints.
*   **Utility Files**
    *   `edge_detect_runner.cu`: An edge detection application that uses different convolution implementations from the project.
    *   `cat.jpg`: An image file used as input for the edge detection application.
    *   `QUICKSTART.md`: A quick start guide for the project.
    *   `README.md`: This file.

## Setup and Compilation (Google Colab)

1.  **Mount Google Drive**: In Google Colab, mount your Google Drive:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2.  **Set Working Directory**: Set the working directory to the project folder:

    ```python
    import os
    project_dir = "/content/drive/MyDrive/ConvTest"
    os.chdir(project_dir)
    print("Current directory:", os.getcwd())
    ```

    Ensure that all CUDA source files and `cat.jpg` are located in this directory.

3.  **Compilation**: Use `nvcc` to compile the CUDA code.  The `-arch=sm_75` flag is crucial for targeting Tesla T4 GPUs. Use the following flags for optimization: `-O3 --use_fast_math -maxrregcount=32` (or `-maxrregcount=64` for `edge_detect_runner.cu`).

    Example:

    ```bash
    nvcc -arch=sm_75 -O3 --use_fast_math 2.0_naive_convolution_complete.cu -o phase2_baseline
    ```

4.  **Execution**: Run the compiled executables using shell commands:

    ```bash
    ./phase2_baseline
    ```

## Performance Expectations

*   Phase 2: 2-5x speedup over CPU.
*   Phase 3: 5-10x speedup with shared memory.
*   Phase 4: 10-20x speedup with kernel fusion.
*   Phase 5: 15-25x speedup with final optimizations.

## Edge Detection

The `edge_detect_runner.cu` application performs edge detection using different convolution implementations.

1.  **Prepare Input**: The notebook loads `cat.jpg`, converts it to grayscale, and saves it as a float32 binary file (`input.f32`).
2.  **Compile**: Compile `edge_detect_runner.cu` with the following command:

    ```bash
    nvcc -arch=sm_75 -O3 -Xptxas -v -maxrregcount=64 edge_detect_runner.cu -o edge_runner
    ```

3.  **Run**: Execute the edge detection with different modes and kernels:

    ```bash
    ./edge_runner --mode=phase2 --kernel=edge --in=input.f32 --out=output.f32 --w={width} --h={height}
    ```

    Replace `{width}` and `{height}` with the actual dimensions of the image.
    Supported modes: `phase2`, `phase4`, `phase5`.
    Supported kernels: `edge`, `sobelx`, `sobely`.

4.  **Visualize**: The notebook visualizes the original and edge-detected images using `matplotlib`.

## Notes

*   This project is designed to run in a Google Colab environment with a Tesla T4 GPU.
*   Ensure that the correct CUDA architecture flag (`-arch=sm_75`) is used during compilation.
*   Experiment with different optimization techniques and analyze their impact on performance.