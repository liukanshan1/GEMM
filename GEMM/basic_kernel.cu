#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include"kernel.h"
#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa5(i,j) sa5[((j)<<5) + (i)]
#define sb5(i,j) sb5[((j)<<5) + (i)]
#define MS 32
#define NS 32
#define KS 32
#define CEIL_DIV(m,n) ( (m) + (n) - 1 ) / (n)
// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 4x1 micro kernel.
__global__  /*__launch_bounds__(256)*/
void mysgemm_v5(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int row1 = (tx & 7) << 2, row2 = row1 + 1, row3 = row1 + 2, row4 = row1 + 3, col = tx >> 3;
    A = &A((bx << 5), 0);
    B = &B(0, (by << 5));
    C = &C((bx << 5), (by << 5));
    __shared__ float sa5[MS * KS];
    __shared__ float sb5[KS * NS];
    float Cres[4] = { 0., 0., 0., 0. };
    float b00;
    for (int k_count = 0; k_count < K; k_count += KS) {
        sa5(row1, col) = A(row1, col);
        sa5(row2, col) = A(row2, col);
        sa5(row3, col) = A(row3, col);
        sa5(row4, col) = A(row4, col);
        sb5(col, row1) = B(row1, col);
        sb5(col, row2) = B(row2, col);
        sb5(col, row3) = B(row3, col);
        sb5(col, row4) = B(row4, col);
        A += (lda << 5); B += 32;
        __syncthreads();
#pragma unroll
        for (int inner_k_count = 0; inner_k_count < KS; inner_k_count++) {
            b00 = sb5(col, inner_k_count);
            Cres[0] += sa5(row1, inner_k_count) * b00;
            Cres[1] += sa5(row2, inner_k_count) * b00;
            Cres[2] += sa5(row3, inner_k_count) * b00;
            Cres[3] += sa5(row4, inner_k_count) * b00;
        }
        __syncthreads();
    }
    C(row1, col) = alpha * Cres[0] + beta * C(row1, col);
    C(row2, col) = alpha * Cres[1] + beta * C(row2, col);
    C(row3, col) = alpha * Cres[2] + beta * C(row3, col);
    C(row4, col) = alpha * Cres[3] + beta * C(row4, col);
}


 // 基础GPU 核函数，用于计算矩阵乘法
__global__ void MatrixMulCUDA(float* C, float* A, float* B, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}


int main() {
    //for (int i = 1; i <= 9; i++) {
        //M = N = 100 * i; // 矩阵 A 的行数
        K = 5000; // 矩阵 A 的列数，同时也是矩阵 B 的行数
        M = 100; // 矩阵 A 的行数
        N = 100; // 矩阵 B 的列数
        clock_t start1, end1;
        clock_t start2, end2;
        // 分配主机内存并初始化矩阵 A 和 B
        float* A = new float[M * K];
        float* B = new float[K * N];
        float* C = new float[M * N];

        random_matrix(A, M, K);
        random_matrix(B, K, N);

        // 分配 GPU 内存
        float* d_A, * d_B, * d_C;
        cudaMalloc((void**)&d_A, sizeof(float) * M * K);
        cudaMalloc((void**)&d_B, sizeof(float) * K * N);
        cudaMalloc((void**)&d_C, sizeof(float) * M * N);

        start1 = clock();
        // 将数据从主机内存复制到 GPU 内存
        cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);

        // 定义线程块和网格大小
        //dim3 threadsPerBlock(32, 32); // 16x16 线程块
        //dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
        //
        dim3 blockDim(32);
        dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
        int alpha = 1, beta = 0;
        mysgemm_v5 << <gridDim, blockDim >> > (M, N, K, alpha, d_A, d_B, beta, d_C);

        // 调用 GPU 核函数计算矩阵乘法
        //MatrixMulCUDA << < numBlocks, threadsPerBlock >> > (d_C, d_A, d_B, M, K, N);
        //MatrixMulCUDA <<< numBlocks, threadsPerBlock >>> (M, N, K, 1, d_A, d_B, 0, d_C);

        // 将结果从 GPU 复制回主机内存
        cudaMemcpy(C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

        end1 = clock();
        double endtime = (double)(end1 - start1) / CLOCKS_PER_SEC;
        endtime *= 1000;
        printf("GPU time: %f ms\n", endtime);
   /* }*/

     //打印结果矩阵 C
    //std::cout << "Result Matrix C:" << std::endl;
    //for (int i = 0; i < M; ++i) {
    //    for (int j = 0; j < N; ++j) {
    //        std::cout << C[i * N + j] << " ";
    //    }
    //    std::cout << std::endl;
    //}

    // 释放 GPU 内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 释放主机内存
    delete[] A;
    delete[] B;
    delete[] C;

    // GEMM on CPU
    //float* C_cpu = new float[M * N];
    //memset(C_cpu, 0, sizeof(float) * M * N);
    //start2 = clock();
    //matMulCPU(C_cpu, A, B);
    //end2 = clock();
    //double endtime2 = (double)(end2 - start2) / CLOCKS_PER_SEC;
    //endtime2 *= 1000;
    //printf("CPU time: %f ms\n", endtime2);

    //double errorCountForGEMM = 0;
    ///* GEMM 结果验证 */
    //for (int i = 0; i < M * N; i++)
    //{
    //    if (abs(C_cpu[i] - C[i]) > 1e-5)
    //    {
    //        errorCountForGEMM++;
    //        //printf("GEMM Wrong!\n");
    //        //std::cout << abs(C_cpu[i] - C[i]) << std::endl;
    //        //break;
    //    }
    //}
    //printf("Error times for GEMM: %f\n", errorCountForGEMM);
    //printf("Error rate of GEMM: %f\n", errorCountForGEMM / (M * N));
    //print_vector(C, 10);
    //print_vector(C_cpu, 10);

    return 0;
}
