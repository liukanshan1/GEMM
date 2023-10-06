#include<stdio.h>
#include<stdlib.h>
#include<mkl.h>
#include "../GEMM/kernel.h"
int main()
{
    /* 参数设置 */
    random_M_N_K();
    M = 950;
    N = 950;
    K = 9000;
    float* A, * B, *C;
    A = (float*)mkl_malloc(M * K * sizeof(float), 64);
    B = (float*)mkl_malloc(K * N * sizeof(float), 64);
    C = (float*)mkl_malloc(M * N * sizeof(float), 64);
    random_matrix(A, M, K);
    random_matrix(B, K, N);

    // 矩阵乘法
    clock_t start1, end1;
    start1 = clock();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K, B, N, 0.0, C, N);
    end1 = clock();
    printf("mkl time: %f ms\n", ((double)(end1 - start1) / CLOCKS_PER_SEC) * 1000);

    /* 结果验证 */
    float* C_cpu = new float[M * N];
    memset(C_cpu, 0, sizeof(float) * M * N);
    matMulCPU(C_cpu, A, B);
    double errorCountForMKL = 0;
    for (int i = 0; i < M * N; i++)
    {
        if (abs(C_cpu[i] - C[i]) > 1e-5)
        {
            errorCountForMKL+=1;
            //printf("mkl Wrong!\n");
            //std::cout << abs(C_cpu[i] - C[i]) << std::endl;
            //break;
        }
    }
    printf("Error times of MKL: %f\n", errorCountForMKL);
    printf("Error rate of MKL: %f\n", errorCountForMKL / (M * N));
    print_vector(C, 10);
    print_vector(C_cpu, 10);
    
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    return 0;
}