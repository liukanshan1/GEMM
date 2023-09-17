#include<stdio.h>
#include<stdlib.h>
#include<mkl.h>
#include "../GEMM/kernel.h"
int main()
{
    /* 参数设置 */
    random_M_N_K();
    float* A, * B, *C;
    A = (float*)mkl_malloc(M * K * sizeof(float), 64);
    B = (float*)mkl_malloc(K * N * sizeof(float), 64);
    C = (float*)mkl_malloc(M * N * sizeof(float), 64);
    random_matrix(A, M, K);
    random_matrix(B, K, N);

    // 矩阵乘法
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K, B, N, 0.0, C, N);

    /* 结果验证 */
    float* C_cpu = new float[M * N];
    memset(C_cpu, 0, sizeof(float) * M * N);
    matMulCPU(C_cpu, A, B);
    for (int i = 0; i < M * N; i++)
    {
        if (abs(C_cpu[i] - C[i]) > 1e-5)
        {
            printf("Wrong!");
            std::cout << abs(C_cpu[i] - C[i]) << std::endl;
            break;
        }
    }

    print_vector(C, 10);
    print_vector(C_cpu, 10);
    
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return 0;
}