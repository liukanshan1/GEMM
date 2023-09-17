#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "kernel.h"
#include "cublas_v2.h"

__global__ void MatrixMulCUDA(float* C, float* A, float* B,
    int wA, int wB, int hA, int hB)
{
    //Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    //Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //将矩阵划分为子矩阵，对子矩阵的乘法应用块内线程并行计算，最后将它们的值相加得到C的一个元素值
    int aBegin = by * block_size * wA;	//A的子矩阵的行坐标
    int aStep = block_size;				//A的子矩阵列坐标的移动步长
    int aEnd = aBegin + wA - 1;			//限定一个终点

    int bBegin = bx * block_size;
    int bStep = block_size * wB;

    float Csub = 0;		//定义在block(x,. y)块中（ty, tx）对应位置的C的元素值

    int subAw = block_size;
    int subAh = block_size;
    int subBh = block_size;
    int subBw = block_size;
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {

        if (a + aStep - 1 > aEnd)		//A矩阵最后一列的块的列数少于BLOCK_SIZE
        {
            subAw = aEnd - a + 1;
        }
        else
        {
            subAw = block_size;
        }
        subBh = subAw;

        if ((by + 1) * block_size > hA)		//A矩阵最后一行的块的行数少于BLOCK_SIZE
        {
            subAh = hA - by * block_size;
        }
        else
        {
            subAh = block_size;
        }

        if ((bx + 1) * block_size > wB)		//B矩阵最后一列的块的列数少于BLOCK_SIZE
        {
            subBw = wB - bx * block_size;
        }
        else
        {
            subBw = block_size;
        }

        /* 开辟块内共享内存 */
        __shared__ float As[block_size][block_size];
        __shared__ float Bs[block_size][block_size];

        /* 为行和列范围内的子矩阵对应元素赋值 */
        if (ty < subAh && tx < subAw)
        {
            As[ty][tx] = A[a + ty * wA + tx];
        }
        if (ty < subBh && tx < subBw)
        {
            Bs[ty][tx] = B[b + ty * wB + tx];
        }
        __syncthreads();

        //展开循环来 编译以加速		
#pragma unroll
        //内循环计算每个子矩阵内对应行和列的向量乘积，累加到之前得到的值上
        for (int k = 0; k < subAw; k++)
        {
            if (ty < subAh && tx < subBw)	//满足行和列约束内的元素计算乘积并求和
            {
                Csub += As[ty][k] * Bs[k][tx];
            }
        }
        __syncthreads();
    }

    //满足行和列约束内的元素计算乘积并求和
    if (ty < subAh && tx < subBw)
    {
        C[by * block_size * wB + bx * block_size + ty * wB + tx] = Csub;
    }
}

int main()
{
    /* 参数设置 */
    random_M_N_K();
    dim3 dimsA(K, M, 1);		//矩阵的宽、高和未使用参数1
    dim3 dimsB(N, K, 1);		//矩阵的宽、高和未使用参数1

    /* 矩阵初始化、内存传递等常规步骤 */
    float* A, * B, * C;
    A = new float[M * K];
    B = new float[K * N];
    C = new float[M * N];
    random_matrix(A, M, K);
    random_matrix(B, K, N);
    memset(C, 0, sizeof(float) * M * N);
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, sizeof(float) * M * K);
    cudaMalloc((void**)&d_B, sizeof(float) * K * N);
    cudaMalloc((void**)&d_C, sizeof(float) * M * N);
    cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    /* 调用核函数计算 */
    dim3 threads(block_size, block_size);
    dim3 grid((dimsB.x - 1) / threads.x + 1, (dimsA.y - 1) / threads.y + 1);

    MatrixMulCUDA <<< grid, threads >>> (d_C, d_A, d_B, dimsA.x, dimsB.x, dimsA.y, dimsB.y);

    /* 结果传回主机端 */
    cudaMemcpy(C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    /* 结果验证 */
    float* C_cpu = new float[M * N];
    memset(C_cpu, 0, sizeof(float) * M * N);
    matMulCPU(C_cpu, A, B);
    for (int i = 0; i < M * N; i++)
    {
        if (abs(C_cpu[i] - C[i]) > 1e-5)
        {
			printf("Wrong!\n");
			break;
		}
	}

    // cublas实现
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    float* d_C_cublas;
    cudaMalloc((void**)&d_C_cublas, sizeof(float) * M * N);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N);
    cudaMemcpy(C, d_C_cublas, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cublasDestroy(handle);
    for (int i = 0; i < M * N; i++)
    {
        if (abs(C_cpu[i] - C[i]) > 1e-5)
        {
            printf("GPUWrong!\n");
            std::cout << abs(C_cpu[i] - C[i]) << std::endl;
            break;
        }
    }

    //print_matrix(C, M, N);
    //print_matrix(C_cpu, M, N);
    print_vector(C, 10);
    print_vector(C_cpu, 10);

    return 0;
}


