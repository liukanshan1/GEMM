
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GEMM.h"
#include <stdio.h>

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;	//结果矩阵C的行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x;	//结果矩阵C的列索引
    for (int e = 0; e < A.width; ++e)
    {
        Cvalue += A.elements[row * A.width + e]			//所有点到点的元素乘积求和
            * B.elements[e * B.width + col];
        C.elements[row * C.width + col] = Cvalue;
    }
}

template<int BLOCK_SIZE> __global__ void MatrixMulCUDA(float* C, float* A, float* B,
    int wA, int wB, int hA, int hB)
{
    //Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    //Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //将矩阵划分为子矩阵，对子矩阵的乘法应用块内线程并行计算，最后将它们的值相加得到C的一个元素值
    int aBegin = by * BLOCK_SIZE * wA;	//A的子矩阵的行坐标
    int aStep = BLOCK_SIZE;				//A的子矩阵列坐标的移动步长
    int aEnd = aBegin + wA - 1;			//限定一个终点

    int bBegin = bx * BLOCK_SIZE;
    int bStep = BLOCK_SIZE * wB;

    float Csub = 0;		//定义在block(x,. y)块中（ty, tx）对应位置的C的元素值

    int subAw = BLOCK_SIZE;
    int subAh = BLOCK_SIZE;
    int subBh = BLOCK_SIZE;
    int subBw = BLOCK_SIZE;
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {

        if (a + aStep - 1 > aEnd)		//A矩阵最后一列的块的列数少于BLOCK_SIZE
        {
            subAw = aEnd - a + 1;
        }
        else
        {
            subAw = BLOCK_SIZE;
        }
        subBh = subAw;

        if ((by + 1) * BLOCK_SIZE > hA)		//A矩阵最后一行的块的行数少于BLOCK_SIZE
        {
            subAh = hA - by * BLOCK_SIZE;
        }
        else
        {
            subAh = BLOCK_SIZE;
        }

        if ((bx + 1) * BLOCK_SIZE > wB)		//B矩阵最后一列的块的列数少于BLOCK_SIZE
        {
            subBw = wB - bx * BLOCK_SIZE;
        }
        else
        {
            subBw = BLOCK_SIZE;
        }

        /* 开辟块内共享内存 */
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

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
        C[by * BLOCK_SIZE * wB + bx * BLOCK_SIZE + ty * wB + tx] = Csub;
    }
}



int main()
{
    /* 参数设置 */
    dim3 dimsA(1055, 2137, 1);		//矩阵的宽、高和未使用参数1
    dim3 dimsB(108, 1055, 1);		//矩阵的宽、高和未使用参数1

    /* 矩阵初始化、内存传递等常规步骤
    ....
    */
    float* A, * B, * C;

    float* d_A, * d_B, * d_C;


    /* 调用核函数计算 */
    dim3 threads(block_size, block_size);
    dim3 grid((dimsB.x - 1) / threads.x + 1, (dimsA.y - 1) / threads.y + 1);

    
   
    MatrixMulCUDA<block_size> <<<grid, threads >>> (d_C, d_A, d_B, dimsA.x, dimsB.x, dimsA.y, dimsB.y);

    return 0;
}
