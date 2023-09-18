#pragma once

#include <random>
#include <iostream>

int M, N, K;

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	int width;
	int height;
	float* elements;
} Matrix;

//随机生成M，N，K
void random_M_N_K()
{
	std::default_random_engine e;
	std::uniform_int_distribution<unsigned> u1(1, 1000);
	std::uniform_int_distribution<unsigned> u2(5000, 10000);
	M = u1(e);
	N = u1(e);
	K = u2(e);
}

//随机生成矩阵
void random_matrix(float* matrix, int row, int col)
{
	std::default_random_engine e;
	std::uniform_real_distribution<float> u(-100, 100);
	for (int i = 0; i < row * col; i++)
	{
		matrix[i] = u(e);
	}
}

void matMulCPU(float* C_cpu, float* A, float* B)
{
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; ++j)
		{
			for (int k = 0; k < K; ++k)
			{
				C_cpu[i * N + j] += A[i * K + k] * B[k * N + j];
			}
		}
	}
}

void print_matrix(float* matrix, int row, int col) {
	for (int i = 0; i < row; ++i)
	{
		std::cout << "[";
		for (int j = 0; j < col; ++j)
		{
			std::cout << matrix[i * col + j] << " ";
		}
		std::cout << "]" << std::endl;
	}
}

void print_vector(float* vector, int len)
{
	for (int i = 0; i < len; ++i)
	{
		std::cout << vector[i] << " ";
	}
	std::cout << std::endl;
}
