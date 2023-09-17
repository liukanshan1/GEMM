#pragma once
#include <random>
#include <iostream>
int M, N, K;

// 初始化参数
int m = 30;
int n = 30;
int k = 30;

enum class MatrixType
{
	RowMajor,
	ColMajor
};

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
void random_matrix(double* matrix, int row, int col)
{
	std::default_random_engine e;
	std::uniform_real_distribution<double> u(-100, 100);
	for (int i = 0; i < row * col; i++)
	{
		matrix[i] = u(e);
	}
}

//Row major 转 col major
void row_to_col(double* matrix, int row, int col)
{
	double* temp = new double[row * col];
	memccpy(temp, matrix, row * col, sizeof(double));
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; ++j)
		{
			matrix[j * row + i] = temp[i * col + j];
		}
	}
	delete[] temp;
}

// Col major 转 row major
void col_to_row(double* matrix, int row, int col)
{
	double* temp = new double[row * col];
	memccpy(temp, matrix, row * col, sizeof(double));
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; ++j)
		{
			matrix[i * col + j] = temp[j * row + i];
		}
	}
	delete[] temp;
}

void print_matrix(double* matrix, int row, int col, MatrixType type) {
if (type == MatrixType::RowMajor)
	{
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
	else
	{
		for (int i = 0; i < row; ++i)
		{
			std::cout << "[";
			for (int j = 0; j < col; ++j)
			{
				std::cout << matrix[j * row + i] << " ";
			}
			std::cout << "]" << std::endl;
		}
	}
}

void print_vector(double* vector, int row)
{
	for (int i = 0; i < row; ++i)
	{
		std::cout << vector[i] << " ";
	}
	std::cout << std::endl;
}
		