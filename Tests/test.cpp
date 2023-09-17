#include "pch.h"
#include "../GEMM/GEMM.h"

TEST(Header, GenMNK) {
	random_M_N_K();
	EXPECT_LE(M, 1000);
	EXPECT_LE(N, 1000);
	EXPECT_LE(K, 10000);
	EXPECT_GT(M, 0);
	EXPECT_GT(N, 0);
	EXPECT_GT(K, 0);
	EXPECT_TRUE(true)<< M<<" " << N <<" " << K << " ";
}

TEST(Header, MatrixGen) {
	double* matrix = new double[100];
	random_matrix(matrix, 10, 10);
	for (int i = 0; i < 100; i++) {
		EXPECT_LE(matrix[i], 100);
		EXPECT_GE(matrix[i], -100);
	}
}
