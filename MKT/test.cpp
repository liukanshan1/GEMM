#include<stdio.h>
#include<stdlib.h>
#include<mkl.h>
int main()
{
    float* A, * B;//两个向量
    int a = 1, b = 1;//标量
    int n = 5;//向量⼤⼩
    A = (float*)mkl_malloc(n * 1 * sizeof(float), 64);
    B = (float*)mkl_malloc(n * 1 * sizeof(float), 64);
    printf("The 1st vector is ");
    for (int i = 0; i < n; i++) {
        A[i] = i;
        printf("%2.0f", A[i]);
    }
    printf("\n");
    printf("The 2st vector is ");
    for (int i = 0; i < n; i++) {
        B[i] = i + 1;
        printf("%2.0f", B[i]);
    }
    printf("\n");
    //计算a*A+b*B
    cblas_saxpby(n, a, A, 1, b, B, 1);
    printf("The a*A+b*B is ");
    for (int i = 0; i < n; i++) {
        printf("%2.0f", B[i]);
    }
    printf("\n");
    mkl_free(A);
    mkl_free(B);
    getchar();
    return 0;
}