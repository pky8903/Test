#include <iostream>
#include <eigen3/Eigen/Dense>
#include <mkl.h>

using Eigen::MatrixXd;

int main(void)
{
//	std::cout << "Test Eigen" << std::endl;
//    
//    MatrixXd m(2, 2); 
//    m(0, 0) = 3;
//    m(1, 0) = 2.5;
//    m(0, 1) = -1;
//    m(1, 1) = m(1, 0) + m(0, 1);
//
//    std::cout << m << std::endl;

    const int m = 3;
    const int n = 3;
    const int k = 3;
    
    double A[m*k] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double B[k*n] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    double C[m*n] = {0};

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k, 1.0, A, k, B, n, 0.0, C, n);

    std::cout << "Result Matrix: " << std::endl;    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << C[i*n + j] << "\t";
        }
        std::cout << std::endl;
    }

	return 0;
}
