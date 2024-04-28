//#define EIGEN_USE_MKL_ALL
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <chrono>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(void)
{
	std::cout << "Test Eigen 1" << std::endl;
    
    int count = 2000;
    int col_size = 20;
    int row_size = 4000;

    MatrixXd random_matrix = Eigen::MatrixXd::Random(row_size, col_size);
    MatrixXd random_matrix_t = random_matrix.transpose();
    
    std::vector<VectorXd> random_vectors(row_size);
    VectorXd output1;
    std::vector<double> output2(row_size);
    std::vector<double> output3(row_size);

    for (auto i = 0; i < row_size; ++i) { 
        VectorXd vec = Eigen::VectorXd::Random(col_size);
        random_vectors[i] = vec;
    }

    // test1
    std::chrono::system_clock::time_point start1 = std::chrono::system_clock::now();
    for (auto i = 0; i < count; ++i) {
        VectorXd coeff = Eigen::VectorXd::Ones(col_size);
        output1 = random_matrix * coeff;  
    }
    std::chrono::duration<double> sec1 = std::chrono::system_clock::now() - start1;
    std::cout << "test1 duration: " << sec1.count() << " s" <<  std::endl;

    // test2
    std::chrono::system_clock::time_point start2 = std::chrono::system_clock::now();
    for (auto i = 0; i < count; ++i) {
        VectorXd coeff = Eigen::VectorXd::Ones(col_size);
        for (auto j = 0; j < row_size; ++j) {
            output2[j] = coeff.dot(random_vectors[j]);
        }
    }
    std::chrono::duration<double> sec2 = std::chrono::system_clock::now() - start2;
    std::cout << "test2 duration: " << sec2.count() << " s" << std::endl;

    // test3
    std::chrono::system_clock::time_point start3 = std::chrono::system_clock::now();
    for (auto i = 0; i < count; ++i) {
        VectorXd coeff = Eigen::VectorXd::Ones(col_size);
        for (auto j = 0; j < row_size; ++j ) {
            VectorXd col = random_matrix_t.col(j);
            output3[j] = coeff.dot(col);
        }
    }
    std::chrono::duration<double> sec3 = std::chrono::system_clock::now() - start3;
    std::cout << "test3 duration: " << sec3.count() << " s" << std::endl;
    
//    bool all_same = true;
//    auto std_output1 = std::vector<double>(output1.data(), output1.data() + output1.size());
//    for (auto i = 0; i < row_size; ++i){
//        std::cout << "i output1, output2: " << i << " " << std_output1[i] << " " << output3[i] << std::endl;
//        all_same = all_same and (std_output1[i] == output3[i]);
//    }
//    
//    if (all_same) { std::cout << "all same" << std::endl; }
//    else { std::cout << "test failed" << std::endl; }

	return 0;
}
