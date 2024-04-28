#include <iostream>
#include <chrono>
#include <armadillo>

int main(void)
{
	std::cout << "Test arma 1" << std::endl;

    int count = 2000;
    int col_size = 20;
    int row_size = 4000;

    arma::mat random_matrix = arma::randu(row_size, col_size);
    arma::mat random_matrix_t = random_matrix.t();
    
    std::vector<arma::vec> random_vectors(row_size); 
    arma::vec arma_output1;

    std::vector<double> output2(row_size);
    std::vector<double> output3(row_size);

    for (auto i = 0; i < row_size; ++i) {
        random_vectors[i] = random_matrix_t.col(i);
    }
    
    // test1 
    std::chrono::system_clock::time_point start1 = std::chrono::system_clock::now();
    for (auto i = 0; i < count; ++i) {
        arma::vec coeff = arma::ones(col_size);
        arma_output1 = random_matrix * coeff;         
    }
    std::chrono::duration<double> sec1 = std::chrono::system_clock::now() - start1;
    std::cout << "test1 duration: " << sec1.count() << " s" << std::endl;

    // test2
    std::chrono::system_clock::time_point start2 = std::chrono::system_clock::now();
    for (auto i = 0; i < count; ++i) {
        arma::vec coeff = arma::ones(col_size);
        for (auto j = 0; j < row_size; ++j) {
            output2[j] = arma::dot(coeff, random_vectors[j]);
        }
    }
    std::chrono::duration<double> sec2 = std::chrono::system_clock::now() - start2;
    std::cout << "test2 duration: " << sec2.count() << " s" << std::endl;

    // test3
    std::chrono::system_clock::time_point start3 = std::chrono::system_clock::now();
    for (auto i = 0; i < count; ++i) {
        arma::vec coeff = arma::ones(col_size);
        for (auto j = 0; j < row_size; ++j) {
            output3[j] = arma::dot(coeff, random_matrix_t.col(j));
        }
    }
    std::chrono::duration<double> sec3 = std::chrono::system_clock::now() - start3;
    std::cout << "test3 duration: " << sec3.count() << " s" << std::endl;

	return 0;
}
