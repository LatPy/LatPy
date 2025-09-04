#include <iostream>

#include <eigen3/Eigen/Dense>

typedef Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXli;        // long-type matrix
typedef Eigen::Matrix<long, 1, Eigen::Dynamic> VectorXli;                                      // long-type vector
typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXld; // long double-type matrix
typedef Eigen::Matrix<long double, 1, Eigen::Dynamic> VectorXld;                               // long double-type vector

extern VectorXld B;
extern MatrixXli basis;
extern MatrixXld mu;

void computeGSO(MatrixXli basis_, MatrixXld &mu_, VectorXld &B_);

extern "C"
{
    void lagrange(long **basis_ptr, const long n, const long m);
}
