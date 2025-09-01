#ifndef LAT_PY_H
#define LAT_PY_H

#include <iostream>

#include <eigen3/Eigen/Dense>

typedef Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXli;        // long-type matrix
typedef Eigen::Matrix<long, 1, Eigen::Dynamic> VectorXli;                                      // long-type vector
typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXld; // long double-type matrix
typedef Eigen::Matrix<long double, 1, Eigen::Dynamic> VectorXld;                               // long double-type vector

extern VectorXld B;
extern MatrixXli basis;
extern MatrixXld mu;

extern "C" void helloPrint();

extern "C" void computeGSO(long** basis_ptr, double **mu_ptr, double *B_ptr, const long n, const long m);

#endif // !LAT_PY_H
