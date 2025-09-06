#include "reduction.h"

#include <iostream>

#include <eigen3/Eigen/Dense>

void computeR(MatrixXli basis_)
{
    const long n = basis_.rows(), m = basis_.cols();
    MatrixXld Q = MatrixXld::Zero(n, m);
    R = MatrixXld::Zero(n, n);
    VectorXld v;

    for (long i = 0; i < n; ++i)
    {
        v = basis_.row(i).cast<long double>();
        R.row(i).head(i) = Q.block(0, 0, i, n) * v.transpose();
        v -= (Q.block(0, 0, i, n).transpose() * R.row(i).head(i).transpose());
        R.coeffRef(i, i) = v.norm();
        Q.row(i) = v / R.coeff(i, i);
    }
}
