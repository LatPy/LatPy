#include "reduction.h"

#include <iostream>

#include <eigen3/Eigen/Dense>

void computeGSO(MatrixXli basis_)
{
    const long n = basis_.rows(), m = basis_.cols();
    long i, j;
    MatrixXld gso_basis(n, m);

    mu = MatrixXld::Identity(n, n);
    B = VectorXld::Zero(n);

    for (i = 0; i < n; ++i)
    {
        mu.coeffRef(i, i) = 1;
        gso_basis.row(i) = basis.row(i).cast<long double>();

        for (j = 0; j < i; ++j)
        {
            mu.coeffRef(i, j) = basis.row(i).cast<long double>().dot(gso_basis.row(j)) / B.coeff(j);
            gso_basis.row(i) -= mu.coeff(i, j) * gso_basis.row(j);
        }
        B.coeffRef(i) = gso_basis.row(i).squaredNorm();
    }
}
