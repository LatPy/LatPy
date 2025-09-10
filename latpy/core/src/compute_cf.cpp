#include "core.h"

#include <iostream>

#include <eigen3/Eigen/Dense>

void computeCF(const long n, const long m)
{
    R = MatrixXld::Zero(n, m);
    mu = MatrixXld::Identity(n, n);
    s = VectorXld::Zero(n);

    for (long i = 0, j, k; i < n; ++i)
    {
        for (j = 0; j < i; ++j)
        {
            R.coeffRef(i, j) = basis.row(i).dot(basis.row(j));
            for (k = 0; k < j; ++k)
            {
                R.coeffRef(i, j) -= R.coeff(i, k) * mu.coeff(j, k);
            }
            mu.coeffRef(i, j) = R.coeff(i, j) / R.coeff(j, j);
        }

        s.coeffRef(0) = basis.row(i).squaredNorm();
        for (j = 1; j <= i; ++j)
        {
            s.coeffRef(j) = s.coeff(j - 1) - mu.coeff(i, j - 1) * R.coeff(i, j - 1);
        }
        R.coeffRef(i, i) = s.coeff(i);
    }
}
