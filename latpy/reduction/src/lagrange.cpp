#include "reduction.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

extern "C" void lagrange(long **basis_ptr, const long n, const long m)
{
    long i, j, q;
    VectorXli v;

    basis = MatrixXli::Zero(n, m);
    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis.coeffRef(i, j) = basis_ptr[i][j];
        }
    }

    if (basis.row(0).squaredNorm() > basis.row(1).squaredNorm())
    {
        basis.row(0).swap(basis.row(1));
    }
    do
    {
        q = -round(static_cast<double>(basis.row(0).dot(basis.row(1))) / basis.row(0).squaredNorm());
        v = basis.row(1) + q * basis.row(0);
        basis.row(1) = basis.row(0);
        basis.row(0) = v;
    } while (basis.row(0).squaredNorm() < basis.row(1).squaredNorm());
    basis.row(0).swap(basis.row(1));

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis_ptr[i][j] = basis.coeff(i, j);
        }
    }
}
