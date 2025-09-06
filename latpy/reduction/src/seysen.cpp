#include "reduction.h"

#include <iostream>

#include <eigen3/Eigen/Dense>

extern "C" void seysen(long **basis_ptr, const long n, const long m)
{
    long i, j;

    basis = MatrixXli::Zero(n, m);

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis.coeffRef(i, j) = basis_ptr[i][j];
        }
    }

    computeR(basis);
    std::cout << R << std::endl;
    basis = seysenUnimodular(R, n, m) * basis;

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis_ptr[i][j] = basis.coeff(i, j);
        }
    }
}
