#include "core.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

extern "C" long volume(long **basis_ptr, const long n, const long m)
{
    basis = MatrixXli::Zero(n, m);
    for (long i = 0, j; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis.coeffRef(i, j) = basis_ptr[i][j];
        }
    }

    if(n == m)
    {
        return static_cast<long>(fabs(basis.cast<double>().determinant()));
    }
    else
    {
        return static_cast<long>(sqrt((basis * basis.transpose()).cast<double>().determinant()));
    }
}
