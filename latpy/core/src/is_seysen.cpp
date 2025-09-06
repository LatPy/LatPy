#include "core.h"

#include <iostream>

bool isSeysen(const MatrixXld R)
{
    if (R.rows() == 1)
    {
        return true;
    }
    else
    {
        long i, j;
        const long n = R.rows();
        const long n_half = floorf(n * 0.5);
        MatrixXli U = MatrixXli::Zero(n, n), U11, U21, U22;
        MatrixXld R11, R21, R22;
        R11 = R.block(0, 0, n_half, n_half);
        R21 = R.block(n_half, 0, n - n_half, n_half);
        R22 = R.block(n_half, n_half, n - n_half, n - n_half);

        MatrixXld R_ = R21 * R11.inverse();
        long double max = -1;
        for (i = 0; i < n; ++i)
        {
            for (j = 0; j < n; ++j)
            {
                if (fabsl(R_.coeff(i, j)) > max)
                {
                    max = fabsl(R_.coeff(i, j));
                }
            }
        }

        return (max < 0.5) and  isSeysen(R11) and isSeysen(R22);
    }
}

extern "C" bool isSeysen(long **basis_ptr, const long n, const long m)
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

    return isSeysen(computeR(basis));
}
