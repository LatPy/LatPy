#include "reduction.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

MatrixXli seysenUnimodular(const MatrixXld R_, const long n, const long m)
{
    if (n == 1)
    {
        return MatrixXli::Ones(1, 1);
    }
    else
    {
        const long n_half = floorf(n * 0.5);
        MatrixXli U = MatrixXli::Zero(n, n), U11, U21, U22;
        MatrixXld R11, R21, R22;
        R11 = R_.block(0, 0, n_half, n_half);
        R21 = R_.block(n_half, 0, n - n_half, n_half);
        R22 = R_.block(n_half, n_half, n - n_half, n - n_half);

        U11 = seysenUnimodular(R11, n_half, n_half);
        U22 = seysenUnimodular(R22, n - n_half, n - n_half);

        R21 = U22.cast<long double>() * R21;
        U21 = (-R21 * R11.inverse()).array().round().cast<long>();
        R21 += U21.cast<long double>() * R11;
        U21 = U21 * U11;

        for (long i = 0, j; i < n; ++i)
        {
            for (j = 0; j < n; ++j)
            {
                if ((i < n_half) and (j < n_half))
                {
                    U.coeffRef(i, j) = U11.coeff(i, j);
                }
                else if ((i >= n_half) and (j < n_half))
                {
                    U.coeffRef(i, j) = U21.coeff(i - n_half, j);
                }
                else if ((i >= n_half) and (j >= n_half))
                {
                    U.coeffRef(i, j) = U22.coeff(i - n_half, j - n_half);
                }
                else
                {
                    U.coeffRef(i, j) = 0;
                }
            }
        }
        return U;
    }
}
