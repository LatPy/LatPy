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
        MatrixXli U = MatrixXli::Zero(n, n), U1, W, U2;
        MatrixXld A, C, D, A_;
        A = R_.block(0, 0, n_half, n_half);
        C = R_.block(n_half, 0, n - n_half, n_half);
        D = R_.block(n_half, n_half, n - n_half, n - n_half);

        U1 = seysenUnimodular(A, n_half, n_half);
        U2 = seysenUnimodular(D, n - n_half, n - n_half);

        A_ = (U1.cast<long double>() * A).inverse();
        W = (U2.cast<long double>() * C * A_).array().round().cast<long>();

        U.block(0, 0, n_half, n_half) = U1;
        U.block(n_half, 0, n - n_half, n_half) = -W * U1;
        U.block(n_half, n_half, n - n_half, n - n_half) = U2;

        return U;
    }
}
