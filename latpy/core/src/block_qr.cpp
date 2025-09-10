#include "core.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

void blockQR(const long k, const bool is_shifted, MatrixXld& Q, MatrixXld& B_star)
{
    if (k == 0)
    {
        B_star.row(0) = basis.row(0).cast<long double>();
        R.coeffRef(0, 0) = B_star.row(0).norm();
        Q.row(0) = B_star.row(0) / R.coeff(0, 0);
    }
    else if ((k == 1) && is_shifted)
    {
        B_star.row(0) = basis.row(0).cast<long double>();
        R.coeffRef(0, 0) = B_star.row(0).norm();
        Q.row(0) = B_star.row(0) / R.coeff(0, 0);

        B_star.row(1) = basis.row(1).cast<long double>();
        R.coeffRef(1, 0) = B_star.row(1).dot(Q.row(0));
        B_star.row(1) -= R.coeff(1, 0) * Q.row(0);
        R.coeffRef(1, 1) = B_star.row(1).norm();
        Q.row(1) = B_star.row(1) / R.coeff(1, 1);
    }
    else
    {
        B_star.row(k) = basis.row(k).cast<long double>();
        for (long i = 0; i < k; ++i)
        {
            R.coeffRef(k, i) = B_star.row(k).dot(Q.row(i));
            B_star.row(k) -= R.coeff(k, i) * Q.row(i);
        }
        R.coeffRef(k, k) = B_star.row(k).norm();
        Q.row(k) = B_star.row(k) / R.coeff(k, k);
    }

    s.coeffRef(0) = basis.row(k).squaredNorm();
    for (long h = 1; h < k; ++h)
    {
        s.coeffRef(h) = s.coeff(h - 1) - R.coeff(k, h - 1) * R.coeff(k, h - 1);
    }
}
