#include "reduction.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include "core.h"

void updateDualDeepInsertionGSO(const long k, const long l, const VectorXld hD, const long n)
{
    long i, j, h;
    long double sum;
    MatrixXld xi = mu;

    for (i = l + 1; i < n; ++i)
    {
        sum = 0.0L;
        for (h = k; h <= l; ++h)
        {
            sum += hmu.coeff(k, h) * mu.coeff(i, h);
        }
        xi.coeffRef(i, l) = sum;
    }

    for (j = k; j < l; ++j)
    {
        for (i = j + 1; i < l; ++i)
        {
            sum = 0.0L;
            for (h = k; h <= j; ++h)
            {
                sum += hmu.coeff(k, h) * mu.coeff(i + 1, h);
            }

            xi.coeffRef(i, j) = (mu.coeff(i + 1, j + 1) * hD.coeff(j)) / hD.coeff(j + 1) - (hmu.coeff(k, j + 1) * sum) / (hD.coeff(j + 1) * B.coeff(j + 1));
        }
        xi.coeffRef(l, j) = -hmu.coeff(k, j + 1) / (hD.coeff(j + 1) * B.coeff(j + 1));
        for (i = l + 1; i < n; ++i)
        {
            sum = 0.0L;
            for (h = k; h <= j; ++h)
            {
                sum += hmu.coeff(k, h) * mu.coeff(i, h);
            }

            xi.coeffRef(i, j) = (mu.coeff(i, j + 1) * hD.coeff(j)) / hD.coeff(j + 1) - (hmu.coeff(k, j + 1) * sum) / (hD.coeff(j + 1) * B.coeff(j + 1));
        }
    }

    for (j = 0; j < k; ++j)
    {
        for (i = k; i < l; ++i)
        {
            xi.coeffRef(i, j) = mu.coeff(i + 1, j);
        }
        xi.coeffRef(l, j) = mu.coeff(k, j);
    }

    mu = xi;

    for (j = k; j < l; ++j)
    {
        B.coeffRef(j) = hD.coeff(j + 1) * B.coeff(j + 1) / hD.coeff(j);
    }
    B.coeffRef(l) = 1.0L / hD.coeff(l);
}
