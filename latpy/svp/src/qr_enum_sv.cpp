#include "svp.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include "core.h"

bool qrEnumSV(VectorXli &coeff, const double radius_, const bool pruning, const long start, const long end)
{
    const long n = end - start;
    bool has_solution = false;
    long i, j, r[n + 1];
    long last_nonzero = 0;
    long double temp;
    long width[n];
    VectorXli temp_vector = VectorXli::Zero(n);
    double radius[n];
    long double center[n];
    long double sigma[n + 1][n];
    long rho[n + 1];

    // Set radius for enumeration
    coeffPruning(n, pruning);
    for (i = 0; i < n; ++i)
    {
        radius[i] = eps[n - i - 1] * radius_;
    }

    temp_vector.coeffRef(0) = 1;
    for (i = 0; i < n; ++i)
    {
        r[i] = i;
        width[i] = 0;
        center[i] = 0;
        for (j = 0; j <= n; ++j)
        {
            sigma[j][i] = 0;
        }
        rho[i] = 0;
    }
    rho[n] = 0;

    for (long k = 0;;)
    {
        temp = static_cast<long double>(temp_vector.coeff(k)) - center[k];
        temp *= temp;
        rho[k] = rho[k + 1] + temp * R.coeff(k + start, k + start) * R.coeff(k + start, k + start);

        if (rho[k] <= radius[n - k - 1])
        {
            if (k == 0)
            {
                has_solution = true;
                coeff = temp_vector;
                for (i = 0; i < n; ++i)
                {
                    radius[i] = fminl(0.99 * rho[0], radius[i]);
                }
            }
            else
            {
                --k;
                if (r[k + 1] >= r[k])
                {
                    r[k] = r[k + 1];
                }
                for (i = r[k]; i > k; --i)
                {
                    sigma[i][k] = sigma[i + 1][k] + (R.coeff(i + start, k + start) * temp_vector.coeff(i)) / R.coeff(k + start, k + start);
                }
                center[k] = -sigma[k + 1][k];
                temp_vector.coeffRef(k) = std::lroundl(center[k]);
                width[k] = 1;
            }
        }
        else
        {
            ++k;
            if (k == n)
            {
                return has_solution;
            }
            else
            {
                r[k] = k;
                if (k >= last_nonzero)
                {
                    last_nonzero = k;
                    ++temp_vector.coeffRef(k);
                }
                else
                {
                    if (temp_vector.coeff(k) > center[k])
                    {
                        temp_vector.coeffRef(k) -= width[k];
                    }
                    else
                    {
                        temp_vector.coeffRef(k) += width[k];
                    }
                    ++width[k];
                }
            }
        }
    }
}

extern "C" void qrEnumSV(long **basis_ptr, long *coeff, const bool pruning, const long n, const long m)
{
    bool has_solution = false;
    long i, j, r[n + 1];
    long last_nonzero = 0;
    long double temp;
    long width[n];
    VectorXli temp_vector = VectorXli::Zero(n);
    double radius[n];
    long double center[n];
    long double sigma[n + 1][n];
    long rho[n + 1];

    basis = MatrixXli::Zero(n, m);
    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis.coeffRef(i, j) = basis_ptr[i][j];
        }
    }
    computeR();

    // Set radius for enumeration
    coeffPruning(n, pruning);
    for (i = 0; i < n; ++i)
    {
        radius[i] = eps[n - i - 1] * R.coeff(0, 0) * R.coeff(0, 0);
    }

    temp_vector.coeffRef(0) = 1;
    for (i = 0; i < n; ++i)
    {
        r[i] = i;
        width[i] = 0L;
        center[i] = 0.0L;
        for (j = 0; j <= n; ++j)
        {
            sigma[j][i] = 0.0L;
        }
        rho[i] = 0.0L;
    }
    rho[n] = 0;

    for (long k = 0;;)
    {
        temp = static_cast<long double>(temp_vector.coeff(k)) - center[k];
        temp *= temp;
        rho[k] = rho[k + 1] + temp * R.coeff(k, k) * R.coeff(k, k);

        if (rho[k] <= radius[n - k - 1])
        {
            if (k == 0)
            {
                has_solution = true;

                for (i = 0; i < n; ++i)
                {
                    coeff[i] = temp_vector.coeff(i);
                    radius[i] = fminl(0.99 * rho[0], radius[i]);
                }
            }
            else
            {
                --k;
                if (r[k + 1] >= r[k])
                {
                    r[k] = r[k + 1];
                }
                for (i = r[k]; i > k; --i)
                {
                    sigma[i][k] = sigma[i + 1][k] + (R.coeff(i, k) * temp_vector.coeff(i)) / R.coeff(k, k);
                }
                center[k] = -sigma[k + 1][k];
                temp_vector.coeffRef(k) = std::lroundl(center[k]);
                width[k] = 1;
            }
        }
        else
        {
            ++k;
            if (k == n)
            {
                if (not has_solution)
                {
                    for (i = 0; i < n; ++i)
                    {
                        coeff[i] = 0L;
                    }
                }
                return;
            }
            else
            {
                r[k] = k;
                if (k >= last_nonzero)
                {
                    last_nonzero = k;
                    ++temp_vector.coeffRef(k);
                }
                else
                {
                    if (temp_vector.coeff(k) > center[k])
                    {
                        temp_vector.coeffRef(k) -= width[k];
                    }
                    else
                    {
                        temp_vector.coeffRef(k) += width[k];
                    }
                    ++width[k];
                }
            }
        }
    }
}
