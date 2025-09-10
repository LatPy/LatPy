#include "reduction.h"

#include <iostream>
#include <cmath>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "core.h"

void updateDeepInsertionGSO(const long i, const long k, const long n)
{
    long j, l;
    long double t, eps;
    std::vector<long double> P(n, 0), D(n, 0), S(n, 0);

    P[k] = D[k] = B.coeff(k);
    for (j = k - 1; j >= i; --j)
    {
        P[j] = mu.coeff(k, j) * B.coeff(j);
        D[j] = D[j + 1] + mu.coeff(k, j) * P[j];
    }

    for (j = k; j > i; --j)
    {
        t = mu.coeff(k, j - 1) / D[j];
        for (l = n - 1; l > k; --l)
        {
            S[l] += mu.coeff(l, j) * P[j];
            mu.coeffRef(l, j) = mu.coeff(l, j - 1) - t * S[l];
        }
        for (l = k; l > j; --l)
        {
            S[l] += mu.coeff(l - 1, j) * P[j];
            mu.coeffRef(l, j) = mu.coeff(l - 1, j - 1) - t * S[l];
        }
    }

    t = 1.0 / D[i];

    for (l = n - 1; l > k; --l)
    {
        mu.coeffRef(l, i) = t * (S[l] + mu.coeff(l, i) * P[i]);
    }
    for (l = k; l >= i + 2; --l)
    {
        mu.coeffRef(l, i) = t * (S[l] + mu.coeff(l - 1, i) * P[i]);
    }

    mu.coeffRef(i + 1, i) = t * P[i];
    for (j = 0; j < i; ++j)
    {
        eps = mu.coeff(k, j);
        for (l = k; l > i; --l)
        {
            mu.coeffRef(l, j) = mu.coeff(l - 1, j);
        }
        mu.coeffRef(i, j) = eps;
    }

    for (j = k; j > i; --j)
    {
        B.coeffRef(j) = D[j] * B.coeff(j - 1) / D[j - 1];
    }
    B.coeffRef(i) = D[i];
}
