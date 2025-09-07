#include "reduction.h"

#include <iostream>
#include <cmath>
#include <vector>

#include <eigen3/Eigen/Dense>

void updateDeepInsertionR(const long i, const long k, const long n)
{
    const VectorXld tmp = R.diagonal();
    VectorXld D(n), P(n), S = VectorXld::Zero(n), G = R.diagonal();
    long double T, t;
    long j, g;

    D.coeffRef(k) = R.coeff(k, k) * R.coeff(k, k);
    P.coeffRef(k) = R.coeff(k, k);
    for (long j = k - 1; j >= i; --j)
    {
        P.coeffRef(j) = R.coeff(k, j);
        D.coeffRef(j) = D.coeff(j + 1) + R.coeff(k, j) * R.coeff(k, j);
    }

    // dianonal elements
    for (j = k; j > i; --j)
    {
        G.coeffRef(j) = G.coeff(j - 1) * sqrtl(D.coeff(j) / D.coeff(j - 1));
    }
    G.coeffRef(i) = sqrtl(D.coeff(i));

    // lower triangle elements
    for (j = k; j > i; --j)
    {
        T = R.coeff(k, j - 1) / (R.coeff(j - 1, j - 1) * D.coeff(j));
        for (g = n - 1; g > k; --g)
        {
            S.coeffRef(g) += R.coeff(g, j) * P.coeff(j);
            R.coeffRef(g, j) = (G.coeff(j) * R.coeff(g, j - 1)) / R.coeff(j - 1, j - 1) - G.coeff(j) * T * S.coeff(g);
        }
        for (g = k; g > j; --g)
        {
            S.coeffRef(g) += R.coeff(g - 1, j) * P.coeff(j);
            R.coeffRef(g, j) = (G.coeff(j) * R.coeff(g - 1, j - 1)) / R.coeff(j - 1, j - 1) - G.coeff(j) * T * S.coeff(g);
        }
    }

    T = G.coeff(i) / D.coeff(i);
    for (g = n - 1; g > k; --g)
    {
        R.coeffRef(g, i) = T * (S.coeff(g) + R.coeff(g, i) * P.coeff(i));
    }
    for (long g = k; g > i + 1; --g)
    {
        R.coeffRef(g, i) = T * (S.coeff(g) + R.coeff(g - 1, i) * P.coeff(i));
    }
    R.coeffRef(i + 1, i) = T * P.coeff(i) * R.coeff(i, i);

    for (j = 0; j < i; ++j)
    {
        t = (G.coeff(j) * R.coeff(k, j)) / R.coeff(j, j);
        for (g = k; g > i; --g)
        {
            R.coeffRef(g, j) = (G.coeff(j) * R.coeff(g - 1, j)) / R.coeff(j, j);
        }
        R.coeffRef(i, j) = t;
    }

    for (j = i; j <= k; ++j)
    {
        R.coeffRef(j, j) = G.coeff(j);
    }
}
