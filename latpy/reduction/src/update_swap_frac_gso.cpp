#include "reduction.h"

#include <iostream>
#include <cmath>

#include <NTL/ZZ.h>
#include <NTL/vec_ZZ.h>
#include <NTL/mat_ZZ.h>

#include "core.h"

void updateSwapFracGSO(const long k, const long n)
{
    long i, j;
    NTL::ZZ d;
    NTL::vec_ZZ C_num = B_num, C_den = B_den;
    NTL::mat_ZZ nu_num = mu_num, nu_den = mu_den;

    C_num[k - 1] = B_num[k] * B_den[k - 1] * mu_den[k][k - 1] * mu_den[k][k - 1] + mu_num[k][k - 1] * mu_num[k][k - 1] * B_den[k] * B_num[k - 1];
    C_den[k - 1] = B_den[k] * B_den[k - 1] * mu_den[k][k - 1] * mu_den[k][k - 1];
    d = NTL::GCD(C_num[k - 1], C_den[k - 1]);
    C_num[k - 1] /= d;
    C_den[k - 1] /= d;
    C_num[k] = B_num[k - 1] * B_num[k] * C_den[k - 1];
    C_den[k] = B_den[k - 1] * B_den[k] * C_num[k - 1];
    d = NTL::GCD(C_num[k], C_den[k]);
    C_num[k] /= d;
    C_den[k] /= d;

    nu_num[k][k - 1] = mu_num[k][k - 1] * B_num[k - 1] * C_den[k - 1];
    nu_den[k][k - 1] = mu_den[k][k - 1] * B_den[k - 1] * C_num[k - 1];
    d = NTL::GCD(nu_num[k][k - 1], nu_den[k][k - 1]);
    nu_num[k][k - 1] /= d;
    nu_den[k][k - 1] /= d;
    for (j = 0; j < k - 1; ++j)
    {
        nu_num[k - 1][j] = mu_num[k][j];
        nu_den[k - 1][j] = mu_den[k][j];
        d = NTL::GCD(nu_num[k - 1][j], nu_den[k - 1][j]);
        nu_num[k - 1][j] /= d;
        nu_den[k - 1][j] /= d;
        nu_num[k][j] = mu_num[k - 1][j];
        nu_den[k][j] = mu_den[k - 1][j];
        d = NTL::GCD(nu_num[k][j], nu_den[k][j]);
        nu_num[k][j] /= d;
        nu_den[k][j] /= d;
    }
    for (i = k + 1; i < n; ++i)
    {
        nu_num[i][k] = mu_num[i][k - 1] * mu_den[k][k - 1] * mu_den[i][k] - mu_num[k][k - 1] * mu_num[i][k] * mu_den[i][k - 1];
        nu_den[i][k] = mu_den[i][k - 1] * mu_den[k][k - 1] * mu_den[i][k];
        d = NTL::GCD(nu_num[i][k], nu_den[i][k]);
        nu_num[i][k] /= d;
        nu_den[i][k] /= d;
        nu_num[i][k - 1] = mu_num[i][k] * nu_den[k][k - 1] * nu_den[i][k] + mu_den[i][k] * nu_num[k][k - 1] * nu_num[i][k];
        nu_den[i][k - 1] = mu_den[i][k] * nu_den[k][k - 1] * nu_den[i][k];
        d = NTL::GCD(nu_num[i][k - 1], nu_den[i][k - 1]);
        nu_num[i][k - 1] /= d;
        nu_den[i][k - 1] /= d;
    }

    B_num = C_num;
    B_den = C_den;
    mu_num = nu_num;
    mu_den = nu_den;
}
