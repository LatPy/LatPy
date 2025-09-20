#include "reduction.h"

#include <iostream>
#include <cmath>

#include <NTL/ZZ.h>
#include <NTL/vec_ZZ.h>
#include <NTL/mat_ZZ.h>

#include "core.h"

void updateDeepInsertionFracGSO(const long i, const long k, const long n)
{
    NTL::ZZ d;
    NTL::vec_ZZ C_num = B_num, C_den = B_den;
    NTL::mat_ZZ nu_num = mu_num, nu_den = mu_den;

    for (long l = k, j; l > i; --l)
    {
        C_num[l - 1] = B_num[l] * B_den[l - 1] * mu_den[l][l - 1] * mu_den[l][l - 1] + mu_num[l][l - 1] * mu_num[l][l - 1] * B_den[l] * B_num[l - 1];
        C_den[l - 1] = B_den[l] * B_den[l - 1] * mu_den[l][l - 1] * mu_den[l][l - 1];
        d = NTL::GCD(C_num[l - 1], C_den[l - 1]);
        C_num[l - 1] /= d;
        C_den[l - 1] /= d;
        C_num[l] = B_num[l - 1] * B_num[l] * C_den[l - 1];
        C_den[l] = B_den[l - 1] * B_den[l] * C_num[l - 1];
        d = NTL::GCD(C_num[l], C_den[l]);
        C_num[l] /= d;
        C_den[l] /= d;

        nu_num[l][l - 1] = mu_num[l][l - 1] * B_num[l - 1] * C_den[l - 1];
        nu_den[l][l - 1] = mu_den[l][l - 1] * B_den[l - 1] * C_num[l - 1];
        d = NTL::GCD(nu_num[l][l - 1], nu_den[l][l - 1]);
        nu_num[l][l - 1] /= d;
        nu_den[l][l - 1] /= d;
        for (j = 0; j < l - 1; ++j)
        {
            nu_num[l - 1][j] = mu_num[l][j];
            nu_den[l - 1][j] = mu_den[l][j];
            d = NTL::GCD(nu_num[l - 1][j], nu_den[l - 1][j]);
            nu_num[l - 1][j] /= d;
            nu_den[l - 1][j] /= d;
            nu_num[l][j] = mu_num[l - 1][j];
            nu_den[l][j] = mu_den[l - 1][j];
            d = NTL::GCD(nu_num[l][j], nu_den[l][j]);
            nu_num[l][j] /= d;
            nu_den[l][j] /= d;
        }
        for (j = l + 1; j < n; ++j)
        {
            nu_num[j][l] = mu_num[j][l - 1] * mu_den[l][l - 1] * mu_den[j][l] - mu_num[l][l - 1] * mu_num[j][l] * mu_den[j][l - 1];
            nu_den[j][l] = mu_den[j][l - 1] * mu_den[l][l - 1] * mu_den[j][l];
            d = NTL::GCD(nu_num[j][l], nu_den[j][l]);
            nu_num[j][l] /= d;
            nu_den[j][l] /= d;
            nu_num[j][l - 1] = mu_num[j][l] * nu_den[l][l - 1] * nu_den[j][l] + mu_den[j][l] * nu_num[l][l - 1] * nu_num[j][l];
            nu_den[j][l - 1] = mu_den[j][l] * nu_den[l][l - 1] * nu_den[j][l];
            d = NTL::GCD(nu_num[j][l - 1], nu_den[j][l - 1]);
            nu_num[j][l - 1] /= d;
            nu_den[j][l - 1] /= d;
        }

        B_num = C_num;
        B_den = C_den;
        mu_num = nu_num;
        mu_den = nu_den;
    }
}
