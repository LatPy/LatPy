#include "reduction.h"

#include <iostream>
#include <cmath>

#include <NTL/ZZ.h>
#include <NTL/vec_ZZ.h>
#include <NTL/mat_ZZ.h>
#include <NTL/RR.h>

#include "core.h"

void fracDeepLLL(
    long **basis_ptr,
    const long a,
    const long b,
    const long n,
    const long m)
{
    long i, j, k, h;
    NTL::ZZ d, q;
    NTL::ZZ C_num, C_den;
    NTL::vec_ZZ t;
    NTL::mat_ZZ basis_ntl;

    basis_ntl.SetDims(n, m);
    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis_ntl[i][j] = NTL::to_ZZ(basis_ptr[i][j]);
        }
    }

    fracGSO(basis_ntl);

    for (k = 1; k < n;)
    {
        for (j = k - 1; j > -1; --j)
        {
            if ((NTL::abs(mu_num[k][j]) << 1) > NTL::abs(mu_den[k][j]))
            {
                q = NTL::RoundToZZ(NTL::to_RR(mu_num[k][j]) / NTL::to_RR(mu_den[k][j]));
                basis_ntl[k] -= q * basis_ntl[j];

                for (h = 0; h <= j; ++h)
                {
                    mu_num[k][h] = mu_num[k][h] * mu_den[j][h] - q * mu_den[k][h] * mu_num[j][h];
                    mu_den[k][h] *= mu_den[j][h];
                    d = NTL::GCD(mu_num[k][h], mu_den[k][h]);
                    mu_num[k][h] /= d;
                    mu_den[k][h] /= d;
                }
            }
        }

        NTL::InnerProduct(C_num, basis_ntl[k], basis_ntl[k]);
        C_den = 1;
        for (i = 0; i < k;)
        {
            if (b * C_num * B_den[i] >= a * C_den * B_num[i])
            {
                C_num = C_num * mu_den[k][i] * mu_den[k][i] * B_den[i] - C_den * mu_num[k][i] * mu_num[k][i] * B_num[i];
                C_den = C_den * mu_den[k][i] * mu_den[k][i] * B_den[i];
                d = NTL::GCD(C_num, C_den);
                C_num /= d;
                C_den /= d;
                ++i;
            }
            else
            {
                t = basis_ntl[k];
                for (j = k; j > i; --j)
                {
                    basis_ntl[j] = basis_ntl[j - 1];
                }
                basis_ntl[i] = t;
                updateDeepInsertionFracGSO(i, k, n);

                if (i > 1)
                {
                    k = i - 1;
                }
                else
                {
                    k = 0;
                }
            }
        }
        ++k;
    }

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis_ptr[i][j] = NTL::to_long(basis_ntl[i][j]);
        }
    }
}
