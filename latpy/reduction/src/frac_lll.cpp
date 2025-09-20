#include "reduction.h"

#include <iostream>
#include <cmath>

#include <NTL/ZZ.h>
#include <NTL/vec_ZZ.h>
#include <NTL/mat_ZZ.h>
#include <NTL/RR.h>

#include "core.h"

void fracLLL(
    long **basis_ptr,
    const long a,
    const long b,
    const long n,
    const long m)
{
    long i, j, k, h;

    NTL::ZZ d, q;
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

        if (B_num[k] * B_den[k - 1] * b * mu_den[k][k - 1] * mu_den[k][k - 1] >= (a * mu_den[k][k - 1] * mu_den[k][k - 1] - b * mu_num[k][k - 1] * mu_num[k][k - 1]) * B_num[k - 1] * B_den[k])
        {
            ++k;
        }
        else
        {
            basis_ntl[k].swap(basis_ntl[k - 1]);
            updateSwapFracGSO(k, n);

            if (k > 2)
            {
                --k;
            }
            else
            {
                k = 1;
            }
        }
    }

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis_ptr[i][j] = NTL::to_long(basis_ntl[i][j]);
        }
    }
}
