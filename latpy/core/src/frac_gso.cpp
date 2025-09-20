#include "core.h"

#include <iostream>
#include <cmath>

#include <NTL/ZZ.h>
#include <NTL/vec_ZZ.h>
#include <NTL/mat_ZZ.h>

void fracGSO(NTL::mat_ZZ b)
{
    const long n = b.NumRows();
    long i, j, k, h;
    NTL::ZZ sum, g;
    NTL::mat_ZZ D, d;
    NTL::mat_ZZ G = b * NTL::transpose(b);
    
    B_num.SetLength(n);
    B_den.SetLength(n);
    mu_num.SetDims(n, n);
    mu_den.SetDims(n, n);
    D.SetDims(n, n);

    for (i = 0; i < n; ++i)
    {
        mu_num[i][i] = 1;
        for (j = 0; j < n; ++j)
        {
            mu_den[i][j] = 1;
        }
    }

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            if ((i == 0) && (j == 0))
            {
                D[i][j] = 1;
            }
            else
            {
                d.SetDims(i, i);
                for (k = 0; k < i; ++k)
                {
                    for (h = 0; h < i; ++h)
                    {
                        if (k < j)
                        {
                            d[k][h] = G[k][h];
                        }
                        else
                        {
                            d[k][h] = G[k + 1][h];
                        }
                    }
                }
                D[i][j] = NTL::determinant(d);
            }
        }
    }

    for (i = 0; i < n; ++i)
    {
        sum = 0;
        for (j = 0; j <= i; ++j)
        {
            for (k = 0; k <= i; ++k)
            {
                if ((j + k) % 2 == 0)
                {
                    sum += D[i][j] * D[i][k] * G[j][k];
                }
                else
                {
                    sum -= D[i][j] * D[i][k] * G[j][k];
                }
            }
        }
        B_num[i] = sum;
        B_den[i] = D[i][i] * D[i][i];
        g = NTL::GCD(B_num[i], B_den[i]);
        B_num[i] /= g;
        B_den[i] /= g;

        for (j = 0; j < i; ++j)
        {
            sum = 0;
            for (k = 0; k <= j; ++k)
            {
                if ((j + k) & 1)
                {
                    sum -= D[j][k] * G[i][k];
                }
                else
                {
                    sum += D[j][k] * G[i][k];
                }
            }
            mu_num[i][j] = D[j][j] * sum;

            sum = 0;
            for (k = 0; k <= j; ++k)
            {
                for (h = 0; h <= j; ++h)
                {
                    if ((k + h) & 1)
                    {
                        sum -= D[j][k] * D[j][h] * G[k][h];
                    }
                    else
                    {
                        sum += D[j][k] * D[j][h] * G[k][h];
                    }
                }
            }
            mu_den[i][j] = sum;

            g = NTL::GCD(mu_num[i][j], mu_den[i][j]);
            mu_num[i][j] /= g;
            mu_den[i][j] /= g;
        }
    }
}
