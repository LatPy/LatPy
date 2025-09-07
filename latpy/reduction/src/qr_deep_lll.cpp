#include "reduction.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>

void qrDeepLLL(
    long **basis_ptr,
    const double delta,
    const double eta,
    const long gamma,
    const bool output_sl,
    const bool output_rhf,
    const long n,
    const long m)
{
    bool flag;
    long q, i, j, k;
    long double C;
    const double eta_hat = (eta + 0.5) * 0.5;
    FILE *log_sl, *log_rhf;
    NTL::mat_ZZ basis_ntl;

    basis_ntl.SetDims(n, m);
    basis = MatrixXli::Zero(n, m);

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis.coeffRef(i, j) = basis_ptr[i][j];
            basis_ntl[i][j] = NTL::to_ZZ(basis_ptr[i][j]);
        }
    }

    if (n == m)
    {
        volume = NTL::abs(NTL::determinant(basis_ntl));
    }
    else
    {
        volume = NTL::SqrRoot(NTL::determinant(basis_ntl * NTL::transpose(basis_ntl)));
    }

    computeR(basis);

    if (output_sl)
    {
        log_sl = fopen("sl_log.csv", "w");
        fprintf(log_sl, "val\n");
    }
    if (output_rhf)
    {
        log_rhf = fopen("rhf_log.csv", "w");
        fprintf(log_rhf, "val\n");
    }

    for (k = 1; k < n;)
    {
        for (j = k - 1; j > -1; --j)
        {
            if (fabsl(R.coeff(k, j)) > eta_hat * fabsl(R.coeff(j, j)))
            {
                q = static_cast<long>(roundf(R.coeff(k, j) / R.coeff(j, j)));
                basis.row(k) -= q * basis.row(j);
                R.row(k).head(j + 1) -= static_cast<long double>(q) * R.row(j).head(j + 1);
            }
        }

        flag = false;
        C = basis.row(k).squaredNorm();
        for (i = 0; i < k;)
        {
            if (C < delta * R.coeff(i, i) * R.coeff(i, i))
            {
                if ((i <= gamma) and (k - i + 1 <= gamma))
                {
                    flag = true;
                }
            }

            if (not flag)
            {
                C -= R.coeff(k, i) * R.coeff(k, i);
                ++i;
            }
            else
            {
                deepInsertion(i, k);

                updateDeepInsertionR(i, k, n);

                if (i >= 1)
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

        if (output_sl)
        {
            B = R.diagonal().array().pow(2);
            fprintf(log_sl, "%Lf\n", sl(n));
        }
        if (output_rhf)
        {
            fprintf(log_rhf, "%Lf\n", rhf(n));
        }
    }

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis_ptr[i][j] = basis.coeff(i, j);
        }
    }

    if (output_sl)
    {
        fclose(log_sl);
    }
    if (output_rhf)
    {
        fclose(log_rhf);
    }
}
