#include "reduction.h"

#include <iostream>
#include <cmath>
#include <cstdlib>

#include <eigen3/Eigen/Dense>

#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>

extern "C" void LLL(
    long **basis_ptr,
    const double delta,
    const double eta,
    const bool output_sl,
    const bool output_rhf,
    const long n,
    const long m)
{
    long i, j, k, q;
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

    computeGSO(basis);

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
            if (fabsl(mu.coeff(k, j)) > eta_hat)
            {
                q = round(mu.coeff(k, j));
                basis.row(k) -= q * basis.row(j);
                mu.row(k).head(j + 1) -= static_cast<long double>(q) * mu.row(j).head(j + 1);
            }
        }

        if ((k > 0) && (B.coeff(k) < (delta - mu.coeff(k, k - 1) * mu.coeff(k, k - 1)) * B.coeff(k - 1)))
        {
            basis.row(k - 1).swap(basis.row(k));
            updateSwapGSO(k, n);

            --k;
        }
        else
        {
            ++k;
        }

        if (output_sl)
        {
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

void LLL(const double delta, const long end, const long n)
{
    long j, k, q;

    for (k = 1; k < end;)
    {
        for (j = k - 1; j > -1; --j)
        {
            if (fabsl(mu.coeff(k, j)) > 0.5)
            {
                q = round(mu.coeff(k, j));
                basis.row(k) -= q * basis.row(j);
                mu.row(k).head(j + 1) -= static_cast<long double>(q) * mu.row(j).head(j + 1);
            }
        }

        if ((k > 0) && (B.coeff(k) < (delta - mu.coeff(k, k - 1) * mu.coeff(k, k - 1)) * B.coeff(k - 1)))
        {
            basis.row(k - 1).swap(basis.row(k));
            updateSwapGSO(k, n);

            --k;
        }
        else
        {
            ++k;
        }
    }
}
