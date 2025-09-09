#include "reduction.h"

#include <iostream>
#include <cmath>

#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>
#include <NTL/RR.h>
#include <NTL/mat_RR.h>
#include <NTL/LLL.h>

#include <eigen3/Eigen/Dense>

extern "C" void deepLLL(
    long **basis_ptr,
    const double delta,
    const double eta,
    const long gamma,
    const bool output_sl,
    const bool output_rhf,
    const bool output_err,
    const long n,
    const long m)
{
    bool flag;
    long i, j, k, q;
    const double eta_hat = (eta + 0.5) * 0.5;
    long double C;
    FILE *log_sl, *log_rhf, *err;
    MatrixXld err_mat = MatrixXld::Zero(n, n);
    NTL::mat_ZZ basis_ntl;
    NTL::vec_RR B_ntl;
    NTL::mat_RR mu_ntl;

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
    if (output_err)
    {
        err = fopen("err.csv", "w");
        fprintf(err, "val\n");
    }

    for (k = 1; k < n;)
    {
        for (j = k - 1; j >= 0; --j)
        {
            if (fabsl(mu.coeff(k, j)) > eta_hat)
            {
                q = static_cast<long>(round(mu.coeff(k, j)));
                basis.row(k) -= q * basis.row(j);

                for (i = 0; i <= j; ++i)
                {
                    mu.coeffRef(k, i) -= mu.coeff(j, i) * static_cast<long double>(q);
                }
            }
        }

        flag = false;
        C = static_cast<long double>(basis.row(k).squaredNorm());
        for (i = 0; i < k;)
        {
            if (C < delta * B.coeff(i))
            {
                if ((i <= gamma) and (k - i + 1 <= gamma))
                {
                    flag = true;
                }
            }

            if (not flag)
            {
                C -= mu.coeff(k, i) * mu.coeff(k, i) * B.coeff(i);
                ++i;
            }
            else
            {
                deepInsertion(i, k);
                updateDeepInsertionGSO(i, k, n);

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
            fprintf(log_sl, "%Lf\n", sl(n));
        }
        if (output_rhf)
        {
            fprintf(log_rhf, "%Lf\n", rhf(n));
        }
    }

    if (output_err)
    {
        for (i = 0; i < n; ++i)
        {
            for (j = 0; j < m; ++j)
            {
                basis_ntl[i][j] = NTL::to_ZZ(basis.coeff(i, j));
            }
        }
        NTL::ComputeGS(basis_ntl, mu_ntl, B_ntl);
        for (i = 0; i < n; ++i)
        {
            for (j = 0; j < i; ++j)
            {
                err_mat.coeffRef(i, j) = NTL::to_double(mu_ntl[i][j]) - mu.coeff(i, j);
            }
        }
        fprintf(err, "%Le\n", err_mat.squaredNorm());
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
    if (output_err)
    {
        fclose(err);
    }
}

void deepLLL(const double delta, const long gamma, const long end, const long h, const long n)
{
    bool flag;
    long i, j, k, q;
    long double C;

    for (k = h; k < end;)
    {
        for (j = k - 1; j >= 0; --j)
        {
            if (fabsl(mu.coeff(k, j)) > 0.5)
            {
                q = std::lroundl(mu.coeff(k, j));
                basis.row(k) -= q * basis.row(j);

                for (i = 0; i <= j; ++i)
                {
                    mu.coeffRef(k, i) -= mu.coeff(j, i) * static_cast<long double>(q);
                }
            }
        }

        flag = false;
        C = static_cast<long double>(basis.row(k).squaredNorm());
        for (i = 0; i < k;)
        {
            if (C < delta * B.coeff(i))
            {
                if ((i <= gamma) and (k - i + 1 <= gamma))
                {
                    flag = true;
                }
            }

            if (not flag)
            {
                C -= mu.coeff(k, i) * mu.coeff(k, i) * B.coeff(i);
                ++i;
            }
            else
            {
                deepInsertion(i, k);
                updateDeepInsertionGSO(i, k, n);

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
    }
}
