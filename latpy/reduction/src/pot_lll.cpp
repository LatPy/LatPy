#include "reduction.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>
#include <NTL/RR.h>
#include <NTL/mat_RR.h>
#include <NTL/LLL.h>

#include "core.h"

extern "C" void potLLL(
    long **basis_ptr,
    const double delta,
    const double eta,
    const bool output_sl,
    const bool output_rhf,
    const bool output_err,
    const long n,
    const long m)
{
    long i, j, k, q, l;
    long double P, P_min, S;
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
        vol = NTL::abs(NTL::determinant(basis_ntl));
    }
    else
    {
        vol = NTL::SqrRoot(NTL::determinant(basis_ntl * NTL::transpose(basis_ntl)));
    }

    computeGSO();

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

    for (l = 0; l < n;)
    {
        for (j = l - 1; j > -1; --j)
        {
            if (fabsl(mu.coeff(l, j)) > 0.5)
            {
                q = std::lroundl(mu.coeff(l, j));
                basis.row(l) -= q * basis.row(j);
                mu.row(l).head(j + 1) -= static_cast<long double>(q) * mu.row(j).head(j + 1);
            }
        }

        P = P_min = 1.0;
        k = 0;
        for (j = l - 1; j >= 0; --j)
        {
            S = (mu.row(l).segment(j, l - j).array().square() * B.segment(j, l - j).array()).sum();
            P *= (B.coeff(l) + S) / B.coeff(j);
            if (P < P_min)
            {
                k = j;
                P_min = P;
            }
        }

        if (delta > P_min)
        {
            deepInsertion(k, l);
            updateDeepInsertionGSO(k, l, n);
            l = k;
        }
        else
        {
            ++l;
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

void potLLL(const double delta, const long n)
{
    long i, j, k, q, l;
    long double P, P_min, S;

    for (l = 0; l < n;)
    {
        for (j = l - 1; j > -1; --j)
        {
            if (fabsl(mu.coeff(l, j)) > 0.5)
            {
                q = std::lroundl(mu.coeff(l, j));
                basis.row(l) -= q * basis.row(j);
                mu.row(l).head(j + 1) -= static_cast<long double>(q) * mu.row(j).head(j + 1);
            }
        }

        P = P_min = 1.0;
        k = 0;
        for (j = l - 1; j >= 0; --j)
        {
            S = (mu.row(l).segment(j, l - j).array().square() * B.segment(j, l - j).array()).sum();
            P *= (B.coeff(l) + S) / B.coeff(j);
            if (P < P_min)
            {
                k = j;
                P_min = P;
            }
        }

        if (delta > P_min)
        {
            deepInsertion(k, l);
            updateDeepInsertionGSO(k, l, n);
            l = k;
        }
        else
        {
            ++l;
        }
    }
}
