#include "reduction.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>
#include <NTL/RR.h>
#include <NTL/vec_RR.h>
#include <NTL/mat_RR.h>
#include <NTL/LLL.h>

#include "core.h"

void dualPotLLL(
    long **basis_ptr,
    const double delta,
    const bool output_sl,
    const bool output_rhf,
    const bool output_err,
    const long n,
    const long m)
{
    long i, j, k, l, h, q;
    long double P, P_min, S, D;
    VectorXld hD(n);
    FILE *log_sl, *log_rhf, *err;
    MatrixXld err_mat = MatrixXld::Zero(n, n);
    NTL::mat_ZZ basis_ntl;
    NTL::vec_RR B_ntl;
    NTL::mat_RR mu_ntl;

    basis_ntl.SetDims(n, m);
    basis = MatrixXli::Zero(n, m);
    hmu = MatrixXld::Zero(n, n);
    hB = VectorXld::Zero(n);

    LLL(basis_ptr, delta, 0.5, false, false, false, n, m);

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

    for (k = n - 1; k > -1;)
    {
        hmu.coeffRef(k, k) = 1.0L;

        for (j = k + 1; j < n; ++j)
        {
            hmu.coeffRef(k, j) = 0.0L;
            for (i = k; i < j; ++i)
            {
                hmu.coeffRef(k, j) -= mu.coeff(j, i) * hmu.coeff(k, i);
            }

            if (fabsl(hmu.coeff(k, j)) > 0.5)
            {
                q = std::lroundl(hmu.coeff(k, j));
                basis.row(j) += q * basis.row(k);
                hmu.row(k).segment(j, n - j) -= q * hmu.row(j).segment(j, n - j);
                mu.row(j).head(k + 1) += q * mu.row(k).head(k + 1);
            }
        }

        P = P_min = 1.0L;
        l = n - 1L;
        for (j = k + 1; j < n; ++j)
        {
            S = 0.0L;
            for (j = k + 1; j < n; ++j)
            {
                S += hmu.coeff(k, i) * hmu.coeff(k, i) / B.coeff(i);
            }
            P *= B.coeff(j);
            P *= S;

            if (P < P_min)
            {
                l = j;
                P_min = P;
            }
        }

        if (delta > P_min)
        {
            D = 1.0L / B.coeff(k);
            hD.setZero();

            hD.coeffRef(k) = D;
            for (h = k + 1; h < n; ++h)
            {
                D += hmu.coeff(k, h) * hmu.coeff(k, h) / B.coeff(h);
                hD.coeffRef(h) = D;
            }

            dualDeepInsertion(k, l);
            updateDualDeepInsertionGSO(k, l, hD, n);

            k = l;
        }
        else
        {
            --k;
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
