#include "reduction.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>
#include <NTL/RR.h>
#include <NTL/mat_RR.h>
#include <NTL/LLL.h>

void L2(
    long **basis_ptr,
    const double delta,
    const double eta,
    const bool output_sl,
    const bool output_rhf,
    const bool output_err,
    const long n,
    const long m)
{
    long i, j, k, k_, x, h;
    const double delta_hat = (delta + 1) * 0.5;
    const double eta_bar = (eta + 0.5) * 0.5;
    long double max;
    FILE *log_sl, *log_rhf, *err;
    MatrixXld err_mat = MatrixXld::Zero(n, n);
    NTL::vec_RR B_ntl;
    NTL::mat_RR mu_ntl;
    NTL::mat_ZZ basis_ntl;

    basis = MatrixXli::Zero(n, m);
    basis_ntl.SetDims(n, m);

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

    computeCF(n, m);

    for (k = 1; k < n;)
    {
        R.coeffRef(0, 0) = basis.row(0).squaredNorm();
        mu.coeffRef(0, 0) = 1.0;
        for (;;)
        {
            for (j = 0; j <= k; ++j)
            {
                R.coeffRef(k, j) = basis.row(k).dot(basis.row(j));
                for (h = 0; h < j; ++h)
                {
                    R.coeffRef(k, j) -= R.coeff(k, h) * mu.coeff(j, h);
                }
                mu.coeffRef(k, j) = R.coeff(k, j) / R.coeff(j, j);
            }

            s.coeffRef(0) = basis.row(k).squaredNorm();
            for (j = 1; j <= k; ++j)
            {
                s.coeffRef(j) = s.coeff(j - 1) - mu.coeff(k, j - 1) * R.coeff(k, j - 1);
            }
            R.coeffRef(k, k) = s.coeff(k);

            max = -1;
            for (i = 0; i < k; ++i)
            {
                if (fabsl(mu.coeff(k, i)) > max)
                {
                    max = fabsl(mu.coeff(k, i));
                }
            }

            if (max > eta_bar)
            {
                for (i = k - 1; i >= 0; --i)
                {
                    x = static_cast<long>(round(mu.coeff(k, i)));
                    basis.row(k) -= x * basis.row(i);
                    for (j = 0; j <= i; ++j)
                    {
                        mu.coeffRef(k, j) -= static_cast<long double>(x) * mu.coeff(i, j);
                    }
                }
            }
            else
            {
                break;
            }
        }

        if (output_sl)
        {
            B = R.diagonal();
            fprintf(log_sl, "%Lf\n", sl(n));
        }
        if (output_rhf)
        {
            fprintf(log_rhf, "%Lf\n", rhf(n));
        }

        k_ = k;
        while ((k > 0) && (delta_hat * R.coeff(k - 1, k - 1) >= s.coeff(k - 1)))
        {
            basis.row(k - 1).swap(basis.row(k));
            --k;
        }

        for (i = 0; i < k; ++i)
        {
            mu.coeffRef(k, i) = mu.coeff(k_, i);
            R.coeffRef(k, i) = R.coeff(k_, i);
            R.coeffRef(k, k) = s.coeff(k);
        }

        ++k;
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
