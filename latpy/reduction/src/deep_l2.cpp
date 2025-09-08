#include "reduction.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>
#include <NTL/RR.h>
#include <NTL/mat_RR.h>
#include <NTL/LLL.h>

void deepL2(
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
    long i, j, k;
    const double delta_hat = (delta + 1) * 0.5;
    const double eta_hat = (eta + 0.5) * 0.5;
    bool is_size_reduced;
    bool is_shifted = false;
    VectorXli X(n);
    s = VectorXld::Zero(n);
    MatrixXld Q = MatrixXld::Zero(n, n), B_star = MatrixXld::Zero(n, m), err_mat = MatrixXld::Zero(n, n);
    FILE *log_sl, *log_rhf, *err;
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

    B_star.row(0) = basis.row(0).cast<long double>();
    R.coeffRef(0, 0) = B_star.row(0).norm();
    Q.row(0) = B_star.row(0) / R.coeff(0, 0);
    for (k = 1; k < n;)
    {
        X.setZero();
        for (;;)
        {
            blockQR(k, is_shifted, Q, B_star);

            is_size_reduced = true;
            for (i = 0; i < k; ++i)
            {
                if (fabsl(R.coeff(k, i)) > eta_hat * fabsl(R.coeff(i, i)))
                {
                    is_size_reduced = false;
                    break;
                }
            }

            if (is_size_reduced)
            {
                break;
            }
            else
            {
                for (i = k - 1; i > -1; --i)
                {
                    X.coeffRef(i) = std::lroundl(R.coeff(k, i) / R.coeff(i, i));
                    for (j = 0; j < i; ++j)
                    {
                        R.coeffRef(k, j) -= X.coeff(i) * R.coeff(i, j);
                    }
                }

                for (i = 0; i < k; ++i)
                {
                    basis.row(k) -= X.coeff(i) * basis.row(i);
                }
            }
        }

        is_shifted = false;

        for (i = 0; i < k;)
        {
            if (delta_hat * R.coeff(i, i) * R.coeff(i, i) > s.coeff(i))
            {
                deepInsertion(i, k);

                if (i > 0)
                {
                    k = i - 1;
                }
                else
                {
                    is_shifted = true;
                    k = 0;
                }
            }
            else
            {
                ++i;
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
                err_mat.coeffRef(i, j) = NTL::to_double(mu_ntl[i][j]) - R.coeff(i, j) / R.coeff(j, j);
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
