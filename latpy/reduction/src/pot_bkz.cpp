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

void potBKZ(
    long **basis_ptr,
    const double delta,
    const long beta,
    const long max_loops,
    const bool output_sl,
    const bool output_rhf,
    const bool output_err,
    const long n,
    const long m)
{
    long i, j, k, l, z, d;
    long n_tour = 0;
    VectorXli v, w;
    VectorXld logB(n);
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

    for (z = j = 0; z < n - 1;)
    {
        if (j == n - 2)
        {
            j = 0;
            if (max_loops > -1)
            {
                ++n_tour;
                if (n_tour >= max_loops)
                {
                    break;
                }
            }
        }
        ++j;

        if (j + beta < n)
        {
            k = j + beta - 1;
        }
        else
        {
            k = n - 1;
        }

        d = k - j + 1;

        if (potENUM(v, delta, j, k + 1))
        {
            z = 0;

            w = v * basis.block(j, 0, d, m);

            basis_ntl.SetDims(n + 1, m);
            for (l = 0; l < m; ++l)
            {
                for (i = 0; i < j; ++i)
                {
                    basis_ntl[i][l] = NTL::to_ZZ(basis.coeffRef(i, l));
                }
                basis_ntl[j][l] = NTL::to_ZZ(w[l]);
                for (i = j + 1; i < n + 1; ++i)
                {
                    basis_ntl[i][l] = NTL::to_ZZ(basis.coeffRef(i - 1, l));
                }
            }
            NTL::LLL_FP(basis_ntl, 0.99);
            for (i = 0; i < n; ++i)
            {
                for (l = 0; l < m; ++l)
                {
                    basis.coeffRef(i, l) = NTL::to_long(basis_ntl[i + 1][l]);
                }
            }

            computeGSO();
            potLLL(delta, n);
        }
        else
        {
            ++z;
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
