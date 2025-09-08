#include "reduction.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>
#include <NTL/LLL.h>

#include "svp.h"

extern "C" void BKZ(
    long **basis_ptr,
    const double delta,
    const long beta,
    const long max_loops,
    const bool pruning,
    const bool output_sl,
    const bool output_rhf,
    const bool output_err,
    const long n,
    const long m)
{
    long z, i, j, num_tour = 0, k = 0, h, d, l, p;
    long double radius;
    FILE *log_sl, *log_rhf, *err;
    VectorXli t, coeff_vec, v;
    MatrixXld err_mat = MatrixXld::Zero(n, n);
    NTL::mat_ZZ basis_ntl;
    NTL::vec_RR B_ntl;
    NTL::mat_RR mu_ntl;

    LLL(basis_ptr, delta, 0.5, false, false, false, n, m);

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

    computeGSO(basis);

    for (z = k = 0; z < n - 2;)
    {
        if (k == n - 1)
        {
            k = 0;
            if (max_loops > -1)
            {
                ++num_tour;
                if (num_tour > max_loops)
                {
                    break;
                }
            }
        }
        ++k;
        l = std::min(k + beta - 1, n);
        h = std::min(l + 1, n);
        d = l - k + 1;

        radius = delta * B.coeff(k - 1);

        if (enumSV(coeff_vec, radius, mu, B, pruning, k - 1, l))
        {
            v = coeff_vec * basis.block(k - 1, 0, d, m);

            z = 0;

            basis_ntl.SetDims(n + 1, m);
            for (j = 0; j < n + 1; ++j)
            {
                if (j < k - 1)
                {
                    for (p = 0; p < m; ++p)
                    {
                        basis_ntl[j][p] = basis.coeff(j, p);
                    }
                }
                else if (j == k - 1)
                {
                    for (p = 0; p < m; ++p)
                    {
                        basis_ntl[j][p] = v.coeff(p);
                    }
                }
                else
                {
                    for (p = 0; p < m; ++p)
                    {
                        basis_ntl[j][p] = basis.coeff(j - 1, p);
                    }
                }
            }
            NTL::LLL_FP(basis_ntl, 0.99);
            for (j = 0; j < n; ++j)
            {
                for (p = 0; p < m; ++p)
                {
                    basis.coeffRef(j, p) = NTL::to_long(basis_ntl[j + 1][p]);
                }
            }

            computeGSO(basis);
        }
        else
        {
            LLL(delta, h, n);

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
