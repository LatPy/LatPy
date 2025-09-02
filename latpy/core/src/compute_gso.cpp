#include "core.h"

#include <iostream>
#include <cstdlib>

void computeGSO(MatrixXli basis_, MatrixXld &mu_, VectorXld &B_)
{
    const long n = basis_.rows(), m = basis_.cols();
    long i, j;
    MatrixXld gso_basis(n, m);

    mu_ = MatrixXld::Identity(n, n);
    B_ = VectorXld::Zero(n);

    for (i = 0; i < n; ++i)
    {
        mu_.coeffRef(i, i) = 1;
        gso_basis.row(i) = basis.row(i).cast<long double>();

        for (j = 0; j < i; ++j)
        {
            mu_.coeffRef(i, j) = basis.row(i).cast<long double>().dot(gso_basis.row(j)) / B_.coeff(j);
            gso_basis.row(i) -= mu_.coeff(i, j) * gso_basis.row(j);
        }
        B_.coeffRef(i) = gso_basis.row(i).squaredNorm();
    }
}


extern "C" void computeGSO(long **basis_ptr, double **mu_ptr, double *B_ptr, const long n, const long m)
{
    long i, j;
    MatrixXld gso_basis(n, m);

    mu = MatrixXld::Identity(n, n);
    B = VectorXld::Zero(n);
    basis = MatrixXli::Zero(n, m);
    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis.coeffRef(i, j) = basis_ptr[i][j];
        }
    }

    for (i = 0; i < n; ++i)
    {
        mu.coeffRef(i, i) = 1;
        gso_basis.row(i) = basis.row(i).cast<long double>();

        for (j = 0; j < i; ++j)
        {
            mu.coeffRef(i, j) = basis.row(i).cast<long double>().dot(gso_basis.row(j)) / B.coeff(j);
            gso_basis.row(i) -= mu.coeff(i, j) * gso_basis.row(j);
        }
        B.coeffRef(i) = gso_basis.row(i).squaredNorm();
    }

    for (i = 0; i < n; ++i)
    {
        B_ptr[i] = B.coeff(i);
        for (j = 0; j < n; ++j)
        {
            mu_ptr[i][j] = mu.coeff(i, j);
        }
    }
}
