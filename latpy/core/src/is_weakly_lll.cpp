#include "core.h"

extern "C" bool isWeaklyLLL(long **basis_ptr, const double delta, const long n, const long m)
{
    long i, j, k;
    basis = MatrixXli::Zero(n, m);
    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis.coeffRef(i, j) = basis_ptr[i][j];
        }
    }
    computeGSO(basis, mu, B);

    for (k = 1; k < n; ++k)
    {
        if (B.coeff(k) < (delta - mu.coeff(k, k - 1) * mu.coeff(k, k - 1)) * B.coeff(k - 1))
        {
            return false;
        }
    }

    return true;
}
