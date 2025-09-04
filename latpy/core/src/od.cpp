#include "core.h"

#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>

long double od(long **basis_ptr, const long n, const long m)
{
    long double p = 1.0;
    NTL::ZZ vol;
    NTL::ZZ sq_norm;
    NTL::mat_ZZ basis_ntl;
    basis_ntl.SetDims(n, m);
    for (long i = 0, j; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis_ntl[i][j] = NTL::to_ZZ(basis_ptr[i][j]);
        }

        NTL::InnerProduct(sq_norm, basis_ntl[i], basis_ntl[i]);
        p *= sqrtl(NTL::to_double(sq_norm));
    }

    if (n == m)
    {
        vol = NTL::abs(NTL::determinant(basis_ntl));
    }
    else
    {
        vol = NTL::SqrRoot(NTL::determinant(basis_ntl * NTL::transpose(basis_ntl)));
    }

    return p / NTL::to_double(vol);
}
