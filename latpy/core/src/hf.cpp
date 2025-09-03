#include "core.h"

#include <NTL/ZZ.h>
#include <NTL/vec_ZZ.h>
#include <NTL/mat_ZZ.h>

extern "C" long double hf(long **basis_ptr, const long n, const long m)
{
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
    }

    if (n == m)
    {
        vol = NTL::abs(NTL::determinant(basis_ntl));
    }
    else
    {
        vol = NTL::SqrRoot(NTL::determinant(basis_ntl * NTL::transpose(basis_ntl)));
    }

    NTL::InnerProduct(sq_norm, basis_ntl[0], basis_ntl[0]);
    return sqrtl(NTL::to_double(sq_norm)) / NTL::to_double(NTL::power(vol, 1.0 / n));
}
