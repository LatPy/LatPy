#include "core.h"

#include <iostream>
#include <sstream>
#include <cmath>
#include <string>

#include <eigen3/Eigen/Dense>

#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>

extern "C" char *volume(long **basis_ptr, const long n, const long m)
{
    char *cstr;
    std::string s;
    NTL::mat_ZZ basis_ntl;
    basis_ntl.SetDims(n, m);
    for (long i = 0, j; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis_ntl[i][j] = NTL::to_ZZ(basis_ptr[i][j]);
        }
    }

    std::stringstream ss;

    if (n == m)
    {
        ss << NTL::abs(NTL::determinant(basis_ntl));
        s = ss.str();
        cstr = new char[s.size() + 1];
        std::copy(s.begin(), s.end(), cstr);
        cstr[s.size()] = '\0';
        return cstr;
    }
    else
    {
        ss << NTL::SqrRoot(NTL::determinant(basis_ntl * NTL::transpose(basis_ntl)));
        s = ss.str();
        cstr = new char[s.size() + 1];
        std::copy(s.begin(), s.end(), cstr);
        cstr[s.size()] = '\0';
        return cstr;
    }
}
