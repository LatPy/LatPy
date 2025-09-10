#include "core.h"

#include <iostream>
#include <cmath>
#include <string>
#include <sstream>

#include <eigen3/Eigen/Dense>

#include <NTL/RR.h>

extern "C" char *pot(long **basis_ptr, const long n, const long m)
{
    long i, j;
    char *cstr;
    std::string s;
    std::stringstream ss;

    basis = MatrixXli::Zero(n, m);
    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis.coeffRef(i, j) = basis_ptr[i][j];
        }
    }
    computeGSO();

    NTL::RR p = NTL::to_RR(1);
    for (i = 0; i < n; ++i)
    {
        p *= NTL::pow(NTL::to_RR((double)B.coeff(i)), NTL::to_RR(n - i));
    }
    ss << p;
    s = ss.str();
    cstr = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), cstr);
    cstr[s.size()] = '\0';
    return cstr;
}
