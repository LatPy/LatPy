#include "reduction.h"

#include <iostream>

#include <eigen3/Eigen/Dense>

#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>

long double rhf(const long n)
{
    long double norm;
    long double hermite_factor;

    norm = basis.row(0).cast<long double>().norm();
    hermite_factor = norm / NTL::to_double(NTL::power(volume, 1.0 / n));
    return powl(hermite_factor, 1.0 / n);
}
