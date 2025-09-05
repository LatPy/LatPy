#include "reduction.h"

#include <iostream>

#include <eigen3/Eigen/Dense>

void deepInsertion(const long i, const long k)
{
    const VectorXli t = basis.row(k);
    for (long j = k; j > i; --j)
    {
        basis.row(j) = basis.row(j - 1);
    }
    basis.row(i) = t;
}
