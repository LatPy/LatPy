#include "reduction.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include "core.h"

void dualDeepInsertion(const long k, const long l)
{
    const VectorXli t = basis.row(k);
    for (long j = k; j < l; ++j)
    {
        basis.row(j) = basis.row(j + 1);
    }
    basis.row(l) = t;
}
