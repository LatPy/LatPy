#include "reduction.h"

#include <iostream>
#include <vector>

long double diagR(const long i, const long j, std::vector<std::vector<bool>> visited)
{
    if (visited[i][j])
    {
        return R.coeff(j, j);
    }
    else
    {
        visited[i][j] = true;
        // Case of using diagR, B has pre-updated diag(R).
        return B.coeff(j);
    }
}
