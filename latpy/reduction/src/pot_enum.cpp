#include "reduction.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

#define LOG099 -0.010050335853501441183548857558547706085515007674629873378L

bool potENUM(VectorXli &coeff, const double delta, const long start, const long end)
{
    const long n = end - start;
    bool has_solution = false;
    long last_nonzero = 0, k;
    long i, j;
    long r[n + 1];
    long double R = logl(B.coeff(start)), P = 0, tmp;
    long width[n];
    long double center[n], D[n + 1], sigma[n + 1][n];
    VectorXli v(n);
    v.setZero();
    v.coeffRef(0) = 1;

    for (i = 0; i < n; ++i)
    {
        r[i] = i;
        width[i] = 0;
        center[i] = 0;
        D[i] = 0;
        for (j = 0; j <= n; ++j)
        {
            sigma[j][i] = 0;
        }
    }
    r[n] = n;
    D[n] = 0;

    for (k = 0;;)
    {
        tmp = static_cast<long double>(v.coeff(k)) - center[k];
        tmp *= tmp;
        D[k] = D[k + 1] + tmp * B.coeff(k + start);

        if ((k + 1) * logl(D[k]) + P < (k + 1) * LOG099 + R)
        {
            if (k == 0)
            {
                has_solution = true;
                coeff = v;
                // return true;
                R = LOG099 + fminl(LOG099 + logl(D[0]), R);
            }
            else
            {
                P += logl(D[k]);
                --k;
                if (r[k] < r[k + 1])
                {
                    r[k] = r[k + 1];
                }

                for (i = r[k]; i > k; --i)
                {
                    sigma[i][k] = sigma[i + 1][k] + mu.coeff(i + start, k + start) * v.coeff(i);
                }
                center[k] = -sigma[k + 1][k];
                v.coeffRef(k) = std::lroundl(center[k]);
                width[k] = 1;
            }
        }
        else
        {
            ++k;
            if (k == n)
            {
                return has_solution;
            }
            else
            {
                r[k - 1] = k;
                if (k >= last_nonzero)
                {
                    last_nonzero = k;
                    ++v.coeffRef(k);
                    P = 0;
                    R = B.segment(start, last_nonzero + 1).array().log().sum();
                }
                else
                {
                    if (v.coeff(k) > center[k])
                    {
                        v.coeffRef(k) -= width[k];
                    }
                    else
                    {
                        v.coeffRef(k) += width[k];
                    }
                    ++width[k];
                    P -= logl(D[k]);
                }
            }
        }
    }
}
