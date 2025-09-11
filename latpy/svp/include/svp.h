#include <iostream>

#include <eigen3/Eigen/Dense>

typedef Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXli;        // long-type matrix
typedef Eigen::Matrix<long, 1, Eigen::Dynamic> VectorXli;                                      // long-type vector
typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXld; // long double-type matrix
typedef Eigen::Matrix<long double, 1, Eigen::Dynamic> VectorXld;                               // long double-type vector

extern VectorXld eps;

void coeffPruning(const long n, const bool pruning);

bool enumSV(VectorXli &coeff, const double radius_, MatrixXld mu, VectorXld B, const bool pruning, const long start, const long end);

/**
 * @brief enumerate shortest vector on lattice with QR-factorization
 * 
 * @param coeff coefficient vector
 * @param radius_ radius for searching
 * @param pruning make use of pruning or not
 * @param start 
 * @param end 
 * @return true 
 * @return false 
 */
bool qrEnumSV(VectorXli &coeff, const double radius_, const bool pruning, const long start, const long end);

extern "C"
{
    /**
     * @brief Enumerate shortest vector on the lattice
     * 
     * @param basis_ptr lattice basis matrix
     * @param coeff coefficient vector
     * @param pruning make use of pruning or not
     * @param n rank of lattice
     * @param m null of lattice
     */
    void enumSV(long **basis_ptr, long *coeff, const bool pruning, const long n, const long m);
}
