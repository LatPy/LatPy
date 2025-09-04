#include <iostream>

#include <eigen3/Eigen/Dense>

typedef Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXli;        // long-type matrix
typedef Eigen::Matrix<long, 1, Eigen::Dynamic> VectorXli;                                      // long-type vector
typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXld; // long double-type matrix
typedef Eigen::Matrix<long double, 1, Eigen::Dynamic> VectorXld;                               // long double-type vector

extern VectorXld B;
extern MatrixXli basis;
extern MatrixXld mu;

/**
 * @brief Compute GSO-informations of the input lattice
 *
 * @param basis_ lattice basis
 */
void computeGSO(MatrixXli basis_);

extern "C"
{
    /**
     * @brief Applies Lagrange reduction to input lattice basis
     * 
     * @param basis_ptr lattice basis matrix
     * @param n rank of lattice
     * @param m null of lattice
     */
    void lagrange(long **basis_ptr, const long n, const long m);

    /**
     * @brief 
     * 
     * @param basis_ptr 
     * @param eta 
     * @param n 
     * @param m 
     */
    void size(long **basis_ptr, const double eta, const long n, const long m);
}
