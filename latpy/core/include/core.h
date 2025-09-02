#ifndef CORE_H
#define CORE_H

#include <iostream>

#include <eigen3/Eigen/Dense>

typedef Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXli;        // long-type matrix
typedef Eigen::Matrix<long, 1, Eigen::Dynamic> VectorXli;                                      // long-type vector
typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXld; // long double-type matrix
typedef Eigen::Matrix<long double, 1, Eigen::Dynamic> VectorXld;                               // long double-type vector

extern VectorXld B;
extern MatrixXli basis;
extern MatrixXld mu;

void computeGSO(MatrixXli basis_, MatrixXld &mu_, VectorXld &B_);

extern "C"
{
    /**
     * @brief Compute GSO-information
     * 
     * @param basis_ptr lattice basis matrix
     * @param mu_ptr 
     * @param B_ptr 
     * @param n 
     * @param m 
     */
    void computeGSO(long **basis_ptr, double **mu_ptr, double *B_ptr, const long n, const long m);

    /**
     * @brief Compute the volume of lattice
     * 
     * @param basis_ptr the inpute lattice basis matrix
     * @param n rank of the lattice
     * @param m null of the lattice
     * @return long the volume of the lattice basis
     */
    long volume(long **basis_ptr, const long n, const long m);

    /**
     * @brief Compute the GSA-slope of lattice basis
     * 
     * @param basis the input lattice basis matrix
     * @param n rank of the lattice
     * @param m null of the lattice
     * @return long double the GSA-slope of the lattice basis
     */
    long double sl(long **basis_ptr, const long n, const long m);

    /**
     * @brief compute the potential of lattice basis
     * 
     * @param basis_ptr 
     * @param n 
     * @param m 
     * @return long double 
     */
    long double pot(long **basis_ptr, const long n, const long m);
}

#endif // !LAT_PY_H
