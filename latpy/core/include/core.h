#ifndef CORE_H
#define CORE_H

#include <iostream>

#include <eigen3/Eigen/Dense>

#include <NTL/ZZ.h>
#include <NTL/vec_ZZ.h>
#include <NTL/mat_ZZ.h>

typedef Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXli;        // long-type matrix
typedef Eigen::Matrix<long, 1, Eigen::Dynamic> VectorXli;                                      // long-type vector
typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXld; // long double-type matrix
typedef Eigen::Matrix<long double, 1, Eigen::Dynamic> VectorXld;                               // long double-type vector

extern VectorXld B;
extern VectorXld s;
extern MatrixXli basis;
extern MatrixXld mu;
extern MatrixXld R;
extern NTL::ZZ vol;
extern MatrixXld hmu;
extern VectorXld hB;

extern NTL::vec_ZZ B_num, B_den;
extern NTL::mat_ZZ mu_num, mu_den;

/**
 * @brief computes GSA-slope
 *
 * @param n rank of lattice
 * @return long double GSA-slope
 */
long double sl(const long n);

/**
 * @brief computes root of Hermite-factor
 *
 * @param n rank of lattice
 * @return long double root of Hermite-factor
 */
long double rhf(const long n);

/**
 * @brief Compute GSO-informations of lattice basis
 *
 */
void computeGSO();

/**
 * @brief Compute GSO-informations of lattice basis without fpa
 *
 */
void fracGSO(NTL::mat_ZZ b);

/**
 * @brief Compute R-factor of lattice basis
 *
 */
void computeR();

/**
 * @brief
 *
 * @param k
 * @param Q
 * @param B_star
 * @param s
 */
void blockQR(const long k, const bool is_shifted, MatrixXld &Q, MatrixXld &B_star);

/**
 * @brief compiutes CF-information of the lattice
 *
 * @param n rank of lattice
 *
 */
void computeCF(const long n, const long m);

bool isSeysen(const MatrixXld R);

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
    char *volume(long **basis_ptr, const long n, const long m);

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
     * @param basis_ptr the input lattice basis matrix
     * @param n rank of the lattice
     * @param m null of the lattice
     * @return long double
     */
    char *pot(long **basis_ptr, const long n, const long m);

    /**
     * @brief compute Hermite factor of the lattice basis
     *
     * @param basis_ptr lattice basis matrix
     * @param n rank of the lattice
     * @param m null of the lattice
     * @return long double Hermite factor of the lattice basis
     */
    long double hf(long **basis_ptr, const long n, const long m);

    /**
     * @brief compute the root of Hermite-factor of the lattice basis
     *
     * @param basis_ptr lattice basis matrix
     * @param n rank of the lattice
     * @param m null of the lattice
     * @return long double root of Hermite-factor of the lattice basis
     */
    long double rhf(long **basis_ptr, const long n, const long m);

    /**
     * @brief compute orthogonality defect of the lattice basis
     *
     * @param basis_ptr lattice basis matrix
     * @param n rank of the lattice
     * @param m null of the lattice
     * @return long double orthogonality defect of the lattice basis
     */
    long double od(long **basis_ptr, const long n, const long m);

    /**
     * @brief Checks if lattice basis is size-reduced or not
     *
     * @param basis_ptr lattice basis matrix
     * @param n rank of lattice
     * @param m null of lattice
     * @return true if lattice basis is size-reduced
     * @return false if lattice basis is not size-reduced
     */
    bool isSize(long **basis_ptr, const long n, const long m);

    /**
     * @brief Checks if lattice basis is seysen-reduced or not
     *
     * @param basis_ptr lattice basis matrix
     * @param n rank of lattice
     * @param m null of lattice
     * @return true if lattice basis is seysen-reduced
     * @return false if lattice basis is not seysen-reduced
     */
    bool isSeysen(long **basis_ptr, const long n, const long m);

    /**
     * @brief Checks if lattice basis is weakly-LLL-reduced, that is, satisfies Lovasz condition or not
     *
     * @param basis_ptr lattice basis matrix
     * @param delta reduction parameter
     * @param n rank of lattice
     * @param m null of lattice
     * @return true if lattice basis is weakly-LLL-reduced
     * @return false if lattice basis is not weakly-LLL-reduced
     */
    bool isWeaklyLLL(long **basis_ptr, const double delta, const long n, const long m);

    /**
     * @brief Checks if lattice basis is weakly-DeepLLL-reduced, that is, satisfies deep-exchange condition or not
     *
     * @param basis_ptr lattice basis matrix
     * @param delta reduction parameter
     * @param n rank of lattice
     * @param m null of lattice
     * @return true if lattice basis is weakly-Deep-LLL-reduced
     * @return false if lattice basis is not weakly-Deep-LLL-reduced
     */
    bool isWeaklyDeepLLL(long **basis_ptr, const double delta, const long n, const long m);
}

#endif // !LAT_PY_H
