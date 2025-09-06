#include <iostream>

#include <eigen3/Eigen/Dense>

#include <NTL/ZZ.h>

typedef Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXli;        // long-type matrix
typedef Eigen::Matrix<long, 1, Eigen::Dynamic> VectorXli;                                      // long-type vector
typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXld; // long double-type matrix
typedef Eigen::Matrix<long double, 1, Eigen::Dynamic> VectorXld;                               // long double-type vector

extern VectorXld B;
extern VectorXld s;
extern MatrixXli basis;
extern MatrixXld mu;
extern MatrixXld R;
extern NTL::ZZ volume;

/**
 * @brief Compute GSO-informations of the input lattice
 *
 * @param basis_ lattice basis
 */
void computeGSO(MatrixXli basis_);

/**
 * @brief Compute R-factor of the input lattice
 *
 * @param basis_ lattice basis
 */
void computeR(MatrixXli basis_);

/**
 * @brief compiutes CF-information of the lattice
 *
 * @param n rank of lattice
 *
 */
void computeCF(const long n, const long m);

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
 * @brief Applies deep-insetion \sigma_{i, k} to lattice basis
 *
 * @param i index
 * @param k index
 */
void deepInsertion(const long i, const long k);

/**
 * @brief Updates GSO-information with swapping of the lattice basis vectors \bm{b}_{k-1} and \bm{b}_{k}
 *
 * @param k index
 * @param n rank of lattice
 */
void updateSwapGSO(const long k, const long n);

/**
 * @brief Updates GSO-informations with applying deep-insetion \sigma_{i, k} to lattice basis
 *
 * @param i index
 * @param k index
 * @param n rank of lattice
 */
void updateDeepInsertionGSO(const long i, const long k, const long n);

/**
 * @brief
 *
 * @param R
 * @param n
 * @param m
 */
MatrixXli seysenUnimodular(const MatrixXld R_, const long n, const long m);

/**
 * @brief Applies LLL-reduction
 *
 * @param delta reduction parameter
 * @param end end index
 */
void LLL(const double delta, const long end, const long n);

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
     * @brief Applies size-reduction to input lattice basis
     *
     * @param basis_ptr lattice basis matrix
     * @param eta reduction parameter
     * @param n rank of lattice
     * @param m null of lattice
     */
    void size(long **basis_ptr, const double eta, const long n, const long m);

    /**
     * @brief Applies Seysen-reduction to inpute lattice basis
     *
     * @param basis_ptr lattice basisi matrix
     * @param n rank of lattice
     * @param m null of lattice
     */
    void seysen(long **basis_ptr, const long n, const long m);

    /**
     * @brief Applies LLL-reduction to lattice basis
     *
     * @param basis_ptr lattice basis matrix
     * @param delta reduction parameter to Lovasz condition
     * @param eta reduction parameter to size-reduction condition
     * @param output_sl output GSA-slope or not
     * @param output_rhf output root of Hermite-factor or not
     * @param n rank of lattice
     * @param m null of lattice
     */
    void LLL(
        long **basis_ptr,
        const double delta,
        const double eta,
        const bool output_sl,
        const bool output_rhf,
        const long n,
        const long m);

    /**
     * @brief Applies DeepLLL-reduction to lattice basis
     *
     * @param basis_ptr lattice basis matrix
     * @param delta reductino parameter to Lovasz condition
     * @param eta reduction parameter to size reduction condition
     * @param output_sl output GSA-slope or not
     * @param output_rhf output root of Hermite-factor or not
     * @param n rank of lattice
     * @param m null of lattice
     */
    void deepLLL(
        long **basis_ptr,
        const double delta,
        const double eta,
        const long gamma,
        const bool output_sl,
        const bool output_rhf,
        const long n,
        const long m);

    /**
     * @brief Applies LLL-reduction to lattice basis with L2-algorithm
     *
     * @param basis_ptr lattice basis matrix
     * @param delta reduction parameter to Lovasz condition
     * @param eta reduction parameter to size-reduction condition
     * @param output_sl output GSA-slope or not
     * @param output_rhf output root of Hermite-factor or not
     * @param n rank of lattice
     * @param m null of lattice
     */
    void L2(
        long **basis_ptr,
        const double delta,
        const double eta,
        const bool output_sl,
        const bool output_rhf,
        const long n,
        const long m);

    /**
     * @brief Applies BKZ-reduction to lattice basis
     *
     * @param basis_ptr lattice basis matrix
     * @param delta reduction parameter to Lovasz condition
     * @param beta block size
     * @param pruning if make use of pruning or not
     * @param output_sl output GSA-slope or not
     * @param output_rhf output root of Hermite-factor or not
     * @param n rank of lattice
     * @param m null of lattice
     */
    void BKZ(
        long **basis_ptr,
        const double delta,
        const long beta,
        const long max_loops,
        const bool pruning,
        const bool output_sl,
        const bool output_rhf,
        const long n,
        const long m);
}
