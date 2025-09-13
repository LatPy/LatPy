#include <iostream>
#include <vector>

#include <eigen3/Eigen/Dense>

typedef Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXli;        // long-type matrix
typedef Eigen::Matrix<long, 1, Eigen::Dynamic> VectorXli;                                      // long-type vector
typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXld; // long double-type matrix
typedef Eigen::Matrix<long double, 1, Eigen::Dynamic> VectorXld;                               // long double-type vector

/**
 * @brief Applies deep-insetion \sigma_{i, k} to lattice basis
 *
 * @param i index
 * @param k index
 */
void deepInsertion(const long i, const long k);

/**
 * @brief Applies dual-deep-insertion \widehat{\sigma}_{k, l} to lattice basis
 *
 * @param k index
 * @param l index
 */
void dualDeepInsertion(const long k, const long l);

/**
 * @brief Updates GSO-information with swapping of the lattice basis vectors \bm{b}_{k-1} and \bm{b}_{k}
 *
 * @param k index
 * @param n rank of lattice
 */
void updateSwapGSO(const long k, const long n);

/**
 * @brief Update R-factor with swapping of the lattice basis vectors \bm{b}_{k-1} and \bm{b}_{k}
 *
 * @param k
 * @param n
 */
void updateSwapR(const long k, const long n);

/**
 * @brief Updates GSO-informations with applying deep-insetion \sigma_{i, k} to lattice basis
 *
 * @param i index
 * @param k index
 * @param n rank of lattice
 */
void updateDeepInsertionGSO(const long i, const long k, const long n);

void updateDualDeepInsertionGSO(const long k, const long l, const VectorXld hD, const long n);

/**
 * @brief Updates R-factor with applying deep-insetion \sigma_{i, k} to lattice basis
 *
 * @param i index
 * @param k index
 * @param n rank of lattice
 */
void updateDeepInsertionR(const long i, const long k, const long n);

/**
 * @brief enumerate delta-anomolous vector
 *
 * @param coeff coefficient vector
 * @param delta reduction paramerter
 * @param start star index
 * @param end end index
 * @return true
 * @return false
 */
bool potENUM(VectorXli &coeff, const double delta, const long start, const long end);

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

/**
 * @brief Applies DeepLLL-reduction
 *
 * @param delta reduction parameter
 * @param gamma depth
 * @param end end index
 * @param h start index
 * @param n rank of lattice
 */
void deepLLL(const double delta, const long gamma, const long end, const long h, const long n);

/**
 * @brief Allplies LLL=reduction with QR-factorization
 *
 * @param delta reduction parameter
 * @param end end index
 * @param n rank of lattice
 */
void qrLLL(const double delta, const long end, const long n);

/**
 * @brief Applies DeepLLL-reduction with QR-factorization
 *
 * @param delta reduction parameter
 * @param gamma depth
 * @param end end index
 * @param h start index
 * @param n rank of lattice
 */
void qrDeepLLL(const double delta, const long gamma, const long end, const long h, const long n);

/**
 * @brief Applies PotLLL-reduction
 *
 * @param delta reduction parameter
 * @param n rank of lattice
 */
void potLLL(const double delta, const long n);

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
        const bool output_err,
        const long n,
        const long m);

    /**
     * @brief Applies LLL-reduction to lattice basis with QR factorization
     *
     * @param basis_ptr lattice basis matrix
     * @param delta reduction parameter for Lovasz condition
     * @param eta reduction parameter for size-reduction condition
     * @param output_sl output GSA-slope ot not
     * @param output_rhf outpur root of Hermite-factor or not
     * @param n rank of lattice
     * @param m null of lattice
     */
    void qrLLL(
        long **basis_ptr,
        const double delta,
        const double eta,
        const bool output_sl,
        const bool output_rhf,
        const bool output_err,
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
        const bool output_err,
        const long n,
        const long m);

    /**
     * @brief Applies deepLLL-reduction to lattice basis with QR factorization
     *
     * @param basis_ptr lattice basis matrix
     * @param delta reduction parameter to Lovasz condition
     * @param eta reduction parameter to size-reduction condition
     * @param gamma depth of deep-reduction
     * @param output_sl output GSA-slope or not
     * @param output_rhf output root of Hermite-factor or not
     * @param n rank of lattice
     * @param m null of lattice
     */
    void qrDeepLLL(
        long **basis_ptr,
        const double delta,
        const double eta,
        const long gamma,
        const bool output_sl,
        const bool output_rhf,
        const bool output_err,
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
        const bool output_err,
        const long n,
        const long m);

    /**
     * @brief Applies DeepLLL-reduction to lattice basis with L2-like algorithm
     *
     * @param basis_ptr lattice basis matrix
     * @param delta reduction parameter for deep-exchange condition
     * @param eta reduction parameter for size-reduction condition
     * @param gamma depth
     * @param output_sl output GSA-slope or not
     * @param output_rhf output root of Hermite-factor or not
     * @param n rank of lattice
     * @param m null of lattice
     */
    void deepL2(
        long **basis_ptr,
        const double delta,
        const double eta,
        const long gamma,
        const bool output_sl,
        const bool output_rhf,
        const bool output_err,
        const long n,
        const long m);

    void potLLL(
        long **basis_ptr,
        const double delta,
        const double eta,
        const bool output_sl,
        const bool output_rhf,
        const bool output_err,
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
        const bool output_err,
        const long n,
        const long m);

    /**
     * @brief Applies DeepBKZ-reduction to lattice basis
     *
     * @param basis_ptr lattice basis matrix
     * @param delta reduction parameter to deep-exchange condition
     * @param beta block-size
     * @param gamma depth for DeepLLL
     * @param max_loops maximam loops times of tour
     * @param pruning if make use of pruning or not
     * @param output_sl output GSA-slope or not
     * @param output_rhf output root of the Hermite-factor
     * @param output_err output error or not
     * @param n rank of lattice
     * @param m null of lattice
     */
    void deepBKZ(
        long **basis_ptr,
        const double delta,
        const long beta,
        const long gamma,
        const long max_loops,
        const bool pruning,
        const bool output_sl,
        const bool output_rhf,
        const bool output_err,
        const long n,
        const long m);

    /**
     * @brief Applies BKZ-reduction with QR factorization
     *
     * @param basis_ptr lattice basis matrix
     * @param delta reduction parameter
     * @param beta block size
     * @param max_loops maximam loops times
     * @param pruning if make use of pruning or not
     * @param output_sl putput GSA-slope or not
     * @param output_rhf output root of Hermite-factor or not
     * @param output_err output err or not
     * @param n rank of lattice
     * @param m null of lattice
     */
    void qrBKZ(
        long **basis_ptr,
        const double delta,
        const long beta,
        const long max_loops,
        const bool pruning,
        const bool output_sl,
        const bool output_rhf,
        const bool output_err,
        const long n,
        const long m);

    /**
     * @brief Applies DeepBKZ to lattice basis with QR-factorization
     *
     * @param basis_ptr lattice basis matrix
     * @param delta reduction parameter
     * @param beta blocksize
     * @param gamma depth
     * @param max_loops maximam loops times
     * @param pruning make use of pruning or not
     * @param output_sl output root of Hermite-factor
     * @param output_rhf output root of Hermite-factor
     * @param output_err output error or not
     * @param n rank of lattice
     * @param m null of latticeF
     */
    void qrDeepBKZ(
        long **basis_ptr,
        const double delta,
        const long beta,
        const long gamma,
        const long max_loops,
        const bool pruning,
        const bool output_sl,
        const bool output_rhf,
        const bool output_err,
        const long n,
        const long m);

    /**
     * @brief Applies PotBKZ to lattice basis
     *
     * @param basis_ptr lattice basis matrix
     * @param delta reduction parameter
     * @param beta blocksize
     * @param max_loops maximam loop-times
     * @param output_sl output GSA-slope or not
     * @param output_rhf output root of Hermite-factor or not
     * @param output_err outpur error or not
     * @param n rank of lattice
     * @param m null of lattice
     */
    void potBKZ(
        long **basis_ptr,
        const double delta,
        const long beta,
        const long max_loops,
        const bool output_sl,
        const bool output_rhf,
        const bool output_err,
        const long n,
        const long m);

    /**
     * @brief
     *
     * @param basis_ptr
     * @param delta
     * @param output_sl
     * @param output_rhf
     * @param output_err
     * @param n
     * @param m
     */
    void dualPotLLL(
        long **basis_ptr,
        const double delta,
        const bool output_sl,
        const bool output_rhf,
        const bool output_err,
        const long n,
        const long m);
}
