
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <armadillo>


#include "header_file.hpp"

// Define Gaussian 
//     if (!inputFile) return 1;
    

//     double X1, Y1, Z1, alpha1;
//     int L1;
//     double X2, Y2, Z2, alpha2;
//     int L2;

//     // read input
//     if (!(inputFile >> X1 >> Y1 >> Z1 >> alpha1 >> L1))    return 1;

//     if (!(inputFile >> X2 >> Y2 >> Z2 >> alpha2 >> L2))  return 1;

//     inputFile.close();

//     // Gaussian shells
//     std::vector<Gaussian> shell_A = generate_shell(L1, arma::vec({X1, Y1, Z1}), alpha1);
//     std::vector<Gaussian> shell_B = generate_shell(L2, arma::vec({X2, Y2, Z2}), alpha2);


// // Compute and print overlap matrix
// arma::mat overlap_matrix(shell_B.size(), shell_A.size());
// for (size_t i = 0; i < shell_B.size(); ++i) {
//     for (size_t j = 0; j < shell_A.size(); ++j) {
//         double val = S_AB_Summation(shell_A[j], shell_B[i]);
//         overlap_matrix(i, j) = val;
//     }
// }




// overlap_matrix.print("Overlap Matrix:");

    
//     return 0;
// }



//all functions in a shell
std::vector<Gaussian> generate_shell(int L, arma::vec R, std::vector<double> exponent) {
    std::vector<Gaussian> shell;

    
    // Dummy values for coeffs, norms, and energy (since this shell is uncontracted)
    arma::vec empty_coeff(exponent.size(), arma::fill::zeros);
    arma::vec empty_norm(exponent.size(), arma::fill::ones);
    double dummy_energy = 0.0;

    if (L == 2) { // d shell
        shell.push_back(Gaussian(R, exponent, arma::Col<int>({2, 0, 0}), empty_coeff, empty_norm, dummy_energy));  // d_xx
        shell.push_back(Gaussian(R, exponent, arma::Col<int>({1, 1, 0}), empty_coeff, empty_norm, dummy_energy));  // d_xy
        shell.push_back(Gaussian(R, exponent, arma::Col<int>({1, 0, 1}), empty_coeff, empty_norm, dummy_energy));  // d_xz
        shell.push_back(Gaussian(R, exponent, arma::Col<int>({0, 2, 0}), empty_coeff, empty_norm, dummy_energy));  // d_yy
        shell.push_back(Gaussian(R, exponent, arma::Col<int>({0, 1, 1}), empty_coeff, empty_norm, dummy_energy));  // d_yz
        shell.push_back(Gaussian(R, exponent, arma::Col<int>({0, 0, 2}), empty_coeff, empty_norm, dummy_energy));  // d_zz
    } else if (L == 1) { // p shell
        shell.push_back(Gaussian(R, exponent, arma::Col<int>({1, 0, 0}), empty_coeff, empty_norm, dummy_energy));  // p_x
        shell.push_back(Gaussian(R, exponent, arma::Col<int>({0, 1, 0}), empty_coeff, empty_norm, dummy_energy));  // p_y
        shell.push_back(Gaussian(R, exponent, arma::Col<int>({0, 0, 1}), empty_coeff, empty_norm, dummy_energy));  // p_z
    } else if (L == 0) { // s shell
        shell.push_back(Gaussian(R, exponent, arma::Col<int>({0, 0, 0}), empty_coeff, empty_norm, dummy_energy));  // s
    }

    return shell;
}

    



//compute factorial n (n!)
int factorial(int n) {
    if (n <= 1) return 1; //set factoral of 0 and 1 to be 1
    int factorial = 1;
    for (int i = 1; i <= n; ++i) factorial *= i;
    return factorial;
}

// compute double factorial  n (n!!)
int double_factorial(int n) {
    if (n <= 0) return 1; //set neg numbers to be factorial 1
    int result = 1;
    for (int i = n; i > 0; i -= 2) result *= i;
    return result;
}

// compute the Gaussian product center using Armadillo
// Formula: P = (alpha R_A + alpha R_B) / (alpha + β)

arma::vec compute_gaussian_product_center(const Gaussian &A, const Gaussian &B, size_t k, size_t l) {
    double alpha = A.exponent[k];
    double beta  = B.exponent[l];
    return (alpha * A.R + beta * B.R) / (alpha + beta);
    
}
// binomial coefficient (n choose k)
double binomial_coefficient(int n, int k) {
    if (k > n) return 0;
    return factorial(n) / (factorial(k) * factorial(n - k));
}

//  polynomial using binomial theorem to return a vector of terms 
std::vector<double> expand_polynomial(double A_coordinate, double B_coordinate, double P_center, int l_A, int l_B) {
    std::vector<double> polynomial_terms;
    for (int i = 0; i <= l_A; ++i) {
        for (int j = 0; j <= l_B; ++j) { 
            //compute binomial factors of A and B using binomial coefficient function
            double binomial_A = binomial_coefficient(l_A, i);
            double binomial_B = binomial_coefficient(l_B, j);
            //compute the first and second terms
            double first_term = std::pow(P_center - A_coordinate, l_A - i);
            double second_term = std::pow(P_center - B_coordinate, l_B - j);
            //first and second terms with the binomial coefficients
            double polynomial_value = binomial_A * binomial_B * first_term * second_term;
            polynomial_terms.push_back(polynomial_value);
        }
    }
    return polynomial_terms;
}




// Compute the exponential prefactor using Armadillo for gaussian product

double compute_exponential_prefactor(const Gaussian &A, const Gaussian &B, size_t k, size_t l) {
    double alpha = A.exponent[k];
    double beta  = B.exponent[l];
    double gamma = alpha + beta;

    double R_AB2 = arma::dot(A.R - B.R, A.R - B.R);  // ||R_A - R_B||^2
    double pre_exp = std::exp(-(alpha * beta / gamma) * R_AB2);

    double prefactor = std::pow(M_PI / gamma, 1.5) * pre_exp;
    return prefactor;
}




// Computes unnormalized primitive overlap S_kl between two primitives  --- EXTRACTED from S_AB 
double primitive_overlap(
    double alpha, arma::vec R_A, int l_A, int m_A, int n_A,
    double beta,  arma::vec R_B, int l_B, int m_B, int n_B
) {
    double gamma = alpha + beta;
    arma::vec R_P = (alpha * R_A + beta * R_B) / gamma;
    double prefactor = std::pow(M_PI / gamma, 1.5) *
                       std::exp(-(alpha * beta / gamma) * arma::dot(R_A - R_B, R_A - R_B));

    double Sx = 0.0, Sy = 0.0, Sz = 0.0;

    // x dimension
    for (int i = 0; i <= l_A; ++i) {
        for (int j = 0; j <= l_B; ++j) {
            if ((i + j) % 2 != 0) continue;
            double denom = std::pow(2 * gamma, (i + j) / 2.0);
            auto poly = expand_polynomial(R_A(0), R_B(0), R_P(0), l_A, l_B);
            int index = i * (l_B + 1) + j;
            if (index >= poly.size()) continue;
            double poly_term = poly[index];
            Sx += (double_factorial(i + j - 1) * poly_term) / denom;
        }
    }

    // y dimension
    for (int i = 0; i <= m_A; ++i) {
        for (int j = 0; j <= m_B; ++j) {
            if ((i + j) % 2 != 0) continue;
            double denom = std::pow(2 * gamma, (i + j) / 2.0);
            auto poly = expand_polynomial(R_A(1), R_B(1), R_P(1), m_A, m_B);
            int index = i * (m_B + 1) + j;
            if (index >= poly.size()) continue;
            double poly_term = poly[index];
            Sy += (double_factorial(i + j - 1) * poly_term) / denom;
        }
    }

    // z dimension
    for (int i = 0; i <= n_A; ++i) {
        for (int j = 0; j <= n_B; ++j) {
            if ((i + j) % 2 != 0) continue;
            double denom = std::pow(2 * gamma, (i + j) / 2.0);
            auto poly = expand_polynomial(R_A(2), R_B(2), R_P(2), n_A, n_B);
            int index = i * (n_B + 1) + j;
            if (index >= poly.size()) continue;
            double poly_term = poly[index];
            Sz += (double_factorial(i + j - 1) * poly_term) / denom;
        }
    }

    double result = prefactor * Sx * Sy * Sz;

    // std::cout << "[DEBUG][primitive_overlap] l_A=" << l_A << " m_A=" << m_A << " n_A=" << n_A
    //       << " | l_B=" << l_B << " m_B=" << m_B << " n_B=" << n_B
    //       << " | Sx=" << Sx << " Sy=" << Sy << " Sz=" << Sz << " → result=" << result << "\n";


    return result;
}



// Compute the Overlap Integral for S_AB between 2 Guassian functions using analytical summation
double S_AB_Summation(const Gaussian &A, const Gaussian &B) {
    double total_overlap = 0.0;
    for (size_t k = 0; k < A.exponent.size(); ++k) {
        for (size_t l = 0; l < B.exponent.size(); ++l) {
            double alpha = A.exponent[k];
            double beta  = B.exponent[l];
            arma::vec R_P = compute_gaussian_product_center(A, B, k, l);
            double prefactor = compute_exponential_prefactor(A, B, k, l);
    
            double Sx = 0.0, Sy = 0.0, Sz = 0.0;
    
            for (int dim = 0; dim < 3; ++dim) {
                auto l_A = A.L;
                auto l_B = B.L;
                
    
                for (int i = 0; i <= l_A(dim); ++i) {
                    for (int j = 0; j <= l_B(dim); ++j) {
                        if ((i + j) % 2 != 0) continue;
    
                        double denom = std::pow(2 * (alpha + beta), (i + j) / 2.0);
                        auto poly = expand_polynomial(A.R(dim), B.R(dim), R_P(dim), l_A(dim), l_B(dim));
                        int index = i * (l_B(dim) + 1) + j;
                        if (index >= poly.size()) continue;
                        double poly_term = poly[index];
    
                        double sum = (double_factorial(i + j - 1) * poly_term) / denom;
    
                        if (dim == 0) Sx += sum;
                        if (dim == 1) Sy += sum;
                        if (dim == 2) Sz += sum;
                    }
                }
            }
    
            double Nk = A.normalization_constants[k];
            double Nl = B.normalization_constants[l];
            double ck = A.contraction_coefficient[k];
            double cl = B.contraction_coefficient[l];
    
            total_overlap += ck * cl * Nk * Nl * prefactor * Sx * Sy * Sz;
        }
    }
    return total_overlap;
}