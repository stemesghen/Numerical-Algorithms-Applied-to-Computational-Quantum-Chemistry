#include <iostream> 
#include <fstream>
#include <filesystem>
#include <string> 
#include <cstdlib>
#include <stdexcept>

#include <nlohmann/json.hpp> // This is the JSON handling library
#include <armadillo> 

#include "header_file.hpp"




// convenience definitions so the code is more readable
namespace fs = std::filesystem;
using json = nlohmann::json; 


std::vector<Atom> read_xyz_file(const std::string& xyz_file_path) {
    std::ifstream file(xyz_file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open XYZ file: " + xyz_file_path);
    }

    int num_atoms;
    file >> num_atoms;

    std::string line;
    std::getline(file, line); 
    std::getline(file, line); 

    std::vector<Atom> atoms;
    for (int i = 0; i < num_atoms; ++i) {
        int atomic_number;
        double x, y, z;
        file >> atomic_number >> x >> y >> z;

        Atom atom;
        atom.atomic_number = atomic_number;
        atom.Z = atomic_number;
        atom.R = {x, y, z};
        atom.x = x;
        atom.y = y;
        atom.z = z;
        atom.symbol = ""; 

        atoms.push_back(atom);
    }

    return atoms;
}






/**
 * @brief Computes the x matrix for the CNDO/2 analytic gradient.
 *
 * The x matrix contains coefficients that multiply the derivative of the overlap matrix
 * elements (s_mu_nu) in the gradient expression of the CNDO/2 total energy. According to 
 * the derivation, x_{mu,nu} = P_total(mu,nu) * (beta_A + beta_B) for mu ≠ nu, and 0 for mu == nu.
 *
 * This function assumes an unrestricted SCF calculation, so it combines P_alpha and P_beta 
 * to get the total electron density matrix P_total.
 *
 * @param P_alpha The spin-alpha density matrix (N x N)
 * @param P_beta The spin-beta density matrix (N x N)
 * @param atom_for_orbital A vector mapping each atomic orbital index to its corresponding atom index
 * @param atoms A vector of Atom structs, each containing atomic information (e.g., atomic number)
 * @return arma::mat The x matrix (N x N) used in the CNDO/2 analytic gradient
 */
arma::mat compute_x_matrix(
    const arma::mat& P_alpha,
    const arma::mat& P_beta,
    const std::vector<int>& atom_for_orbital,
    const std::vector<Atom>& atoms
) {
    int N = P_alpha.n_rows;
    arma::mat x(N, N, arma::fill::zeros);

    arma::mat P_total = P_alpha + P_beta;

        // Loop over each pair of atomic orbitals (mu, nu)

    for (int mu = 0; mu < N; ++mu) {
        int A = atom_for_orbital[mu];
        double beta_A = CNDO_formula_parameters(atoms[A].atomic_number).beta;

        for (int nu = 0; nu < N; ++nu) {
            if (mu == nu) continue;// Skip diagonal elements (x(mu,mu) = 0 by definition)

            int B = atom_for_orbital[nu];
            double beta_B = CNDO_formula_parameters(atoms[B].atomic_number).beta;
            // Compute the off-diagonal x(mu,nu) value

            x(mu, nu) = P_total(mu, nu) * (beta_A + beta_B);
        }
    }

    return x;
}







/**
 * @brief Computes the derivative of the overlap matrix S_munu with respect to nuclear positions R_A.
 *
 * This function evaluates dS_munu/dR_A for all atom indices A and all AO pairs (mu, nu),
 * and stores the result in a 3×(N×N) matrix where N is the number of basis functions.
 * Each column corresponds to a mu, nu pair, and each row corresponds to a Cartesian component (x, y, z).
 *
 * The derivative is evaluated using the chain rule applied to the primitive Gaussian overlap integrals,
 * accounting for contributions when either mu or nu is centered on atom A.
 *
 * @param basis_functions Vector of contracted Gaussian basis functions (each contains exponents, contractions, etc.)
 * @param ao_atom_map A mapping from AO index to atom index
 * @param num_atoms The total number of atoms in the molecule
 * @param num_basis_functions The total number of atomic orbitals (basis functions)
 * @return arma::mat A 3×(N×N) matrix of overlap derivatives, where rows correspond to x/y/z and columns to AO pairs (mu,nu)
 */
arma::mat compute_Suv_RA(
    const std::vector<Gaussian>& basis_functions,
    const std::vector<int>& ao_atom_map,
    int num_atoms,
    int num_basis_functions
) {
    arma::mat Suv_RA(3, num_basis_functions * num_basis_functions, arma::fill::zeros);

    for (int A = 0; A < num_atoms; ++A) {
        for (int mu = 0; mu < num_basis_functions; ++mu) {
            for (int nu = 0; nu < num_basis_functions; ++nu) {
                int atom_mu = ao_atom_map[mu];
                int atom_nu = ao_atom_map[nu];
                int idx = mu * num_basis_functions + nu;

                // Only compute derivative if exactly one of mu or nu is on atom A
                bool mu_on_A = (atom_mu == A);
                bool nu_on_A = (atom_nu == A);
                if (mu_on_A == nu_on_A) continue; // skip if both or neither are on A

                arma::vec grad = arma::zeros(3);

                const Gaussian& g_mu = basis_functions[mu];
                const Gaussian& g_nu = basis_functions[nu];

                for (size_t k = 0; k < g_mu.exponent.size(); ++k) {
                    for (size_t l = 0; l < g_nu.exponent.size(); ++l) {
                        double alpha = g_mu.exponent[k];
                        double beta  = g_nu.exponent[l];
                        double ck    = g_mu.contraction_coefficient[k];
                        double cl    = g_nu.contraction_coefficient[l];
                        double Nk    = g_mu.normalization_constants[k];
                        double Nl    = g_nu.normalization_constants[l];

                        arma::vec dS_RA_nu;

                        if (mu_on_A) {
                            dS_RA_nu = derivative_primitive_overlap_RA(
                                alpha, g_mu.R, g_mu.L,
                                beta,  g_nu.R, g_nu.L
                            );
                            grad += ck * cl * Nk * Nl * dS_RA_nu;
                        } else { // nu_on_A
                            dS_RA_nu = derivative_primitive_overlap_RA(
                                beta, g_nu.R, g_nu.L,
                                alpha, g_mu.R, g_mu.L
                            );
                            grad -= ck * cl * Nk * Nl * dS_RA_nu;
                        }
                    }
                }

                for (int d = 0; d < 3; ++d) {
                    Suv_RA(d, idx) = grad(d);
                }
            }
        }
    }

    return Suv_RA;
}



/**
 * @brief Computes the derivative of a primitive Gaussian overlap integral with respect to the center of function A.
 *
 * Evaluates the 3-component vector:
 *     dS/dR_A = [dS/dR_Ax, dS/dR_Ay, dS/dR_Az]
 *
 * Uses known analytic formula for Cartesian Gaussian derivatives.
 * The derivative consists of two terms per Cartesian direction:
 * - A term involving lowering the angular momentum component on A 
 * - A term involving raising the component by 1 and multiplying 
 *
 * @param alpha Exponent of primitive Gaussian on atom A
 * @param R_A Cartesian center of Gaussian A
 * @param L_A Angular momentum tuple of Gaussian A (l, m, n)
 * @param beta Exponent of primitive Gaussian on atom B
 * @param R_B Cartesian center of Gaussian B
 * @param L_B Angular momentum tuple of Gaussian B (l, m, n)
 * @return arma::vec Gradient vector (3x1) representing derivative of S with respect to R_A
 */

arma::vec derivative_primitive_overlap_RA( 
    double alpha, arma::vec R_A, arma::Col<int> L_A,
    double beta,  arma::vec R_B, arma::Col<int> L_B
) {
    double l_A = L_A(0), m_A = L_A(1), n_A = L_A(2);
    double l_B = L_B(0), m_B = L_B(1), n_B = L_B(2);

    arma::vec grad = arma::zeros(3);  // (dx, dy, dz)


    //X-DERIVATIVE
    double term1_x = 0.0;
    if (l_A > 0) {
        term1_x = -l_A * primitive_overlap(alpha, R_A, l_A - 1, m_A, n_A,
                                           beta,  R_B, l_B,     m_B, n_B);
    }
    double term2_x = 2.0 * alpha * primitive_overlap(alpha, R_A, l_A + 1, m_A, n_A,
                                                     beta,  R_B, l_B,     m_B, n_B);
    grad(0) = term1_x + term2_x;

    // Y-DERIVATIVE
    double term1_y = 0.0;
    if (m_A > 0) {
        term1_y = -m_A * primitive_overlap(alpha, R_A, l_A, m_A - 1, n_A,
                                           beta,  R_B, l_B, m_B,     n_B);
    }
    double term2_y = 2.0 * alpha * primitive_overlap(alpha, R_A, l_A, m_A + 1, n_A,
                                                     beta,  R_B, l_B, m_B,     n_B);


    grad(1) = term1_y + term2_y;

    //Z-DERIVATIVE
    double term1_z = 0.0;
    if (n_A > 0) {
        term1_z = -n_A * primitive_overlap(alpha, R_A, l_A, m_A, n_A - 1,
                                           beta,  R_B, l_B, m_B, n_B);
    }
    double term2_z = 2.0 * alpha * primitive_overlap(alpha, R_A, l_A, m_A, n_A + 1,
                                                     beta,  R_B, l_B, m_B, n_B);
    grad(2) = term1_z + term2_z;

    return grad; 
}





/**
 * @brief Computes the spatial derivative of the gamma_AB integral with respect to R_A.
 *
 * This function implements the gradient of the gamma_AB electron repulsion term between 
 * two atoms A and B, assuming their basis functions are s-type contracted Gaussians.
 *
 * The derivative is taken with respect to the position of atom A and expressed as a 3D vector.
 * It uses two different formulas depending on whether A and B are colocated (R_AB → 0) or not.
 * The result is in atomic units and must be converted if needed.
 *
 * @param A Gaussian object for atom A (must be s-type)
 * @param B Gaussian object for atom B (must be s-type)
 * @return arma::vec A 3-element vector
 */
arma::vec compute_deriv_gamma(const Gaussian& A, const Gaussian& B) {
    arma::vec d_gamma(3, arma::fill::zeros);

    bool A_is_s = (arma::accu(A.L) == 0);
    bool B_is_s = (arma::accu(B.L) == 0);

    arma::vec R_A = A.R;
    arma::vec R_B = B.R;

    arma::vec dR = R_A - R_B;
    double R_AB2 = arma::dot(dR, dR);
    double R_AB  = std::sqrt(R_AB2);

    if (A_is_s && B_is_s) {
        for (size_t k = 0; k < 3; ++k) {
            for (size_t kp = 0; kp < 3; ++kp) {
                for (size_t l = 0; l < 3; ++l) {
                    for (size_t lp = 0; lp < 3; ++lp) {

                        double alpha_k  = A.exponent[k];
                        double alpha_kp = A.exponent[kp];
                        double beta_l   = B.exponent[l];
                        double beta_lp  = B.exponent[lp];

                        double ck   = A.contraction_coefficient[k];
                        double ckp  = A.contraction_coefficient[kp];
                        double cl   = B.contraction_coefficient[l];
                        double clp  = B.contraction_coefficient[lp];

                        double norm_k  = A.normalization_constants[k];
                        double norm_kp = A.normalization_constants[kp];
                        double norm_l  = B.normalization_constants[l];
                        double norm_lp = B.normalization_constants[lp];

                        // σ_A, σ_B
                        double sigma_A = 1.0 / (alpha_k + alpha_kp);
                        double sigma_B = 1.0 / (beta_l + beta_lp);

                        double UA = std::pow(M_PI * sigma_A, 1.5);
                        double UB = std::pow(M_PI * sigma_B, 1.5);

                        double V2 = compute_V_gamma(sigma_A, sigma_B);
                        double V  = std::sqrt(V2);

                        arma::vec d_0_term(3, arma::fill::zeros);

                        if (R_AB2 < 1e-10) {
                            // Eq. 3.15 (limit case)
                            double scalar = UA * UB * std::sqrt(2.0 * V2) * std::sqrt(2.0 / M_PI);
                            d_0_term = scalar * dR;
                        } else {
                            // Eq. 3.14 derivative
                            double T = V2 * R_AB2;
                            double erf_term = std::erf(std::sqrt(T));
                            double exp_term = std::exp(-T);

                            // This scalar is directly from the boxed formula
                            double scalar = (
                                (-erf_term / R_AB) + ((2.0 * V) / std::sqrt(M_PI)) * exp_term
                            ) / R_AB2;

                            d_0_term = UA * UB * dR * scalar;
                        }

                        double coeff = ck * ckp * cl * clp * norm_k * norm_kp * norm_l * norm_lp;
                        d_gamma += coeff * d_0_term;
                    }
                }
            }
        }
    }

    return d_gamma;
}





/**
 * @brief Constructs a matrix of gamma_AB gradient magnitudes for all pairs of atoms.
 *
 * This function uses `compute_deriv_gamma()` to compute the vector gradient of γ_AB
 * with respect to R_A.
 *
 * It only uses s-type orbitals for each atom (consistent with CNDO/2 theory).
 * The result is returned in eV units (conversion from Hartree using 27.2114).
 *
 * @param atoms Vector of Atom objects (must include atomic numbers, positions, etc.)
 * @param basis Vector of Gaussian basis functions (should contain s-orbitals for each atom)
 * @param atom_for_orbital A map from AO index to atom index
 * @return arma::mat A symmetric matrix of gradient magnitudes of gamma_AB in eV units
 */

arma::mat compute_deriv_gamma_matrix(const std::vector<Atom>& atoms, 
    const std::vector<Gaussian>& basis,
    const std::vector<int>& atom_for_orbital) 
{
size_t num_atoms = atoms.size();
arma::mat gamma_matrix(num_atoms, num_atoms, arma::fill::zeros);

// Find the first s-type basis function for each atom
std::vector<int> s_orbital_index(num_atoms, -1);
for (size_t mu = 0; mu < basis.size(); ++mu) {
int A = atom_for_orbital[mu];
if (s_orbital_index[A] == -1 && arma::accu(basis[mu].L) == 0) {
s_orbital_index[A] = mu;  // Save the index of the first s orbital
}
}

//builds gamma matrix using these s-orbitals
for (size_t A = 0; A < num_atoms; ++A) {
for (size_t B = 0; B <= A; ++B) {
int mu = s_orbital_index[A];
int nu = s_orbital_index[B];
if (mu == -1 || nu == -1) {
// std::cerr << "Missing s-orbital for atom " << A << " or " << B << std::endl;
continue;
}



arma::vec grad_gamma = compute_deriv_gamma(basis[mu], basis[nu]);
double gamma_val = arma::norm(grad_gamma);  
gamma_matrix(A, B) = gamma_matrix(B, A) = gamma_val * 27.2114;

}
}

return gamma_matrix;
}



/**
 * @brief Computes the gradient of the nuclear repulsion energy with respect to atomic positions.
 *
 * This function calculates the negative gradient (force) due to the classical Coulomb repulsion 
 * between each pair of nuclei. For each atom A, it sums over all other atoms B =! A:
 *
 *     = - Summation_{B =! A} (Z_A * Z_B * (R_A - R_B)) / |R_A - R_B|^3
 *
 * The result is returned as a 3 × Natoms matrix, where each column corresponds to
 * the x/y/z components of the force acting on atom A. The output is converted to eV.
 *
 * @param atoms A vector of Atom objects (each must contain .R and atomic number)
 * @return arma::mat A 3 × Natoms matrix containing the gradient in eV
 */

arma::mat compute_nuclear_repulsion_gradient(const std::vector<Atom>& atoms) {
    int num_atoms = atoms.size();
        // Vnuc_RA will store the 3D gradient for each atom A (as columns)

    arma::mat Vnuc_RA(3, num_atoms, arma::fill::zeros);

        // Loop over all atom pairs A, B

    for (int A = 0; A < num_atoms; ++A) {
        CNDOParameters params_A = CNDO_formula_parameters(atoms[A].atomic_number);
        arma::vec R_A = atoms[A].R;

        for (int B = 0; B < num_atoms; ++B) {
            CNDOParameters params_B = CNDO_formula_parameters(atoms[B].atomic_number);
            arma::vec R_B = atoms[B].R;

            if (A == B) continue;

                  // Vector from B to A

            arma::vec dR = R_A - R_B;
            // Distance between nuclei A and B

            double R_AB2 = arma::dot(dR, dR);
            double R_AB  = std::sqrt(R_AB2);// ||R_A - R_B||

            double Z_A = params_A.Z;
            double Z_B = params_B.Z;





//             std::cout << "Atom pair A=" << A << ", B=" << B << "\n";

//             std::cout << "  R_A = " << R_A.t();  
// std::cout << "  R_B = " << R_B.t();

// std::cout << "  |R_AB| = " << R_AB << std::endl;

            double z_term = (Z_A * Z_B);
            // std::cout << "Z_A = " << Z_A << ", Z_B = " << Z_B << "\n";



            arma::vec numerator_term = R_A - R_B;

            // std::cout << "  R_A - R_B = " << numerator_term.t();


            // grad = -Z_A * Z_B * (R_A - R_B) / |R_A - R_B|^3
            double  denom_term = std::pow(R_AB, 3);

            // std::cout << "  denom_term (R_AB^3) = " << denom_term << std::endl;
            arma::vec grad = - (z_term) * ((numerator_term) / (denom_term));
            // std::cout << "  grad = " << grad.t() << std::endl;

            
            Vnuc_RA.col(A) += grad;
            
        }
    }

    return 27.2114 * Vnuc_RA;  
}






/**
 * @brief Computes y_{AB}, the coefficient multiplying the derivative of gamma_{AB} in the CNDO/2 gradient expression.
 *
 * In CNDO/2, the energy is linear in gamm_{AB}, and this function computes the prefactor y_{AB}
 * by evaluating four components:
 *   1. P_AA tot * P_BB tot - 
 *   2. P AA tot * Z_B -  
 *   3. P_BB tot * Z_A
 *   4. - Summation_{mu in A} Summation_{nu in B} [ (Palpha_{mu,nu})^2 + (Pbeta_{mu,nu})^2 ]
 *
 * @param A Index of atom A
 * @param B Index of atom B
 * @param P_alpha The spin-alpha density matrix (N × N)
 * @param P_beta The spin-beta density matrix (N × N)
 * @param atom_for_orbital A mapping from AO index to atom index
 * @param atoms Vector of Atom objects
 * @return double The scalar y_{AB} used in dE/dR_A (gamma_AB term)
 */

double compute_yab(
    int A, int B,
    const arma::mat& P_alpha,
    const arma::mat& P_beta,
    const std::vector<int>& atom_for_orbital,
    const std::vector<Atom>& atoms
) {
    arma::mat P_total = P_alpha + P_beta;
    int num_orbitals = P_alpha.n_rows;

    // Compute P_AA and P_BB
    double P_AA = 0.0;
    double P_BB = 0.0;
    for (int mu = 0; mu < num_orbitals; ++mu) {
        if (atom_for_orbital[mu] == A)
            P_AA += P_total(mu, mu);
        else if (atom_for_orbital[mu] == B)
            P_BB += P_total(mu, mu);
    }


    double Z_A = CNDO_formula_parameters(atoms[A].atomic_number).Z;
    double Z_B = CNDO_formula_parameters(atoms[B].atomic_number).Z;

    double first_term  = P_AA * P_BB;
    double second_term = Z_B * P_AA;
    double third_term  = Z_A * P_BB;

    double fourth_term = 0.0;

    for (int mu = 0; mu < num_orbitals; ++mu) {
        if (atom_for_orbital[mu] != A) continue;

        for (int nu = 0; nu < num_orbitals; ++nu) {
            if (atom_for_orbital[nu] != B) continue;

            double term = std::pow(P_alpha(mu, nu), 2) + std::pow(P_beta(mu, nu), 2);
            fourth_term += term;

            // if ((A == 0 && B == 1) || (A == 1 && B == 0)) {
            //     std::cout << "[DEBUG] A=" << A << " B=" << B
            //               << " mu=" << mu << " nu=" << nu
            //               << " Pa=" << P_alpha(mu, nu)
            //               << " Pb=" << P_beta(mu, nu)
            //               << " contrib = " << term << "\n";
            // }
        }
    }

    return first_term - second_term - third_term - fourth_term;
}



/**
 * @brief Computes the electronic contribution to the CNDO/2 gradient of the SCF energy.
 *
 * Implements Equation 1.1 from the assignment:
 *     dE/dR_A = Summation_{mu =! nu} x_{mu, nu} ds_{mu, nu}/∂R_A + Summation_{B =! A} y_{AB} dgamma_{AB}/aR_A
 *
 * Handles contributions from:
 *  - Overlap matrix derivatives (x matrix and Suv_RA)
 *  - Gamma matrix derivatives (y matrix and gammaAB_RA)
 *
 * @param x The x_{mu,nu} matrix (N x N)
 * @param y The y_{AB} matrix (Natoms x Natoms)
 * @param Suv_RA Derivatives of overlap matrix dS_{mu,nu}/dR_A (3 x N^2)
 * @param gammaAB_RA Derivatives of gamma matrix dgamma_{AB}/dR_A (3 x Natoms^2)
 * @param atom_for_orbital Map from AO index to atom index
 * @param num_atoms Number of atoms in the system
 * @param num_basis_functions Number of atomic orbitals (basis functions)
 * @return arma::mat A 3 × Natoms matrix of electronic energy gradients in eV
 */


arma::mat compute_electronic_gradient(
    const arma::mat& x_matrix,
    const arma::mat& y_matrix,
    const arma::mat& Suv_RA,
    const arma::mat& gammaAB_RA,
    const std::vector<int>& atom_for_orbital,
    int num_atoms,
    int num_basis_functions
) {
    arma::mat gradient_electronic(3, num_atoms, arma::fill::zeros);  // (3 × Natoms)

    for (int mu = 0; mu < num_basis_functions; ++mu) {
        for (int nu = 0; nu < num_basis_functions; ++nu) {
            
            if (mu != nu) {

                int A = atom_for_orbital[mu];

                for (int d = 0; d < 3; ++d) {

                    int flat_index = mu * num_basis_functions + nu;

                    double s_mu_nu_RA = Suv_RA(d, flat_index);

                    gradient_electronic(d, A) += x_matrix(mu, nu) * s_mu_nu_RA;
                }
            }

        }
    }

    for (int A = 0; A < num_atoms; ++A) {

        for (int B = 0; B < num_atoms; ++B) {

            if (B != A) {

                int flat_index = A * num_atoms + B;

                for (int d = 0; d < 3; ++d) {

                    double gamma_ab_RA = gammaAB_RA(d, flat_index);

                    gradient_electronic(d, A) += y_matrix(A, B) * gamma_ab_RA;
                }
            }

        }
    }

    return gradient_electronic;
}




int main(int argc, char** argv){
    if (argc != 2){
        std::cerr << "Usage: " << argv[0] << " path/to/config.json" << std::endl; 
        return EXIT_FAILURE; 
    }
    // parse the config file 
    fs::path config_file_path(argv[1]);
    if (!fs::exists(config_file_path)){
        std::cerr << "Path: " << config_file_path << " does not exist" << std::endl; 
        return EXIT_FAILURE;
    }

    std::ifstream config_file(config_file_path); 
    json config = json::parse(config_file); 

    // extract the important info from the config file
    fs::path atoms_file_path = config["atoms_file_path"];
    fs::path output_file_path = config["output_file_path"];  
    int num_alpha_electrons = config["num_alpha_electrons"];
    int num_beta_electrons = config["num_beta_electrons"];

    std::vector<Atom> atoms = read_xyz_file(atoms_file_path);
    SCFResult result = run_scf(atoms, num_alpha_electrons, num_beta_electrons);
    


    BasisInfo basis_info = initialize_molecular_basis(atoms);


    int num_atoms =  atoms.size();; // You will have to replace this with the number of atoms in the molecule 
    int num_basis_functions = basis_info.molecular_basis.size(); // you will have to replace this with the number of basis sets in the molecule
    int num_3D_dims = 3; 


const auto& basis = basis_info.molecular_basis;
const auto& atom_for_orbital = basis_info.atom_for_orbital;

std::vector<int> atomic_numbers;
for (const auto& atom : atoms) atomic_numbers.push_back(atom.atomic_number);
std::vector<CNDOParameters> atom_params = get_all_CNDO_parameters(atomic_numbers);


// Compute s-orbital map for gammaAB_RA
std::vector<int> s_orbital_index(atoms.size(), -1);
for (size_t mu = 0; mu < basis.size(); ++mu) {
    int A = atom_for_orbital[mu];
    if (s_orbital_index[A] == -1 && arma::accu(basis[mu].L) == 0) {
        s_orbital_index[A] = mu;
    }
}




//  Compute overlap matrix S (in eV units)
arma::mat S = compute_overlap_matrix(basis, basis);
// std::cout << "Overlap matrix (eV):\n" << S << std::endl;


//for xmunu and yab
arma::mat P_alpha = arma::zeros(basis.size(), basis.size());
arma::mat P_beta = arma::zeros(basis.size(), basis.size());

//Ptotal after final iteration - linked to Hw 4_1.cpp
arma::mat P_total = result.P_total;


//compute x_mu_nu
arma::mat x = compute_x_matrix(
    result.P_alpha,
    result.P_beta,
    result.atom_for_orbital,
    atoms
);
x.print("x matrix");

// std::cout << "\n[DEBUG] x(mu,mu) diagonal elements:\n";
// for (int mu = 0; mu < x.n_rows; ++mu) {
//     std::cout << "x(" << mu << "," << mu << ") = " << x(mu, mu) << std::endl;
// }

// Create the Z vector 
std::vector<double> Z;
for (const auto& param : atom_params) Z.push_back(param.Z);

//compute y_ab
arma::mat y(num_atoms, num_atoms, arma::fill::zeros);
for (int A = 0; A < num_atoms; ++A) {
    for (int B = A + 1; B < num_atoms; ++B) {
        double val = compute_yab(A, B, result.P_alpha, result.P_beta, result.atom_for_orbital, atoms);
        y(A, B) = val;
y(B, A) = val;
    }
}

y.print("y matrix:");



//compute the derived S_mu_nu
arma::mat Suv_RA = compute_Suv_RA(basis, atom_for_orbital, num_atoms, num_basis_functions);
// std::cout << "\n[DEBUG] derived S munu  terms:\n";

// for (int mu = 0; mu < num_basis_functions; ++mu) {
//     int idx = mu * num_basis_functions + mu;
//     std::cout << "Suv_RA(1, " << idx << ") = " << Suv_RA(1, idx) << std::endl;
// }

    // for (int d = 0; d < 3; ++d) {
    //     // std::cout << "Suv_RA[" << d << "] " << "xyz"[d] << "):\n";
    //     arma::vec Sd_flat = Suv_RA.row(d).t();  // convert to column
    //     arma::mat Sd = arma::reshape(Sd_flat, num_basis_functions, num_basis_functions);
    //     Sd.raw_print(std::cout);
    //     std::cout << "\n";
    // }
    

//compute the derived gamma_ab
arma::mat gammaAB_RA(3, num_atoms * num_atoms, arma::fill::zeros);

    for (int A = 0; A < num_atoms; ++A) {
        for (int B = 0; B < num_atoms; ++B) {
            if (A == B) continue;
    
            int mu = s_orbital_index[A];
            int nu = s_orbital_index[B];
            if (mu == -1 || nu == -1) continue;
      
            
    
            const Gaussian& gA = basis[mu];
            const Gaussian& gB = basis[nu];
    
            arma::vec d_gamma_RA = compute_deriv_gamma(gA, gB);
    
            gammaAB_RA.col(A * num_atoms + B) = 27.2114 * d_gamma_RA;
            gammaAB_RA.col(B * num_atoms + A) = -27.2114 * d_gamma_RA;
        }

    }
    
    // std::cout << "\n gammaAB_RA norms \n";
    // for (int A = 0; A < num_atoms; ++A) {
    //     for (int B = 0; B < num_atoms; ++B) {
    //         if (A == B) continue;
    //         int idx = A * num_atoms + B;
    //         std::cout << "||γ(" << A << "," << B << ")|| = " << arma::norm(gammaAB_RA.col(idx)) << "\n";
    //     }
    // }



//compute the electronic gradient (derived E-CNDO/2 without the nuclear repulsion (Vnuc))
arma::mat gradient_electronic = compute_electronic_gradient(
    x, y, Suv_RA, gammaAB_RA, atom_for_orbital, num_atoms, num_basis_functions
);



// gradient_electronic.col(0).print("gradient_electronic for Atom 0");


//compute the nuclear repulsion (Vnuc)
arma::mat gradient_nuclear = compute_nuclear_repulsion_gradient(atoms);

//compute the total derived gradient (derived E-CNDO/2 from gradient electronic +  the nuclear repulsion (Vnuc))
arma::mat gradient(num_3D_dims, num_atoms);  
gradient = gradient_electronic + gradient_nuclear;



// Check for translational invariance 
// arma::rowvec sum_force = arma::sum(gradient, 1).t(); 
// sum_force.print("Sum of total gradient (should be 0 if translationally invariant):");



    // You do not need to modify the code below this point 

    // Set print configs
    std::cout << std::fixed << std::setprecision(4) << std::setw(8) << std::right ; 

    // inspect your answer via printing
    Suv_RA.print("Suv_RA");
    Suv_RA.save("Suv_RA.txt", arma::raw_ascii);
    gammaAB_RA.print("gammaAB_RA");
    gradient_nuclear.print("gradient_nuclear");
    gradient_electronic.print("gradient_electronic");
    gradient.print("gradient"); 

    // check that output dir exists
    if (!fs::exists(output_file_path.parent_path())){
        fs::create_directories(output_file_path.parent_path()); 
    }
    
    // delete the file if it does exist (so that no old answers stay there by accident)
    if (fs::exists(output_file_path)){
        fs::remove(output_file_path); 
    }

    // write results to file 
    Suv_RA.save(arma::hdf5_name(output_file_path, "Suv_RA", arma::hdf5_opts::append));
    Suv_RA.save("Suv_RA.txt", arma::raw_ascii);
    gammaAB_RA.save(arma::hdf5_name(output_file_path, "gammaAB_RA", arma::hdf5_opts::append));
    gradient_nuclear.save(arma::hdf5_name(output_file_path, "gradient_nuclear", arma::hdf5_opts::append));
    gradient_electronic.save(arma::hdf5_name(output_file_path, "gradient_electronic", arma::hdf5_opts::append));
    gradient.save(arma::hdf5_name(output_file_path, "gradient", arma::hdf5_opts::append));
    
}  


