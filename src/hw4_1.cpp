


#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <armadillo>
#include <cassert>

#define _USE_MATH_DEFINES  
#include <cmath>           

#include "header_file.hpp"


// const bool DEBUG = true;


// MoleculeInput read_molecule_input(const std::string& filename) {
//     std::ifstream inputFile(filename);
//     if (!inputFile) {
//         throw std::runtime_error("Error: Cannot open file " + filename);
//     }

//     MoleculeInput input;

//     int num_atoms;
//     inputFile >> num_atoms >> input.num_alpha_electrons >> input.num_beta_electrons;

//     for (int i = 0; i < num_atoms; ++i) {
//         Atom atom;
//         inputFile >> atom.atomic_number >> atom.x >> atom.y >> atom.z;
//         input.atoms.push_back(atom);
//     }

//     inputFile.close();
//     return input;
// }



SCFResult run_scf(
    const std::vector<Atom>& atoms,
    int num_alpha_electrons,
    int num_beta_electrons
) {
    int p = num_alpha_electrons;
    int q = num_beta_electrons;



//Initialize basis set and mappings
BasisInfo basis_info = initialize_molecular_basis(atoms);
const auto& basis = basis_info.molecular_basis;
const auto& atom_for_orbital = basis_info.atom_for_orbital;
int num_AOs = basis.size();




arma::vec eps_alpha, eps_beta; 
arma::mat C_alpha, C_beta;

//Precompute gamma matrix between atoms (atom-based, not orbital-based)
arma::mat gamma_matrix = compute_gamma_matrix(atoms, basis, atom_for_orbital);
// std::cout << "Gamma matrix (eV):\n" << gamma_matrix << std::endl;


// Compute overlap matrix S (in eV units)
arma::mat S = compute_overlap_matrix(basis, basis);
// std::cout << "Overlap matrix (eV):\n" << S << std::endl;

// std::cout << " p =  " << p << " q =  " << q << "\n";


//Build core Hamiltonian (H = T + V) from CNDO/2 model
arma::mat H_core = compute_core_hamiltonian(basis, atom_for_orbital, atoms, gamma_matrix);  
// std::cout << "H_core:\n" << H_core << std::endl;

//Initialize SCF
//Initialize spin density matrices (alpha and beta)
arma::mat P_alpha = arma::zeros(basis.size(), basis.size());
arma::mat P_beta = arma::zeros(basis.size(), basis.size());
arma::mat P_total = P_alpha + P_beta;


arma::mat F_alpha;
arma::mat F_beta;



double energy_old = 0.0;
for (int iter = 0; iter < 100; ++iter) {
    // std::cout << "Iteration: " << iter << "\n";
   
            // Compute P_AA each iteration
            std::vector<double> p_AA = compute_p_AA(P_total, atom_for_orbital, atoms.size());



        F_alpha = build_fock_alpha(P_alpha, P_total, gamma_matrix, S, basis, atom_for_orbital, atoms);
        F_beta  = build_fock_beta(P_beta, P_total, gamma_matrix, S, basis, atom_for_orbital, atoms);
    

    
    // std::cout << "F_alpha:\n" << F_alpha << std::endl;
    // std::cout << "F_beta:\n" << F_beta << std::endl;
                 


    solve_eigen_problem(F_alpha, C_alpha, eps_alpha);
    solve_eigen_problem(F_beta,  C_beta,  eps_beta);

    // std::cout << "C_alpha (MO coefficients):\n" << C_alpha << std::endl;
    // std::cout << "C_beta (MO coefficients):\n" << C_beta << std::endl;

    
   compute_density_matrices(C_alpha, C_beta, p, q, P_alpha, P_beta, P_total);
    // std::cout << "P_new_alpha:\n" << P_alpha << std::endl;
    // std::cout << "P_new_beta:\n" << P_beta << std::endl; 
    // std::cout << "P_new_total:\n" << P_total << std::endl;


    // std::cout << "Trace P_alpha = " << arma::trace(P_alpha) << "\n";
    // std::cout << "Trace P_beta  = " << arma::trace(P_beta) << "\n";
    // std::cout << "Expected alpha electrons = " << p << ", beta electrons = " << q << "\n";

    double E_alpha = calculate_alpha_energy(P_alpha, H_core, F_alpha);
    // std::cout << "E_alpha:\n" << E_alpha << std::endl;

    double E_beta  = calculate_beta_energy(P_beta,  H_core, F_beta);
    // std::cout << "E_beta:\n" << E_beta << std::endl;

    double E_nuc   = nuclear_repulsion(atoms); 
    // std::cout << "Nuclear Repulsion:\n" << E_nuc << std::endl;

    double total_energy = calculate_total_energy(P_alpha, P_beta, H_core, F_alpha, F_beta, atoms);
    // std::cout << "Iteration " << iter << ": Total Energy = " << total_energy << " eV\n";

    // std::cout << "Electron Energy = " << E_alpha + E_beta << " eV\n";


//     std::cout << "Electron counts:" << std::endl;
// std::cout << "  p (alpha) = " << p << std::endl;
// std::cout << "  q (beta)  = " << q << std::endl;
// std::cout << "  Total electrons = " << p + q << std::endl;




            // Check convergence 
            if (iter > 0 && std::abs(total_energy - energy_old) < 1e-6) {
                // std::cout << "SCF converged after " << iter << " iterations!\n";

                std::cout << "Nuclear Repulsion:\n" << E_nuc << std::endl;
                std::cout << "Electron Energy = " << E_alpha + E_beta << " eV\n";


                break;

            }
            energy_old = total_energy;
        }

        // Package result
    SCFResult result;
    result.P_alpha = P_alpha;
    result.P_beta  = P_beta;
    result.P_total = P_total;
    result.C_alpha = C_alpha;
    result.C_beta  = C_beta;
    result.eps_alpha = eps_alpha;
    result.eps_beta  = eps_beta;
    result.F_alpha = F_alpha;
    result.F_beta  = F_beta;
    result.gamma_matrix = gamma_matrix;
    result.S = S;
    result.basis = basis;
    result.atom_for_orbital = atom_for_orbital;



    

    return result;

    }








// /**
//  * @brief Solves the symmetric eigenvalue problem for a Fock matrix.
//  * 
//  * Diagonalizes the symmetric matrix F to obtain molecular orbital energies (eigenvalues)
//  * and molecular orbital coefficients (eigenvectors).
//  * 
//  * @param F     The symmetric Fock matrix (input).
//  * @param C     Output: Matrix of molecular orbital (MO) coefficients. Each column is an MO.
//  * @param eps   Output: Vector of MO energies (eigenvalues), sorted in ascending order.
//  */

void solve_eigen_problem(
    const arma::mat& F,     // Fock matrix
    arma::mat& C,           // Output: MO coefficients
    arma::vec& eps          // Output: MO energies
) {
    arma::eig_sym(eps, C, F);  // Diagonalizes symmetric matrix F
}


// /**
//  * @brief Retrieves CNDO/2 empirical parameters for a given atomic number.
//  * 
//  * The returned CNDOParameters struct contains (eV):
//  *   - beta     = resonance integral
//  *   - gamma    = two-electron repulsion for s orbitals
//  *   - Is_As    = ionization potential for s orbitals
//  *   - Ip_Ap    = ionization potential for p orbitals
//  *   - Z        = effective valence charge used in CNDO/2
//  * 
//  * @param atomic_number The atomic number (Z) of the atom (e.g. 1 = H, 6 = C, etc.).
//  * @return CNDOParameters struct containing relevant CNDO/2 values.
//  * 
//  * @throws std::invalid_argument if the element is unsupported.
//  */

CNDOParameters CNDO_formula_parameters(int atomic_number) {
    switch (atomic_number) {
        case 1:  // Hydrogen
            return {7.176 , 0.0, -9.0, 1};
        case 6:  // Carbon
            return {14.051, 5.572, -21.0, 4};
        case 7:  // Nitrogen
            return {19.316, 7.275, -25.0, 5};
        case 8:  // Oxygen
            return {25.390, 9.111, -31.0, 6};
        case 9:  // Fluorine
            return {32.272, 11.080, -39.0, 7};
        default:
            throw std::invalid_argument("CNDO/2 parameters Invalid Arg = " + std::to_string(atomic_number));
    }
}
/**
 * @brief Retrieves CNDO/2 parameters for a list of atomic numbers.
 * 
 * @param atomic_numbers A vector of atomic numbers corresponding to the atoms in the molecule.
 * @return A vector of CNDOParameters, one per atom, in the same order as input.
 */

std::vector<CNDOParameters> get_all_CNDO_parameters(const std::vector<int>& atomic_numbers) {
    std::vector<CNDOParameters> params;
    for (int Z : atomic_numbers) {
        params.push_back(CNDO_formula_parameters(Z));
    }
    return params;
}






/**
 * @brief Computes the alpha-spin electron density matrix.
 * 
 * Constructs the alpha density matrix by summing over the outer products of the first `p` molecular orbitals,
 * which are assumed to be occupied by alpha electrons.
 * 
 * @param C_alpha  Molecular orbital coefficient matrix for alpha spin. Each column is an MO.
 * @param p        Number of occupied alpha-spin orbitals (i.e., number of alpha electrons).
 * 
 * @return arma::mat The alpha density matrix P_alpha of size N x N.
 */


arma::mat compute_P_alpha(const arma::mat& C_alpha, int p) {
    arma::mat P_alpha = arma::zeros(C_alpha.n_rows, C_alpha.n_rows);
    for (int i = 0; i < p; ++i) 
        P_alpha += C_alpha.col(i) * C_alpha.col(i).t();
    return P_alpha;
}



// P_beta
/**
 * @brief Computes the beta-spin electron density matrix.
 * 
 * Constructs the beta density matrix by summing over the outer products of the first `q` molecular orbitals,
 * which are assumed to be occupied by beta electrons.
 * 
 * @param C_beta  Molecular orbital coefficient matrix for beta spin. Each column is an MO.
 * @param q       Number of occupied beta-spin orbitals (i.e., number of beta electrons).
 * 
 * @return arma::mat The beta density matrix P_beta of size N x N.
 */

arma::mat compute_P_beta(const arma::mat& C_beta, int q) {
    arma::mat P_beta = arma::zeros(C_beta.n_rows, C_beta.n_rows);
    for (int i = 0; i < q; ++i) 
        P_beta += C_beta.col(i) * C_beta.col(i).t();
    return P_beta;
}

/**
 * @brief Computes alpha, beta, and total electron density matrices.
 * 
 * Uses the molecular orbital coefficients to build:
 *   - P_alpha: sum over outer products of the first `p` alpha-spin orbitals.
 *   - P_beta:  sum over outer products of the first `q` beta-spin orbitals.
 *   - P_total = P_alpha + P_beta.
 * 
 * @param C_alpha   Molecular orbital coefficients for alpha spin.
 * @param C_beta    Molecular orbital coefficients for beta spin.
 * @param p         Number of occupied alpha-spin orbitals.
 * @param q         Number of occupied beta-spin orbitals.
 * @param P_alpha   Output: alpha density matrix (will be filled in-place).
 * @param P_beta    Output: beta density matrix (will be filled in-place).
 * @param P_total   Output: total density matrix (alpha + beta).
 */
void compute_density_matrices(
    const arma::mat& C_alpha, const arma::mat& C_beta,
    int p, int q,
    arma::mat& P_alpha, arma::mat& P_beta, arma::mat& P_total
) {
    P_alpha.zeros();
    P_beta.zeros();

    // Sum over occupied MOs (p for alpha, q for beta)
    for (int i = 0; i < p; ++i)
        P_alpha += C_alpha.col(i) * C_alpha.col(i).t();
    for (int i = 0; i < q; ++i)
        P_beta  += C_beta.col(i)  * C_beta.col(i).t();

    P_total = P_alpha + P_beta;
}


// void compute_density_matrices(
//     const arma::mat& C_alpha,
//     const arma::mat& C_beta,
//     int p, int q,
//     arma::mat& P_alpha,
//     arma::mat& P_beta,
//     arma::mat& P_total
// ) {
//     P_alpha.zeros();
//     P_beta.zeros();

//     for (int i = 0; i < p; ++i)
//         P_alpha += C_alpha.col(i) * C_alpha.col(i).t();
//     for (int i = 0; i < q; ++i)
//         P_beta += C_beta.col(i) * C_beta.col(i).t();

//     P_total = P_alpha + P_beta;

//     // p_total
//     std::cout << "P_total:\n" << P_total << std::endl;
// }

/**
 * @brief Determines whether a given atomic orbital is an s-type orbital.
 * 
 * An s-orbital is defined by having zero angular momentum in all directions,
 * i.e., (l_x, l_y, l_z) = (0, 0, 0).
 * 
 * @param orbital_index          Index of the orbital in the basis set.
 * @param basis_functions        Vector of Gaussian basis functions for the molecule.
 * 
 * @return true                  If the orbital is s-type.
 * @return false                 If the orbital has any angular momentum (p-, d-, etc.).
 */

// check if an orbital is s-type
bool is_s_orbital(int orbital_index, const std::vector<Gaussian>& basis_functions) {
    // Get the angular momentum values (l_x, l_y, l_z)
    int lx = basis_functions[orbital_index].L[0];
    int ly = basis_functions[orbital_index].L[1];
    int lz = basis_functions[orbital_index].L[2];
    
    // If all are zero, it's an s-orbital
    if (lx == 0 && ly == 0 && lz == 0) {
        return true;
    } else {
        return false;
    }
}

//Fock matrix calculation using is_s_orbital function
/**
 * @brief Computes the first diagonal term of the CNDO/2 Fock matrix for a given orbital.
 * 
 * This term accounts for the one-electron energy of an orbital based on its type:
 *   - s-orbital → use the -Is_As parameter,
 *   - p-orbital → use the -Ip_Ap parameter.
 * The correct parameter is selected using `is_s_orbital(...)`.
 * 
 * @param orbital_index      Index of the orbital for which the first term is computed.
 * @param basis              Vector of Gaussian basis functions.
 * @param params             CNDO/2 parameters for the atom that owns the orbital.
 * 
 * @return double            The first term in the Fock matrix diagonal for this orbital.
 */
double calculate_first_term(int orbital_index, 
                          const std::vector<Gaussian>& basis,
                          const CNDOParameters& params) {
    
    if (is_s_orbital(orbital_index, basis)) {
        // Use s-orbital parameter
        return -params.Is_As;
    } else {
        // Use p-orbital parameter
        return -params.Ip_Ap;
    }
}


/**
 * @brief Computes the total electron population (P_AA) on each atom.
 * 
 * Sums the diagonal elements of the total electron density matrix for all orbitals
 * that belong to a given atom. Used in the Fock matrix diagonal construction.
 * 
 * @param P_total            Total electron density matrix (P_alpha + P_beta).
 * @param atom_for_orbital   Mapping from orbital index to owning atom index.
 * @param num_atoms          Total number of atoms in the molecule.
 * 
 * @return std::vector<double>   Vector of length `num_atoms` containing total
 *                               electron populations on each atom (P_AA).
 */
std::vector<double> compute_p_AA(const arma::mat& P_total, 
    const std::vector<int>& atom_for_orbital, 
    int num_atoms) {
    
    std::vector<double> p_AA(num_atoms, 0.0);
    for (int mu = 0; mu < P_total.n_rows; ++mu) {
        int A = atom_for_orbital[mu];
        p_AA[A] += P_total(mu, mu);
    }
    return p_AA;
}



/**
 * @brief Constructs the alpha-spin Fock matrix in the CNDO/2 method.
 *
 * This function computes the full Fock matrix for alpha-spin electrons
 * using:
 *   - the density matrices (P_alpha, P_total),
 *   - atom-atom interaction matrix (gamma),
 *   - the overlap matrix (S),
 *   - Gaussian basis functions, and
 *   - CNDO/2 atomic parameters.
 *
 * The Fock matrix includes:
 *   - Diagonal terms: first (orbital energy), second (electron repulsion on same atom),
 *                     third (interactions with other atoms).
 *   - Off-diagonal terms: using overlap S(mu,nu) and gamma(A,B).
 *
 * @param P_alpha           Alpha electron density matrix.
 * @param P_total           Total electron density matrix (P_alpha + P_beta).
 * @param gamma_matrix      Matrix of gamma_AB values between atoms (in eV).
 * @param S                 Overlap matrix between basis functions.
 * @param basis             Vector of contracted Gaussian basis functions.
 * @param atom_for_orbital  Mapping from orbital index to atom index.
 * @param atoms             Vector of Atom structs containing geometry and Z.
 *
 * @return arma::mat        The resulting alpha-spin Fock matrix.
 */

 double compute_f_offdiagonal_alpha(
    int mu, int nu,                    // Orbital indices
    const arma::mat& S,                // Overlap matrix
    const arma::mat& P_alpha,          // Alpha density matrix
    double gamma_AB,                   // gamma between atoms A and B
    const std::vector<int>& atom_for_orbital,
    const std::vector<CNDOParameters>& params
) {
    int A = atom_for_orbital[mu];      // Atom for orbital mu
    int B = atom_for_orbital[nu];      // Atom for orbital nu
    
    // Average of BETA parameters for atoms A and B
    double beta_avg = 0.5 * (params[A].beta + params[B].beta);
    
    // CNDO/2 off-diagonal formula
    return beta_avg * S(mu, nu) - P_alpha(mu, nu) * gamma_AB;
}

arma::mat build_fock_alpha(
    const arma::mat& P_alpha,         // Alpha electron density matrix
    const arma::mat& P_total,         // Total electron density matrix (alpha + beta)
    const arma::mat& gamma_matrix,    // Precomputed gamma matrix for electron repulsion
    const arma::mat& S,               // Overlap matrix between atomic orbitals
    const std::vector<Gaussian>& basis, // List of atomic orbitals
    const std::vector<int>& atom_for_orbital, // Which atom each orbital belongs to
    const std::vector<Atom>& atoms    // List of atoms in the molecule
) {
    int num_orbitals = basis.size();
    int num_atoms = atoms.size();

    arma::mat F_alpha(num_orbitals, num_orbitals, arma::fill::zeros);

    //  Atom populations
    std::vector<double> atom_population(num_atoms, 0.0);
    for (int mu = 0; mu < num_orbitals; ++mu) {
        int A = atom_for_orbital[mu];
        atom_population[A] += P_total(mu, mu);
    }

    //Diagonal elements
    for (int mu = 0; mu < num_orbitals; ++mu) {
        int A = atom_for_orbital[mu];
        auto atom_params = CNDO_formula_parameters(atoms[A].atomic_number);

        bool is_s_orbital = (basis[mu].L[0] == 0 && 
                             basis[mu].L[1] == 0 && 
                             basis[mu].L[2] == 0);
        double one_electron_term = is_s_orbital ?
                                   -atom_params.Is_As :
                                   -atom_params.Ip_Ap;

        double same_atom_repulsion = 
            (atom_population[A] - atom_params.Z - (P_alpha(mu, mu) - 0.5)) 
            * gamma_matrix(A, A);

        double other_atoms_repulsion = 0.0;
        for (int B = 0; B < num_atoms; ++B) {
            if (B == A) continue;
            auto other_params = CNDO_formula_parameters(atoms[B].atomic_number);
            other_atoms_repulsion += 
                (atom_population[B] - other_params.Z) * gamma_matrix(A, B);
        }

        F_alpha(mu, mu) = one_electron_term + same_atom_repulsion + other_atoms_repulsion;
    }

    //  Off-diagonal elements
    for (int mu = 0; mu < num_orbitals; ++mu) {
        for (int nu = mu + 1; nu < num_orbitals; ++nu) {
            int A = atom_for_orbital[mu];
            int B = atom_for_orbital[nu];

            auto params_A = CNDO_formula_parameters(atoms[A].atomic_number);
            auto params_B = CNDO_formula_parameters(atoms[B].atomic_number);
            double average_beta = 0.5 * (params_A.beta + params_B.beta);

            F_alpha(mu, nu) = average_beta * S(mu, nu)
                              - P_alpha(mu, nu) * gamma_matrix(A, B);
            F_alpha(nu, mu) = F_alpha(mu, nu); // symmetric
        }
    }

    return F_alpha;
}





/**
 * @brief Constructs the beta-spin Fock matrix in the CNDO/2 method.
 *
 * This function computes the full Fock matrix for beta-spin electrons
 * using the same approach as `build_fock_alpha`, but based on the
 * beta density matrix (P_beta).
 *
 * The Fock matrix includes:
 *   - Diagonal terms: orbital energy, same-atom electron repulsion,
 *                     interactions with other atoms.
 *   - Off-diagonal terms: based on overlap and gamma_AB values.
 *
 * @param P_beta            Beta electron density matrix.
 * @param P_total           Total electron density matrix (P_alpha + P_beta).
 * @param gamma_matrix      Matrix of gamma_AB values between atoms (in eV).
 * @param S                 Overlap matrix between basis functions.
 * @param basis             Vector of contracted Gaussian basis functions.
 * @param atom_for_orbital  Mapping from orbital index to atom index.
 * @param atoms             Vector of Atom structs containing geometry and Z.
 *
 * @return arma::mat        The resulting beta-spin Fock matrix.
 */

// Function to build the beta Fock matrix
arma::mat build_fock_beta(
    const arma::mat& P_beta,          // Beta electron density matrix
    const arma::mat& P_total,         // Total electron density matrix (alpha + beta)
    const arma::mat& gamma_matrix,    // Precomputed gamma matrix for electron repulsion
    const arma::mat& S,               // Overlap matrix between atomic orbitals
    const std::vector<Gaussian>& basis, // List of atomic orbitals
    const std::vector<int>& atom_for_orbital, // Which atom each orbital belongs to
    const std::vector<Atom>& atoms    // List of atoms in the molecule
) {
    // Get the number of orbitals and atoms
    int num_orbitals = basis.size();
    int num_atoms = atoms.size();
    
    // Create a zero matrix for the Fock matrix
    arma::mat F_beta(num_orbitals, num_orbitals, arma::fill::zeros);
    
    // Calculate total electron population on each atom (P_AA)
    std::vector<double> atom_population(num_atoms, 0.0);
    for (int mu = 0; mu < num_orbitals; ++mu) {
        int A = atom_for_orbital[mu];  // Which atom this orbital is on
        atom_population[A] += P_total(mu, mu); // Add electron density
    }

    // STEP 2: Build diagonal elements (mu == nu)
    for (int mu = 0; mu < num_orbitals; ++mu) {
        int A = atom_for_orbital[mu];  // Atom this orbital is on
        auto atom_params = CNDO_formula_parameters(atoms[A].atomic_number);
        
  
        bool is_s_orbital = (basis[mu].L[0] == 0 && 
                           basis[mu].L[1] == 0 && 
                           basis[mu].L[2] == 0);
        
        double one_electron_term = is_s_orbital ? 
                                 -atom_params.Is_As :  // For s orbitals
                                 -atom_params.Ip_Ap;   // For p orbitals
        
        //  Electron-electron repulsion on same atom
        double same_atom_repulsion = 
            (atom_population[A] - atom_params.Z - (P_beta(mu, mu) - 0.5)) 
            * gamma_matrix(A, A);
        
        // Electron-electron repulsion from other atoms
        double other_atoms_repulsion = 0.0;
        for (int B = 0; B < num_atoms; ++B) {
            if (B == A) continue;  // Skip the current atom
            auto other_params = CNDO_formula_parameters(atoms[B].atomic_number);
            other_atoms_repulsion += 
                (atom_population[B] - other_params.Z) * gamma_matrix(A, B);
        }
        
        // Combine all terms for the diagonal element
        F_beta(mu, mu) = one_electron_term + same_atom_repulsion + other_atoms_repulsion;
    }

    // Build off-diagonal elements (mu != nu)
    for (int mu = 0; mu < num_orbitals; ++mu) {
        for (int nu = mu + 1; nu < num_orbitals; ++nu) {  
            int A = atom_for_orbital[mu];  // Atom for orbital mu
            int B = atom_for_orbital[nu];  // Atom for orbital nu
            
            // Get parameters for both atoms
            auto params_A = CNDO_formula_parameters(atoms[A].atomic_number);
            auto params_B = CNDO_formula_parameters(atoms[B].atomic_number);
            
            // Average the beta parameters
            double average_beta = 0.5 * (params_A.beta + params_B.beta);
            
            // Calculate the off-diagonal element
            F_beta(mu, nu) = average_beta * S(mu, nu)           // Bonding term
                            - P_beta(mu, nu) * gamma_matrix(A, B); // Repulsion term
            
            // Make the matrix symmetric
            F_beta(nu, mu) = F_beta(mu, nu);
        }
    }
    
    return F_beta;
}




/**
 * @brief Computes the prefactor V for two Gaussian primitives.
 *
 * Used in electron repulsion integrals
 *
 * @param sigma_A  Inverse exponent for Gaussian A.
 * @param sigma_B  Inverse exponent for Gaussian B.
 * @return double  Value of V = 1 / (sigma_A + sigma_B).
 */

//compute V
double compute_V_gamma(double sigma_A, double sigma_B) {

    return 1.0 / (sigma_A + sigma_B);
}




/**
 * @brief Computes the Boys function argument T = V * R^2.
 *
 * This is used in evaluating electron repulsion integrals between Gaussians.
 *
 * @param sigma_A  Inverse exponent for Gaussian A.
 * @param sigma_B  Inverse exponent for Gaussian B.
 * @param R_A      Center of Gaussian A (3D vector).
 * @param R_B      Center of Gaussian B (3D vector).
 * @return double  Value of T = V * |R_A - R_B|^2.
 */
double compute_T_gamma(double sigma_A, double sigma_B, const arma::vec& R_A, const arma::vec& R_B) {
    double V2 = 1.0 / (sigma_A + sigma_B);
    double R_AB2 = arma::dot((R_A - R_B), (R_A - R_B));
    return V2 * R_AB2;
}


/**
 * @brief Computes the Boys function F₀(T) used in Coulomb integrals.
 *
 * \f[
 * F_0(T) = 
 * \begin{cases}
 * 1, & T \to 0 \\
 * \frac{1}{2} \sqrt{\frac{\pi}{T}} \text{erf}(\sqrt{T}), & T > 0
 * \end{cases}
 * \f]
 *
 * @param T  Argument of the Boys function.
 * @return double  Value of F₀(T).
 */
double compute_boys_function(double T) {
    if (T < 1e-8) {
        // Use limiting value: lim_{T → 0} F₀(T) = 1
        return 1.0;
    }
    return 0.5 * std::sqrt(M_PI / T) * std::erf(std::sqrt(T));
}


/**
 * @brief Computes the two-electron repulsion integral γ(A,B) between two s-type Gaussians.
 *
 * Implements the CNDO/2 approximation for gamma integrals between contracted Gaussians
 * using analytical formulas (Eq. 3.14 or 3.15) depending on whether centers overlap.
 * 
 * Only works when both Gaussians are s-type (L = 0,0,0).
 *
 * @param A  Gaussian basis function on atom A.
 * @param B  Gaussian basis function on atom B.
 * @return double  Value of the gamma integral in Hartree.
 */

double compute_gamma(const Gaussian& A, const Gaussian& B) {
    double gamma = 0.0;
    bool A_is_s = (arma::accu(A.L) == 0); // s-orbital if all L=0
    bool B_is_s = (arma::accu(B.L) == 0);


    arma::vec R_A = A.R;
    arma::vec R_B = B.R;

    double R_AB2 = arma::dot((R_A - R_B), (R_A - R_B));  // ||R_A - R_B||²

    if (A_is_s && B_is_s) {


    for (size_t k = 0; k < 3; k ++) {
        for (size_t k_prime = 0; k_prime < 3; k_prime ++) {
            for (size_t l = 0; l < 3; l ++) {
                for(size_t l_prime = 0; l_prime < 3; l_prime ++) {
                   

                    double alpha_k = A.exponent[k];
                    double alpha_k_prime = A.exponent[k_prime];
                    double beta_l = B.exponent[l];
                    double beta_l_prime = B.exponent[l_prime];

                    double alpha_ck  = A.contraction_coefficient[k];
                    double alpha_ck_prime = A.contraction_coefficient[k_prime];
                    double beta_cl  = B.contraction_coefficient[l];
                    double beta_cl_prime = B.contraction_coefficient[l_prime];

                    //compute sigma_A and sigma_B
                    double sigma_A = 1.0 / (alpha_k + alpha_k_prime);
                    double sigma_B = 1.0 / (beta_l + beta_l_prime);

                    //compute UA and UB
                    double UA = std::pow(M_PI * sigma_A, 1.5);
                    double UB = std::pow(M_PI * sigma_B, 1.5);

                    // double prefactor = 2.0 * std::pow(M_PI, 2.5);



                    double V2 = compute_V_gamma(sigma_A, sigma_B);


//dk = ck * norm
double norm_k  = A.normalization_constants[k];
double norm_kp = A.normalization_constants[k_prime];
double norm_l  = B.normalization_constants[l];
double norm_lp = B.normalization_constants[l_prime];



double zero_integral = 0.0;
                    
if (R_AB2 < 1e-10) {
    // Eq. 3.15
    double sqrt_2V2 = std::sqrt(2.0 * V2);
    double sqrt_2_pi = std::sqrt(2.0 / M_PI);
    zero_integral = UA * UB * sqrt_2V2 * sqrt_2_pi;
}
else {
    // Eq. 3.14
    double T = V2 * R_AB2;
    zero_integral = UA * UB * std::sqrt(1.0 / R_AB2) * std::erf(std::sqrt(T));
}
         


                    double gamma_term = alpha_ck * alpha_ck_prime * beta_cl * beta_cl_prime * norm_k * norm_kp * norm_l * norm_lp * zero_integral;



gamma += gamma_term;

                }

            }
                
            }
        }
    }

    return gamma ; 
}


/**
 * @brief Builds the symmetric matrix of γ(A,B) values between atoms in the molecule.
 *
 * This function selects the first s-type orbital on each atom (if available) and uses
 * it to compute the gamma integral with every other atom's s-type orbital.
 *
 * The resulting matrix is converted to electron-volts (eV).
 *
 * @param atoms              List of Atom structs (position and atomic number).
 * @param basis              Molecular basis functions (contracted Gaussians).
 * @param atom_for_orbital   Mapping from orbital index to atom index.
 * @return arma::mat         Symmetric gamma matrix in eV units.
 */
arma::mat compute_gamma_matrix(const std::vector<Atom>& atoms, 
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

// Now build gamma matrix using these s-orbitals
for (size_t A = 0; A < num_atoms; ++A) {
for (size_t B = 0; B <= A; ++B) {
int mu = s_orbital_index[A];
int nu = s_orbital_index[B];
if (mu == -1 || nu == -1) {
std::cerr << "Missing s-orbital for atom " << A << " or " << B << std::endl;
continue;
}

double gamma_val = compute_gamma(basis[mu], basis[nu]);
gamma_matrix(A, B) = gamma_matrix(B, A) = gamma_val * 27.2114;  // Hartree to eV
}
}

return gamma_matrix;
}







/**
 * @brief Computes the core Hamiltonian matrix H_core using the CNDO/2 method.
 *
 * The core Hamiltonian includes one-electron terms: orbital energy, electron-nuclear attraction,
 * and averaged repulsion from all other nuclei. Diagonal elements are built from:
 *
 *  - Orbital ionization potential (from CNDO parameters)
 *  - Coulomb attraction to its own nucleus (gamma_AA term)
 *  - Inter-nuclear Coulomb repulsion from other atoms (gamma_AB terms)
 *
 * Off-diagonal terms are approximated using the overlap and average beta parameters
 * 
 * @param basis               List of contracted Gaussian basis functions.
 * @param atom_for_orbital    Mapping from orbital index to atom index.
 * @param atoms               List of atoms in the molecule.
 * @param gamma_matrix        Atom-based Coulomb interaction matrix in eV.
 * @return arma::mat          Core Hamiltonian matrix H_core in eV.
 */


arma::mat compute_core_hamiltonian(
    const std::vector<Gaussian>& basis,
    const std::vector<int>& atom_for_orbital,
    const std::vector<Atom>& atoms,
    const arma::mat& gamma_matrix)  // Now atom-based
{
    size_t N = basis.size();
    arma::mat H_core(N, N, arma::fill::zeros);


      std::vector<int> atomic_numbers;
      for (const auto& atom : atoms) {
          atomic_numbers.push_back(atom.atomic_number);
      }
      std::vector<CNDOParameters> atom_params = get_all_CNDO_parameters(atomic_numbers);

      

    for (size_t mu = 0; mu < N; ++mu) {
        int A = atom_for_orbital[mu];
        const CNDOParameters& params_A = atom_params[A];
        double Z_A = params_A.Z;

        
        for (size_t nu = 0; nu < N; ++nu) {
            int B = atom_for_orbital[nu];
            const CNDOParameters& params_B = atom_params[B];

            if (mu == nu) {
                // Diagonal elements
                double gamma_AA = gamma_matrix(A, A);  // Atom-based
                double sum_ZB_gamma_AB = 0.0;
                
                for (size_t B_idx = 0; B_idx < atoms.size(); ++B_idx) {
                    if (B_idx == A) continue;
                    double gamma_AB = gamma_matrix(A, B_idx);  // Atom-based
                    sum_ZB_gamma_AB += atom_params[B_idx].Z * gamma_AB;
                }
                double first_term = calculate_first_term(mu, basis, params_A);


                double second_term = -(params_A.Z - 0.5) * gamma_AA;
                double third_term = -sum_ZB_gamma_AB;
                
                double h_diag = first_term + second_term + third_term;

                // H_core(mu, mu) = first_term + second_term + third_term;
                H_core(mu, mu) = h_diag;

                
                // debug_hcore_diagonal_term(mu, A, params_A, gamma_AA, Z_A, first_term, second_term, third_term, h_diag);


            } else {
                // Off-diagonal elements
                double S_mu_nu = S_AB_Summation(basis[mu], basis[nu]);
                H_core(mu, nu) = 0.5 * (params_A.beta + params_B.beta) * S_mu_nu;
            }
        }
    }
    return H_core;
}




/**
 * @brief Computes the classical nuclear-nuclear repulsion energy.
 *
 * Uses valence-only CNDO/2 nuclear charges
 * The distances R_AB are computed in Bohr, and result is converted to eV.
 *
 * @param atoms       List of atoms with positions and atomic numbers.
 * @return double     Total nuclear repulsion energy in eV.
 */

double nuclear_repulsion(const std::vector<Atom>& atoms) {
    double energy = 0.0;

    // Retrieve valence Z values for all atoms (CNDO/2)
    std::vector<CNDOParameters> params;
    for (const auto& atom : atoms)
        params.push_back(CNDO_formula_parameters(atom.atomic_number));

    // Loop over unique atom pairs (B < A to avoid double-counting)
    for (size_t A = 0; A < atoms.size(); ++A) {
        for (size_t B = 0; B < A; ++B) {
            arma::vec RA = {atoms[A].x, atoms[A].y, atoms[A].z};
            arma::vec RB = {atoms[B].x, atoms[B].y, atoms[B].z};
            double R_AB = arma::norm(RA - RB);  // Already in Bohr

            double ZA = params[A].Z;  // Valence Z
            double ZB = params[B].Z;

            // CNDO/2 nuclear repulsion in eV
            energy += (ZA * ZB / R_AB) * 27.2114;
        }
    }

    return energy;
}




/**
 * @brief Computes the electronic energy contribution from alpha electrons.
 *
 * Uses the Roothaan expression
 *
 * @param P_alpha     Alpha-spin density matrix.
 * @param H           Core Hamiltonian matrix.
 * @param F_alpha     Alpha-spin Fock matrix.
 * @return double     Alpha electronic energy in eV.
 */
//e-alpha 
double calculate_alpha_energy(const arma::mat& P_alpha, const arma::mat& H, const arma::mat& F_alpha) {
    return 0.5 * arma::accu(P_alpha % (H + F_alpha)); // % = element-wise multiplication
}




/**
 * @brief Computes the electronic energy contribution from beta electrons.
 *
 * Uses the Roothaan expression
 *
 * @param P_beta      Beta-spin density matrix.
 * @param H           Core Hamiltonian matrix.
 * @param F_beta      Beta-spin Fock matrix.
 * @return double     Beta electronic energy in eV.
 */
//e-beta
double calculate_beta_energy(const arma::mat& P_beta, const arma::mat& H, const arma::mat& F_beta) {
    return 0.5 * arma::accu(P_beta % (H + F_beta));
}




/**
 * @brief Computes the total electronic energy of the system including nuclear repulsion.
 *
 * Total energy is:
 * E_total = E_alpha + E_beta + E_nuc 
 * where:
 *  - E_alpha is from alpha-spin electrons,
 *  - E_beta  is from beta-spin electrons,
 *  - E_nuc   is classical nuclear repulsion.
 *
 * @param P_alpha     Alpha-spin density matrix.
 * @param P_beta      Beta-spin density matrix.
 * @param H           Core Hamiltonian matrix.
 * @param F_alpha     Alpha-spin Fock matrix.
 * @param F_beta      Beta-spin Fock matrix.
 * @param atoms       Atomic information (used for nuclear repulsion).
 * @return double     Total energy of the system in eV.
 */

double calculate_total_energy(
    const arma::mat& P_alpha, const arma::mat& P_beta,
    const arma::mat& H,
    const arma::mat& F_alpha, const arma::mat& F_beta,
    const std::vector<Atom>& atoms
) {
    double E_alpha = calculate_alpha_energy(P_alpha, H, F_alpha);
    double E_beta  = calculate_beta_energy(P_beta, H, F_beta);
    double E_nuc   = nuclear_repulsion(atoms);


//     std::cout << "[ENERGY DEBUG] E_alpha = " << calculate_alpha_energy(P_alpha, H, F_alpha) << "\n";
// std::cout << "[ENERGY DEBUG] E_beta  = " << calculate_beta_energy(P_beta, H, F_beta) << "\n";
// std::cout << "[ENERGY DEBUG] E_nuc   = " << nuclear_repulsion(atoms) << "\n";

    return E_alpha + E_beta + E_nuc; 
}
