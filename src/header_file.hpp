





#ifndef HEADER_FILE_HPP
#define HEADER_FILE_HPP

#include <vector>
#include <string>
#include <cmath>
#include <armadillo>





// struct Atom {
//     int atomic_number;
//     double x, y, z;
// };


struct Atom {
    std::string symbol;
    int atomic_number;
    double x, y, z;
    arma::vec R;
    int Z;
    
    Atom() : R(3) { // Initialize R as 3D vector
        R = {x, y, z};
    }
};


//Gaussian Struct 
struct Gaussian {
    arma::vec R;
    std::vector<double> exponent;
    arma::Col<int> L;
    arma::vec contraction_coefficient;
    arma::vec normalization_constants;
    double orbital_energy;

    Gaussian() = default; 

    Gaussian(const arma::vec& R_,
             const std::vector<double>& exponent_,
             const arma::Col<int>& L_,
             const arma::vec& coeff_,
             const arma::vec& norm_,
             double energy_)
        : R(R_), exponent(exponent_), L(L_),
          contraction_coefficient(coeff_), normalization_constants(norm_),
          orbital_energy(energy_) {}
};




struct BasisInfo {
    std::vector<Gaussian> molecular_basis;
    std::vector<int> atom_for_orbital; // Maps AO index to the atom index
};

// Molecular input structure
struct MoleculeInput {
    std::vector<Atom> atoms;
    int num_alpha_electrons;
    int num_beta_electrons;
};

struct CNDOParameters {
    double Is_As;  // 1/2 (I_s + A_s)
    double Ip_Ap;  // 1/2 (I_p + A_p)
    double beta;   // -β (already negative)
    int Z;       // Valence electrons
};



// MoleculeInput read_molecule_input(const std::string& filename);

std::vector<Atom> read_xyz_file(const std::string& xyz_file_path);
int get_atomic_number(const std::string& symbol);

// std::vector<Atom> compute_bohr(const std::vector<Atom>& atoms);
// std::vector<Atom> read_atoms_from_file(const std::string& filename);

// Basis Setup -
// In header_file.hpp
BasisInfo initialize_molecular_basis(const std::vector<Atom>& atoms);

// std::vector<Gaussian> initialize_molecular_basis(const std::vector<Atom>& atoms);
std::vector<Gaussian> setup_molecule(const std::vector<Atom>& atoms);
std::vector<Gaussian> generate_shell(int L, arma::vec R, std::vector<double> exponent);

//void compute_scf(const std::vector<Atom>& atoms, const std::vector<Gaussian>& molecular_basis);
// void compute_scf(const std::vector<Atom>& atoms, const std::vector<Gaussian>& molecular_basis)
// void compute_scf(const std::vector<Atom>& atoms, const BasisInfo& basis_info);

// arma::mat build_density_matrix(const arma::mat& C, int n_occ);
double compute_electron_energy(const arma::mat& P, const arma::mat& F);


// void compute_scf(const std::vector<Atom>& atoms, const BasisInfo& basis_info);


int factorial(int n);
int double_factorial(int n);
double binomial_coefficient(int n, int k);
double compute_squared_distance(const arma::vec& A, const arma::vec& B);

//Gaussian Integrals
double compute_exponential_prefactor(const Gaussian &A, const Gaussian &B, size_t k, size_t l);
arma::vec compute_gaussian_product_center(const Gaussian &A, const Gaussian &B, size_t k, size_t l);
std::vector<double> expand_polynomial(double A_coord, double B_coord, double P_center, int l_A, int l_B);

// Overlap Integrals
double primitive_overlap(
    double alpha, arma::vec R_A, int l_A, int m_A, int n_A,
    double beta,  arma::vec R_B, int l_B, int m_B, int n_B
);
double compute_SAA_self_overlap(const Gaussian &A, size_t k);

double normalization_constant(const Gaussian& g, size_t idx);

double S_AB_Summation(const Gaussian& A, const Gaussian& B);
arma::mat compute_overlap_matrix(const std::vector<Gaussian>& shell_A, const std::vector<Gaussian>& shell_B);


//SCF & Hamiltonian
arma::mat transformation_matrix(const arma::mat& S);
arma::mat hamiltonian(const std::vector<Gaussian>& molecular_basis, const arma::mat& S);
// void solve_eigenproblem(const arma::mat& H, const arma::mat& S, arma::vec& epsilon, arma::mat& C);
double compute_total_energy(const arma::vec& epsilon, int total_electrons);

int count_atoms(const std::vector<Atom>& atoms, int atomic_number);
int N_basis_function(int num_carbon, int num_hydrogen);
int compute_number_electrons(int num_carbon, int num_hydrogen);


// CNDO Parameters & Fock Matrix
// CNDOParameters CNDO_formula_parameters(int Z);
// std::vector<CNDOParameters> get_all_CNDO_parameters(const std::vector<int>& Zs);
// // arma::mat compute_fock_matrix(const arma::mat& P_spin, const arma::mat& P_total, const arma::mat& S,
// //                                const std::vector<Gaussian>& basis, const std::vector<int>& atom_for_orbital,
// //                                const std::vector<int>& atomic_numbers);
arma::mat compute_overlap_matrix(const std::vector<Gaussian>& basis1, const std::vector<Gaussian>& basis2);
arma::mat compute_gamma_matrix(const std::vector<Atom>& atoms, 
    const std::vector<Gaussian>& basis,
    const std::vector<int>& atom_for_orbital) ;
    
arma::mat compute_core_hamiltonian(const std::vector<Gaussian>& basis, 
                                  const std::vector<int>& atom_for_orbital,
                                  const std::vector<Atom>& atoms,
                                  const arma::mat& gamma_matrix);

// void solve_eigen_problem(const arma::mat& F, arma::mat& C, arma::vec& eps);
void solve_eigen_problem(
    const arma::mat& F,     // Fock matrix
    arma::mat& C,           // Output: MO coefficients
    arma::vec& eps          // Output: MO energies
);
CNDOParameters CNDO_formula_parameters(int atomic_number);
std::vector<CNDOParameters> get_all_CNDO_parameters(const std::vector<int>& atomic_numbers);
bool is_s_orbital(int orbital_index, const std::vector<Gaussian>& basis_functions);
std::vector<double> compute_p_AA(const arma::mat& P_total, 
                                const std::vector<int>& atom_for_orbital, 
                                int num_atoms);



arma::mat compute_P_alpha(const arma::mat& C_alpha, int p);
arma::mat compute_P_beta(const arma::mat& C_beta, int q);
// void compute_density_matrices(const arma::mat& C_alpha, const arma::mat& C_beta,
//                             int p, int q,
//                             arma::mat& P_alpha, arma::mat& P_beta, arma::mat& P_total);

                            void compute_density_matrices(
                                const arma::mat& C_alpha, const arma::mat& C_beta,
                                int p, int q,
                                arma::mat& P_alpha, arma::mat& P_beta, arma::mat& P_total
                            );
arma::mat build_fock_alpha(const arma::mat& P_alpha,
                          const arma::mat& P_total,
                          const arma::mat& gamma_matrix,
                          const arma::mat& S,
                          const std::vector<Gaussian>& basis,
                          const std::vector<int>& atom_for_orbital,
                          const std::vector<Atom>& atoms);
arma::mat build_fock_beta(const arma::mat& P_beta,
                         const arma::mat& P_total,
                         const arma::mat& gamma_matrix,
                         const arma::mat& S,
                         const std::vector<Gaussian>& basis,
                         const std::vector<int>& atom_for_orbital,
                         const std::vector<Atom>& atoms);
double nuclear_repulsion(const std::vector<Atom>& atoms);
double calculate_alpha_energy(const arma::mat& P_alpha, const arma::mat& H, const arma::mat& F_alpha);
double calculate_beta_energy(const arma::mat& P_beta, const arma::mat& H, const arma::mat& F_beta);
double calculate_total_energy(const arma::mat& P_alpha, const arma::mat& P_beta,
                            const arma::mat& H,
                            const arma::mat& F_alpha, const arma::mat& F_beta,
                            const std::vector<Atom>& atoms);

// Helper functions for integrals
double S_AB_Summation(const Gaussian& A, const Gaussian& B);
double compute_gamma(const Gaussian& A, const Gaussian& B);
double compute_V_gamma(double sigma_A, double sigma_B);
double compute_T_gamma(double sigma_A, double sigma_B, const arma::vec& R_A, const arma::vec& R_B);
double compute_boys_function(double T);

void debug_atom_positions_and_distances(const std::vector<Atom>& atoms);

void debug_cndo_parameters(const std::vector<Atom>& atoms) ;
void debug_gamma_matrix(const arma::mat& gamma_matrix) ;
void debug_p_AA(const std::vector<double>& p_AA, const std::vector<Atom>& atoms) ;
void debug_fock_diagonal(int mu, double fval, double t1, double t2, double t3) ;
void debug_matrix(const arma::mat& M, const std::string& name);
void debug_hcore_diagonal_term(
    int mu,
    int A,
    const CNDOParameters& params_A,
    double gamma_AA,
    double Z_A,
    double first_term,
    double second_term,
    double third_term,
    double h_diag
);


struct SCFResult {
    arma::mat P_alpha;
    arma::mat P_beta;
    arma::mat P_total;

    arma::mat C_alpha;
    arma::mat C_beta;

    arma::vec eps_alpha;
    arma::vec eps_beta;

    arma::mat F_alpha;
    arma::mat F_beta;

    arma::mat gamma_matrix;
    arma::mat S;

    std::vector<Gaussian> basis;
    std::vector<int> atom_for_orbital;
};
SCFResult run_scf(
    const std::vector<Atom>& atoms,
    int num_alpha_electrons,
    int num_beta_electrons
);

 arma::mat compute_yab(
    const arma::mat& P_alpha,
    const arma::mat& P_beta,
    const std::vector<int>& atom_for_orbital,
    const std::vector<Atom>& atoms
);



//HW 5

// ----- X and Y Matrix Terms -----
// arma::mat compute_x_matrix(
//     const arma::mat& P_alpha,
//     const arma::mat& P_beta,
//     const std::vector<Gaussian>& basis_functions
// );
arma::mat compute_x_matrix(
    const arma::mat& P_alpha,
    const arma::mat& P_beta,
    const std::vector<int>& atom_for_orbital,
    const std::vector<Atom>& atoms
) ;
arma::mat compute_y_matrix(
    const arma::mat& P_alpha,
    const arma::mat& P_beta,
    const std::vector<int>& atom_for_orbital,
    const std::vector<Atom>& atoms
);

double compute_yab(
    int A,
    int B,
    const arma::mat& P_alpha,
    const arma::mat& P_beta,
    const std::vector<int>& atom_for_orbital,
    const std::vector<Atom>& atoms,
    const std::vector<double>& Z 
) ;
// arma::mat compute_y_matrix(const arma::mat& P_alpha,
//     const arma::mat& P_beta,
//     const std::vector<int>& atom_for_orbital,
//     const std::vector<double>& Z);

arma::mat compute_y_matrix(
    const arma::mat& P_alpha,
    const arma::mat& P_beta,
    const std::vector<int>& atom_for_orbital,
    const std::vector<double>& Z
);



double compute_yab_mu_on_A(
    int A,
    int B,
    const arma::mat& P_alpha,
    const arma::mat& P_beta,
    const std::vector<int>& atom_for_orbital,
    const std::vector<Atom>& atoms
) ;

double accumulate_PBB_on_B_not_A(
    int A,
    int B,
    const arma::mat& P_total,
    const std::vector<int>& atom_for_orbital,
    const std::vector<Atom>& atoms
) ;
double compute_yab_nu_on_A(
    int A,
    int B,
    const arma::mat& P_alpha,
    const arma::mat& P_beta,
    const std::vector<int>& atom_for_orbital,
    const std::vector<Atom>& atoms
);

// Derivative of Overlap

arma::mat compute_Suv_RA(
    const std::vector<Gaussian>& basis_functions,
    const std::vector<int>& ao_atom_map,
    const std::vector<Atom>& atoms,
    int num_basis_functions
);

arma::vec derivative_primitive_overlap_RA( 
    double alpha, arma::vec R_A, arma::Col<int> L_A,
    double beta,  arma::vec R_B, arma::Col<int> L_B
);


double primitive_overlap(
    double alpha, arma::vec R_A, int l_A, int m_A, int n_A,
    double beta,  arma::vec R_B, int l_B, int m_B, int n_B
);


double primitive_1d(double alpha, double A, int l_A, double beta, double B, int l_B);
double derivative_1d(double alpha, double A, int l_A, double beta, double B, int l_B);


// ----- CNDO Gamma Matrix & Gradient -----
// double compute_deriv_gamma(const Gaussian& A, const Gaussian& B);


// You can also rename this to `compute_gamma_derivative(...)` if that's your function name
arma::vec  compute_deriv_gamma(const Gaussian& A, const Gaussian& B);
arma::mat compute_deriv_gamma_matrix(
    const std::vector<Atom>& atoms,
    const std::vector<Gaussian>& basis,
    const std::vector<int>& atom_for_orbital
);

// ----- Nuclear Gradient -----
arma::mat compute_nuclear_repulsion_gradient(const std::vector<Atom>& atoms);

// ----- CNDO Utility -----
double compute_V_gamma(double sigma_A, double sigma_B);
double calculate_first_term(int orbital_index, const std::vector<Gaussian>& basis, const CNDOParameters& params);





// arma::mat compute_electronic_gradient(
//     const arma::mat& x_matrix,              // size: N x N
//     const arma::mat& y_matrix,              // size: num_atoms x num_atoms
//     const arma::mat& Suv_RA,                // size: 3 x (N*N)
//     const arma::mat& gammaAB_RA,            // size: 3 x (num_atoms*num_atoms)
//     const std::vector<int>& atom_for_orbital,
//     int num_atoms,
//     int num_basis_functions
// );

arma::mat compute_electronic_gradient(
    const arma::mat& x_matrix,
    const arma::mat& y_matrix,
    const arma::mat& Suv_RA,
    const arma::mat& gammaAB_RA,
    const arma::mat& P_alpha,
    const arma::mat& P_beta,
    const std::vector<int>& atom_for_orbital,
    int num_atoms,
    int num_basis_functions
) ;

#endif // HEADER_FILE_HPP

