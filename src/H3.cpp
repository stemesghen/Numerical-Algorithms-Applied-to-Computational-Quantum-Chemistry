



#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <armadillo>

#include <algorithm>  


#include "header_file.hpp"



BasisInfo initialize_molecular_basis(const std::vector<Atom>& atoms) {
    BasisInfo result;

    for (size_t atom_index = 0; atom_index < atoms.size(); ++atom_index) {
        const Atom& atom = atoms[atom_index]; 

        std::vector<std::vector<int>> angular_momentum = {
            {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}
        };
        std::vector<double> exp, s_coef, p_coef;

        if (atom.atomic_number == 6) {  // Carbon
            exp     = {2.94124940, 0.68348310, 0.22228990};
            s_coef  = {-0.09996723, 0.39951283, 0.70011547};
            p_coef  = {0.15591627, 0.60768372, 0.39195739};
        } 
        else if (atom.atomic_number == 7) {  // Nitrogen
            exp    = {3.78045590, 0.87849660, 0.28571440};
            s_coef = {-0.09996723, 0.39951283, 0.70011547};
            p_coef = {0.15591627, 0.60768372, 0.39195739};
        } 
        else if (atom.atomic_number == 8) {  // Oxygen
            exp    = {5.03315130, 1.16959610, 0.38038900};
            s_coef = {-0.09996723, 0.39951283, 0.70011547};
            p_coef = {0.15591627, 0.60768372, 0.39195739};
        } 
        else if (atom.atomic_number == 9) {  // Fluorine
            exp    = {6.46480320, 1.50228120, 0.48858850};
            s_coef = {-0.09996723, 0.39951283, 0.70011547};
            p_coef = {0.15591627, 0.60768372, 0.39195739};
        } 
        else if (atom.atomic_number == 1) {  // Hydrogen — separate!
            std::vector<double> H_exp  = {3.42525091, 0.62391373, 0.16885540};
            std::vector<double> H_coef = {0.15432897, 0.53532814, 0.44463454};

            Gaussian hydrogen;
            hydrogen.R = {atom.x, atom.y, atom.z};
            hydrogen.L = {0, 0, 0};
            hydrogen.exponent = H_exp;
            hydrogen.contraction_coefficient = H_coef;
            hydrogen.orbital_energy = -13.6;

            hydrogen.normalization_constants.resize(H_exp.size());
            for (size_t i = 0; i < H_exp.size(); ++i)
                hydrogen.normalization_constants[i] = normalization_constant(hydrogen, i);

            result.molecular_basis.push_back(hydrogen);
            result.atom_for_orbital.push_back(atom_index);
            continue;  // Skip the rest of the loop
        } 

        // Shared code for C, N, O, F
        for (int i = 0; i < 4; ++i) {
            Gaussian g;
            g.R = {atom.x, atom.y, atom.z};
            g.L = arma::Col<int>(angular_momentum[i]);
            g.exponent = exp;
            g.contraction_coefficient = (i == 0) ? s_coef : p_coef;
            g.orbital_energy = (i == 0) ? -21.4 : -11.4;

            g.normalization_constants.resize(exp.size());
            for (size_t k = 0; k < exp.size(); ++k) {
                g.normalization_constants[k] = normalization_constant(g, k);
            }

            result.molecular_basis.push_back(g);
            result.atom_for_orbital.push_back(atom_index);
        }
    }

    return result;
}



// BasisInfo initialize_molecular_basis(const std::vector<Atom>& atoms) {
//     BasisInfo result;

//     for (size_t atom_index = 0; atom_index < atoms.size(); ++atom_index) {
//         const Atom& atom = atoms[atom_index]; 

//         std::vector<std::vector<int>> angular_momentum = {
//             {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}
//         };
//         std::vector<double> exp, s_coef, p_coef;

//         if (atom.atomic_number == 6) {  // Carbon
//             exp     = {2.94124940, 0.68348310, 0.22228990};
//             s_coef  = {-0.09996723, 0.39951283, 0.70011547};
//             p_coef  = {0.15591627, 0.60768372, 0.39195739};
//         } 
//         else if (atom.atomic_number == 7) {  // Nitrogen
//             exp    = {3.78045590, 0.87849660, 0.28571440};
//             s_coef = {-0.09996723, 0.39951283, 0.70011547};
//             p_coef = {0.15591627, 0.60768372, 0.39195739};
//         } 
//         else if (atom.atomic_number == 8) {  // Oxygen
//             exp    = {5.03315130, 1.16959610, 0.38038900};
//             s_coef = {-0.09996723, 0.39951283, 0.70011547};
//             p_coef = {0.15591627, 0.60768372, 0.39195739};
//         } 
//         else if (atom.atomic_number == 9) {  // Fluorine
//             exp    = {6.46480320, 1.50228120, 0.48858850};
//             s_coef = {-0.09996723, 0.39951283, 0.70011547};
//             p_coef = {0.15591627, 0.60768372, 0.39195739};
//         } 
//         else if (atom.atomic_number == 1) {  // Hydrogen — separate!
//             std::vector<double> H_exp  = {3.42525091, 0.62391373, 0.16885540};
//             std::vector<double> H_coef = {0.15432897, 0.53532814, 0.44463454};

//             Gaussian hydrogen;
//             hydrogen.R = {atom.x, atom.y, atom.z};
//             hydrogen.L = {0, 0, 0};
//             hydrogen.exponent = H_exp;
//             hydrogen.contraction_coefficient = H_coef;
//             hydrogen.orbital_energy = -13.6;

//             hydrogen.atom_id = atom_index;
//             hydrogen.orbital_type = "s";

//             hydrogen.normalization_constants.resize(H_exp.size());
//             for (size_t i = 0; i < H_exp.size(); ++i)
//                 hydrogen.normalization_constants[i] = normalization_constant(hydrogen, i);

//             result.molecular_basis.push_back(hydrogen);
//             result.atom_for_orbital.push_back(atom_index);
//             continue;
//         } 

//         // Shared code for C, N, O, F
//         for (int i = 0; i < 4; ++i) {
//             Gaussian g;
//             g.R = {atom.x, atom.y, atom.z};
//             g.L = arma::Col<int>(angular_momentum[i]);
//             g.exponent = exp;
//             g.contraction_coefficient = (i == 0) ? s_coef : p_coef;
//             g.orbital_energy = (i == 0) ? -21.4 : -11.4;

//             g.atom_id = atom_index;
//             g.orbital_type = (i == 0) ? "s" :
//                              (i == 1) ? "px" :
//                              (i == 2) ? "py" : "pz";

//             g.normalization_constants.resize(exp.size());
//             for (size_t k = 0; k < exp.size(); ++k)
//                 g.normalization_constants[k] = normalization_constant(g, k);

//             result.molecular_basis.push_back(g);
//             result.atom_for_orbital.push_back(atom_index);
//         }
//     }

//     return result;
// }



// BasisInfo initialize_molecular_basis(const std::vector<Atom>& atoms) {
//     BasisInfo result;

//     //Tracks which atomic orbital belongs to which atom
//     for (size_t atom_index = 0; atom_index < atoms.size(); ++atom_index) {
//         const Atom& atom = atoms[atom_index]; 
//         {
//         if (atom.atomic_number == 6) {  // Carbon
//             std::vector<std::vector<int>> carbon_angular_momentum = {
//                 {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}
//             };
//             std::vector<double> C_exp     = {2.94124940, 0.68348310, 0.22228990};
//             std::vector<double> C_s_coef  = {-0.09996723, 0.39951283, 0.70011547};
//             std::vector<double> C_p_coef  = {0.15591627, 0.60768372, 0.39195739};

//             for (int i = 0; i < 4; i++) {
//                 Gaussian carbon;
//                 carbon.R = {atom.x, atom.y, atom.z};
//                 carbon.L = arma::Col<int>(carbon_angular_momentum[i]);
//                 carbon.exponent = C_exp;
//                 carbon.contraction_coefficient = (i == 0) ? C_s_coef : C_p_coef;
//                 carbon.orbital_energy = (i == 0) ? -21.4 : -11.4;

//                 carbon.normalization_constants.resize(carbon.exponent.size());
//                 for (size_t k = 0; k < carbon.exponent.size(); ++k) {
//                     carbon.normalization_constants[k] = normalization_constant(carbon, k);
//                 }

//                 result.molecular_basis.push_back(carbon);
//                 result.atom_for_orbital.push_back(atom_index);

            
//             }
//         } 
        
//         else if (atom.atomic_number == 1) {  // Hydrogen
//             std::vector<double> H_exp = {3.42525091, 0.62391373, 0.16885540};
//             std::vector<double> H_coef = {0.15432897, 0.53532814, 0.44463454};

//             Gaussian hydrogen;
//             hydrogen.R = {atom.x, atom.y, atom.z};
//             hydrogen.L = {0, 0, 0};
//             hydrogen.exponent = H_exp;
//             hydrogen.contraction_coefficient = H_coef;
//             hydrogen.orbital_energy = -13.6;

//             hydrogen.normalization_constants.resize(hydrogen.exponent.size());
//             for (size_t i = 0; i < hydrogen.exponent.size(); ++i) {
//                 hydrogen.normalization_constants[i] = normalization_constant(hydrogen, i);
//             }

//             result.molecular_basis.push_back(hydrogen);
//             result.atom_for_orbital.push_back(atom_index);
//         }

//         std::vector<std::vector<int>> angular_momentum = {
//             {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}
//         };
//         std::vector<double> exp, s_coef, p_coef;

//         else if (atom.atomic_number == 7 || atom.atomic_number == 8 || atom.atomic_number == 9) {


//             if (atom.atomic_number == 7) { // Nitrogen
//                 exp    = {3.78045590, 0.87849660, 0.28571440};
//                 s_coef = {-0.09996723, 0.39951283, 0.70011547};
//                 p_coef = {0.15591627, 0.60768372, 0.39195739};
//             } else if (atom.atomic_number == 8) { // Oxygen
//                 exp    = {5.03315130, 1.16959610, 0.38038900};
//                 s_coef = {-0.09996723, 0.39951283, 0.70011547};
//                 p_coef = {0.15591627, 0.60768372, 0.39195739};
//             } else if (atom.atomic_number == 9) { // Fluorine
//                 exp    = {6.46480320, 1.50228120, 0.48858850};
//                 s_coef = {-0.09996723, 0.39951283, 0.70011547};
//                 p_coef = {0.15591627, 0.60768372, 0.39195739};
//             }
//         }

//             for (int i = 0; i < 4; ++i) {
//                 Gaussian g;
//                 g.R = {atom.x, atom.y, atom.z};
//                 g.L = arma::Col<int>(angular_momentum[i]);
//                 g.exponent = exp;
//                 g.contraction_coefficient = (i == 0) ? s_coef : p_coef;
//                 g.orbital_energy = (i == 0) ? -21.4 : -11.4;

//                 g.normalization_constants.resize(g.exponent.size());
//                 for (size_t k = 0; k < g.exponent.size(); ++k) {
//                     g.normalization_constants[k] = normalization_constant(g, k);
//                 }

//                 result.molecular_basis.push_back(g);
//                 result.atom_for_orbital.push_back(atom_index);
                
//             }
//         }
//     }

//     return result;
// }








//  // Converts atomic coordinates from Angstroms to Bohr

// std::vector<Atom> compute_bohr(const std::vector<Atom>& atoms) {
//     const double angstrom_to_bohr = 1.0 / 0.52917706;
//     std::vector<Atom> converted;
//     for (const auto& atom : atoms) {
//         converted.push_back({atom.atomic_number, atom.x * angstrom_to_bohr, atom.y * angstrom_to_bohr, atom.z * angstrom_to_bohr});
//     }
//     return converted;
// }



double compute_squared_distance(const arma::vec& A, const arma::vec& B) {
    return arma::dot(A - B, A - B);  // more efficient and correct with arma
}



// compute the Gaussian product center using Armadillo
// Formula: P = (alpha R_A + alpha R_B) / (alpha + β)
// arma::vec compute_gaussian_product_center(const Gaussian &A, const Gaussian &B, size_t k, size_t l) {
//     double alpha = A.exponent[k];  // Primitive k of A
//     double beta  = B.exponent[l];  // Primitive l of B

//     const arma::vec& R_A = A.R;
//     const arma::vec& R_B = B.R;
//     arma::vec Rp = (alpha * R_A + beta * R_B) / (alpha + beta);
//     return Rp;
// }

// int factorial(int n) {
//     return (n <= 1) ? 1 : n * factorial(n - 1);
// }

// int double_factorial(int n) {
//     if (n <= 0) return 1; //set neg numbers to be factorial 1
//     int result = 1;
//     for (int i = n; i > 0; i -= 2) result *= i;
//     return result;
// }

// // binomial coefficient (n choose k)
// double binomial_coefficient(int n, int k) {
//     if (k > n) return 0;
//     return factorial(n) / (factorial(k) * factorial(n - k));
// }




// double double_summation(const Gaussian& a, const Gaussian& b, int dim, int k, int l) {
//     int la = a.L[dim];
//     int lb = b.L[dim];
//     double alpha = a.exponent[k];
//     double beta = b.exponent[l];
//     double Ax = a.R[dim];
//     double Bx = b.R[dim];

//     double p = alpha + beta;
//     double Px = (alpha * Ax + beta * Bx) / p;

//     double sum = 0.0;

//     for (int i = 0; i <= la; ++i) {
//         for (int j = 0; j <= lb; ++j) {
//             if ((i + j) % 2 != 0) continue;  // only even i+j terms contribute
//             double term = binomial_coefficient(la, i) *
//                           binomial_coefficient(lb, j) *
//                           double_factorial(i + j - 1) *
//                           pow(Px - Ax, la - i) *
//                           pow(Px - Bx, lb - j) /
//                           pow(2 * p, (i + j) / 2);
//             sum += term;
//         }
//     }

//     return sum;
// }

// Computes the exponential term used in Gaussian integrals
//
// Parameters:
//  - a: Gaussian function A
//  - b: Gaussian function B
//  - index: The coordinate index (0 = x, 1 = y, 2 = z)
//  - k: Primitive index for Gaussian A
//  - l: Primitive index for Gaussian B
//
// Returns:
//  - The computed exponential factor exp[-(α_k * β_l * (r_A - r_B)^2) / (α_k + β_l)].


// double exponent(const Gaussian& a, const Gaussian& b, int index, int k, int l) {
//     double diff = a.R[index] - b.R[index];
//     double num = -a.exponent[k] * b.exponent[l] * pow(diff, 2);
//     double denom = a.exponent[k] + b.exponent[l];
//     return exp(num / denom);
// }

// Computes a polynomial expansion using the binomial theorem 
// to generate polynomial terms for the Gaussian basis function expansion.
// 
// Parameters:
//  - A_coordinate: The x, y, or z coordinate of center A
//  - B_coordinate: The x, y, or z coordinate of center B
//  - P_center: The computed Gaussian product center coordinate
//  - l_A: Angular momentum quantum number for A
//  - l_B: Angular momentum quantum number for B
// 
// Returns:
//  - A vector of computed polynomial terms.


// double compute_exponential_prefactor(const Gaussian &A, const Gaussian &B, size_t k, size_t l) {
//     double alpha = A.exponent[k];
//     double beta = B.exponent[l];
//     arma::vec R_A(A.R);
//     arma::vec R_B(B.R);

//     double R_AB_squared = compute_squared_distance(R_A, R_B);
    
//     double prefactor = std::exp(-(alpha * beta / (alpha + beta)) * R_AB_squared);
//     prefactor *= std::pow(M_PI / (alpha + beta), 1.5);

//     return prefactor;
// }



// Counts the number of atoms of a given atomic number in a molecule.
//
// Parameters:
//  - atoms: A vector of Atom structures representing the molecule.
//  - atomic_number: The atomic number of the element to count.
//
// Returns:
//  - The count of atoms with the specified atomic number.

int count_atoms(const std::vector<Atom>& atoms, int atomic_number) {
    int count = 0;
    for (const auto& atom : atoms) {
        if (atom.atomic_number == atomic_number) {
            count++;
        }
    }
    return count;
}
// Computes the total number of basis functions for a given hydrocarbon molecule.
//
// Parameters:
//  - num_carbon: Number of carbon atoms
//  - num_hydrogen: Number of hydrogen atoms
//
// Returns:
//  - The total number of basis functions.

int N_basis_function(int num_carbon, int num_hydrogen) {
    return (4 * num_carbon) + num_hydrogen;
}
// Computes the total number of valence electrons in a hydrocarbon molecule.
//
// Parameters:
//  - num_carbon: Number of carbon atoms
//  - num_hydrogen: Number of hydrogen atoms
//
// Returns:
//  - The total number of valence electrons.

int compute_number_electrons(int num_carbon, int num_hydrogen) {
    int electrons = 4 * num_carbon + num_hydrogen;
    return electrons;
}

// Compute squared distance between two 3D points

// double compute_squared_distance(const std::vector<double>& A, const std::vector<double>& B) {
//     return (A[0] - B[0]) * (A[0] - B[0]) +
//            (A[1] - B[1]) * (A[1] - B[1]) +
//            (A[2] - B[2]) * (A[2] - B[2]);
// }



// Computes the normalization constant for a Gaussian basis function.
//
// Parameters:
//  - g: The Gaussian function to normalize
//  - idx: The index of the primitive Gaussian within the contraction
//
// Returns:
//  - The computed normalization constant.

// double compute_primitive_overlap_integral(const Gaussian& A, const Gaussian& B, int dim, size_t k, size_t l) {
//     double alpha = A.exponent[k];
//     double beta = B.exponent[l];

//     double Ax = A.R[dim];
//     double Bx = B.R[dim];

//     double exp_factor = exp(-(alpha * beta / (alpha + beta)) * pow(Ax - Bx, 2));
//     double prefactor = sqrt(M_PI / (alpha + beta));
//     double poly_sum = double_summation(A, B, dim, k, l);

//     return exp_factor * prefactor * poly_sum;
// }





// Computes the contracted overlap integral between two Gaussian functions.
//
// Parameters:
//  - A: The first Gaussian function
//  - B: The second Gaussian function
//
// Returns:
//  - The computed contracted overlap integral.

// double compute_contracted_overlap_integral(const Gaussian& A, const Gaussian& B) {
//     double total_overlap = 0.0;
//     for (size_t k = 0; k < A.exponent.size(); ++k) {
//         for (size_t l = 0; l < B.exponent.size(); ++l) {
//             double Sx = compute_primitive_overlap_integral(A, B, 0, k, l);
//             double Sy = compute_primitive_overlap_integral(A, B, 1, k, l);
//             double Sz = compute_primitive_overlap_integral(A, B, 2, k, l);

//             double Nk = normalization_constant(A, k);
//             double Nl = normalization_constant(B, l);

//             total_overlap += A.contraction_coefficient[k] * B.contraction_coefficient[l] * Nk * Nl * (Sx * Sy * Sz);
//         }
//     }
//     return total_overlap;
// }

// Computes the Hamiltonian matrix from the molecular basis and overlap matrix
// Computes the normalization constant for a given Gaussian basis function.
//
// The normalization constant ensures that the Gaussian function is properly normalized
// so that its self-overlap integral equals 1.
//
// Parameters:
//  - g: The Gaussian function whose normalization constant is being computed.
//  - idx: The index of the primitive Gaussian in the contracted function.
//
// Returns:
//  - The computed normalization constant.

// double normalization_constant(const Gaussian& g, size_t idx) {
//     double alpha = g.exponent[idx]; // Gaussian exponent α_k
//     int lx = g.L[0]; // Angular momentum quantum number along x
//     int ly = g.L[1]; // Angular momentum quantum number along y
//     int lz = g.L[2]; // Angular momentum quantum number along z

//     // Prefactor derived from the normalization formula
//     double prefactor = pow(2 * alpha / M_PI, 0.75);

//     // Compute the normalization factor using factorial properties
//     double normalization_factor = sqrt(
//         pow(4 * alpha, lx + ly + lz) /
//         (factorial(2 * lx - 1) * factorial(2 * ly - 1) * factorial(2 * lz - 1))
//     );

//     return prefactor * normalization_factor;
// }

double normalization_constant(const Gaussian& g, size_t idx) {
    double alpha = g.exponent[idx];
    int lx = g.L[0];
    int ly = g.L[1];
    int lz = g.L[2];

    int l_total = lx + ly + lz;

    double norm = std::pow(2.0, 2 * l_total + 1.5) *
                  std::pow(alpha, l_total + 1.5) /
                  (double_factorial(2 * lx - 1) *
                   double_factorial(2 * ly - 1) *
                   double_factorial(2 * lz - 1) *
                   std::pow(M_PI, 1.5));

                   
    return std::sqrt(norm);
}



// Initializes the molecular basis set by constructing Gaussian functions for each atom.
//
// This function creates contracted Gaussian-type orbitals (GTOs) for Carbon and Hydrogen atoms
// using the STO-3G basis set. The basis set defines the exponents, contraction coefficients, 
// and angular momentum values for each atomic orbital.
//
// Parameters:
//  - atoms: A vector of Atom structures representing the molecule.
//
// Returns:

//  - A vector of Gaussian basis functions that define the molecular orbitals.



// std::vector<Gaussian> initialize_molecular_basis(const std::vector<Atom>& atoms) {
//     std::vector<Gaussian> molecular_basis; // Holds the Gaussian functions for the molecule

//     // Define the angular momentum configurations for Carbon (s and p orbitals)
//     std::vector<std::vector<int>> carbon_angular_momentum = {
//         {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1} // 1s, 2px, 2py, 2pz orbitals
//     };

//     // STO-3G basis set parameters for Carbon
//     std::vector<double> C_exp = {2.94124940, 0.68348310, 0.22228990}; // Primitive exponents
//     std::vector<double> C_s_coef = {-0.09996723, 0.39951283, 0.70011547}; // Contraction coefficients for 2s
//     std::vector<double> C_p_coef = {0.15591627, 0.60768372, 0.39195739}; // Contraction coefficients for 2p

//     // STO-3G basis set parameters for Hydrogen
//     std::vector<double> H_exp = {3.42525091, 0.62391373, 0.16885540}; // Primitive exponents
//     std::vector<double> H_coef = {0.15432897, 0.53532814, 0.44463454}; // Contraction coefficients
//     std::vector<int> hydrogen_angular_momentum = {0, 0, 0}; // Only 1s orbital

//     for (const auto& atom : atoms) {
//         if (atom.atomic_number == 6) {  // Carbon atom
//             for (int i = 0; i < 4; i++) { // 1s, 2px, 2py, 2pz orbitals
//                 Gaussian carbon;
//                 carbon.R = {atom.x, atom.y, atom.z}; // Store atomic position
//                 carbon.L = arma::Col<int>(carbon_angular_momentum[i]);

//                 // carbon.L =  arma::ivec(carbon_angular_momentum[i]); // Assign angular momentum
//                 carbon.exponent = C_exp; // Assign exponents

//                 // Assign contraction coefficients based on the orbital type
//                 carbon.contraction_coefficient = (i == 0) ? C_s_coef : C_p_coef;

//                 // Assign empirical orbital energy (used in Hamiltonian matrix)
//                 carbon.orbital_energy = (i == 0) ? -21.4 : -11.4; // -21.4 eV for 2s, -11.4 eV for 2p

//                 // Compute and store normalization constants for each primitive
//                 carbon.normalization_constants.resize(carbon.exponent.size());
//                 for (size_t k = 0; k < carbon.exponent.size(); ++k) {
//                     carbon.normalization_constants[k] = normalization_constant(carbon, k);
//                 }

//                 molecular_basis.push_back(carbon);
//             }
//         } 
//         else if (atom.atomic_number == 1) {  // Hydrogen atom (only 1s orbital)
//             Gaussian hydrogen;
//             hydrogen.R = {atom.x, atom.y, atom.z}; // Store atomic position
//             hydrogen.L = {0, 0, 0}; // 1s orbital
//             hydrogen.exponent = H_exp; // Assign exponents
//             hydrogen.contraction_coefficient = H_coef; // Assign contraction coefficients
//             hydrogen.orbital_energy = -13.6; // Empirical energy for hydrogen 1s orbital

//             // Compute and store normalization constants for each primitive
//             hydrogen.normalization_constants.resize(hydrogen.exponent.size());
//             for (size_t i = 0; i < hydrogen.exponent.size(); ++i) {
//                 hydrogen.normalization_constants[i] = normalization_constant(hydrogen, i);
//             }

//             molecular_basis.push_back(hydrogen);
//         }
//     }
//     return molecular_basis;
// }




// Computes the transformation matrix X from the overlap matrix S.
//
// This transformation is used to convert the generalized eigenvalue problem 
// HC = SCE into a standard eigenvalue problem. The transformation matrix is 
// computed as: X = S^(-1/2), where S^(-1/2) is the inverse square root of S.
//
// Parameters:
//  - S: The overlap matrix (NxN) of the molecular basis set.
//
// Returns:
//  - An NxN Armadillo matrix representing the transformation matrix X.

arma::mat transformation_matrix(const arma::mat& S) {
    // Compute the eigenvalues and eigenvectors of S
    arma::cx_vec eigval;  // Complex eigenvalues (cx_vec)
    arma::cx_mat eigvec;  // Complex eigenvectors (cx_mat)
    arma::eig_gen(eigval, eigvec, S);

    // Extract real parts of eigenvalues and eigenvectors
    arma::vec real_eigval = arma::real(eigval);
    arma::mat real_eigvec = arma::real(eigvec);

    // Compute the inverse square root of eigenvalues
    arma::vec inv_sqrt_eigenvalues = 1.0 / arma::sqrt(real_eigval);

    // Construct the diagonal matrix S^(-1/2)
    arma::mat S_inv_sqrt = arma::diagmat(inv_sqrt_eigenvalues);

    // Compute the transformation matrix: X = eigvec * S^(-1/2) * eigvec^T
    return real_eigvec * S_inv_sqrt * real_eigvec.t();
}

// Constructs the Hamiltonian matrix for the molecular system.
//
// The extended Hückel method approximates the Hamiltonian as:
// Hμν = 1/2 * K * (Hμμ + Hνν) * Sμν
// where K is an empirical scaling factor (typically 1.75).
//
// Parameters:
//  - molecular_basis: A vector of Gaussian basis functions.
//  - S: The overlap matrix (NxN).
//
// Returns:
//  - An NxN Armadillo matrix representing the Hamiltonian.

// arma::mat hamiltonian(const std::vector<Gaussian>& molecular_basis, const arma::mat& S) {
//     size_t N = molecular_basis.size(); // Number of basis functions
//     arma::mat H(N, N, arma::fill::zeros); // Initialize Hamiltonian matrix with zeros
//     double K = 1.75; // Empirical scaling factor

//     // Set diagonal elements (orbital energies of basis functions)
//     for (size_t i = 0; i < N; ++i) {
//         H(i, i) = molecular_basis[i].orbital_energy;  // Orbital energies from basis set
//     }

//     // Compute off-diagonal elements using the overlap matrix
//     for (size_t i = 0; i < N; i++) {
//         for (size_t j = i + 1; j < N; j++) {
//             H(i, j) = 0.5 * K * (H(i, i) + H(j, j)) * S(i, j); // Off-diagonal elements
//             H(j, i) = H(i, j);  // Ensure symmetry (H is symmetric)
//         }
//     }
//     return H;
// }



// Solves the generalized eigenvalue problem HC = SCE.
//
// Converts the problem into a standard eigenvalue problem using the 
// transformation matrix X = S^(-1/2), then diagonalizes the transformed
// Hamiltonian to obtain molecular orbital energies and coefficients.
//
// Parameters:
//  - H: The Hamiltonian matrix (NxN).
//  - S: The overlap matrix (NxN).
//  - epsilon: Output vector of molecular orbital eigenvalues (sorted).
//  - C: Output matrix of molecular orbital coefficients.
//
// Returns:
//  - Fills `epsilon` with molecular orbital energies and `C` with coefficients.






/*

void solve_eigenproblem(const arma::mat& H, const arma::mat& S, arma::vec& epsilon, arma::mat& C) {
    arma::vec s_eigenvalues; // Eigenvalues of S
    arma::mat s_eigenvectors; // Eigenvectors of S
    arma::eig_sym(s_eigenvalues, s_eigenvectors, S); // Solve S = VΛV^T

    // Compute S^(-1/2) = VΛ^(-1/2)V^T
    arma::mat S_inv_sqrt = s_eigenvectors * arma::diagmat(1.0 / arma::sqrt(s_eigenvalues)) * s_eigenvectors.t();

    // Transform Hamiltonian: H' = X^T * H * X
    arma::mat H_orthogonalized = S_inv_sqrt.t() * H * S_inv_sqrt;

    // Solve standard eigenvalue problem: H'V = Vε
    arma::eig_sym(epsilon, C, H_orthogonalized);

    // Convert eigenvectors back to original basis: C = X * V
    C = S_inv_sqrt * C;
}
*/


// Computes the total electronic energy based on occupied molecular orbitals.
//
// The total energy is computed by summing the eigenvalues of the lowest occupied
// molecular orbitals, assuming a closed-shell system (2 electrons per orbital).
//
// Parameters:
//  - epsilon: Vector of molecular orbital energies (sorted).
//  - total_electrons: The total number of valence electrons in the molecule.
//
// Returns:
//  - The total electronic energy in Hartrees.

// double compute_total_energy(const arma::vec& epsilon, int total_electrons) {
//     int occupied_orbitals = total_electrons / 2; // Number of fully occupied orbitals
//     arma::vec sorted_epsilon = arma::sort(epsilon); // Sort eigenvalues in ascending order
//     double total_energy = 0.0;

//     // Sum the energies of occupied orbitals (multiply by 2 for electron pairing)
//     for (int i = 0; i < occupied_orbitals; i++) {
//         total_energy += 2.0 * sorted_epsilon[i];
//     }

//     return total_energy; // Return the total electronic energy
// }

// Sets up the molecular basis functions from the input atomic coordinates.
//
// This function initializes the molecular basis set using predefined basis functions
// for Carbon and Hydrogen atoms (STO-3G basis set).
//
// Parameters:
//  - atoms: A vector of Atom structures representing the molecule.
//
// Returns:
//  - A vector of Gaussian basis functions for the molecule.







double compute_SAA_self_overlap(const Gaussian &A, size_t k) {
    double alpha = A.exponent[k];
    arma::vec R_A = A.R;
    int l = A.L(0), m = A.L(1), n = A.L(2);

    return primitive_overlap(alpha, R_A, l, m, n,
                             alpha, R_A, l, m, n); // same primitive
}


// 1/sqrt SAA
void normalization_constant(Gaussian &A) {
    for (size_t k = 0; k < A.exponent.size(); ++k) {
        double S_AA = compute_SAA_self_overlap(A, k);
        A.normalization_constants[k] = 1.0 / std::sqrt(S_AA);
    }
}

// double normalization_constant(const Gaussian& g, size_t idx) {
//     double alpha = g.exponent[idx]; // Gaussian exponent α_k
//     int lx = g.L[0]; // Angular momentum quantum number along x
//     int ly = g.L[1]; // Angular momentum quantum number along y
//     int lz = g.L[2]; // Angular momentum quantum number along z

//     // Prefactor derived from the normalization formula
//     double prefactor = pow(2 * alpha / M_PI, 0.75);

//     // Compute the normalization factor using factorial properties
//     double normalization_factor = sqrt(
//         pow(4 * alpha, lx + ly + lz) /
//         (factorial(2 * lx - 1) * factorial(2 * ly - 1) * factorial(2 * lz - 1))
//     );

//     return prefactor * normalization_factor;
// }






// std::vector<Gaussian> setup_molecule(const std::vector<Atom>& atoms) {
//     return initialize_molecular_basis(atoms);
// }



std::vector<Gaussian> setup_molecule(const std::vector<Atom>& atoms) {
    BasisInfo info = initialize_molecular_basis(atoms);
    return info.molecular_basis;
}



arma::mat compute_overlap_matrix(const std::vector<Gaussian>& shell_A,
    const std::vector<Gaussian>& shell_B) {
arma::mat overlap_matrix(shell_B.size(), shell_A.size());

for (size_t i = 0; i < shell_B.size(); ++i) {
for (size_t j = 0; j < shell_A.size(); ++j) {
double val = S_AB_Summation(shell_A[j], shell_B[i]);
overlap_matrix(i, j) = val;
}
}

return overlap_matrix;
}

