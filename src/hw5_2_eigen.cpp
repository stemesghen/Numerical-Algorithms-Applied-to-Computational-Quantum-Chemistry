#include <iostream> 
#include <fstream>
#include <filesystem>
#include <string> 
#include <cstdlib>
#include <stdexcept>

// RENAME THIS FILE TO hw5_2 IF YOU ARE GOING TO USE IT 

#include <nlohmann/json.hpp> // This is the JSON handling library
#include <highfive/highfive.hpp> // This is a hdf5 library for eigen
#include <highfive/eigen.hpp> // this import is required
#include <Eigen/Dense> 

// convenience definitions so the code is more readable
namespace fs = std::filesystem;
using json = nlohmann::json; 


int main(int argc, char** argv){
    // check that a config file is supplied
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

    
    int num_atoms = 1; // You will have to replace this with the number of atoms in the molecule 
    int num_basis_functions = 2; // you will have to replace this with the number of basis sets in the molecule
    int num_3D_dims = 3; 

    // Your answers go in these objects 

    Eigen::MatrixXd Suv_RA(num_3D_dims, num_basis_functions * num_basis_functions); 
    // Ideally, this would be (3, n_funcs, n_funcs) rank-3 tensor
    // but we're flattening (n-funcs, n-atoms) into a single dimension (n-funcs ^ 2)
    // this is because tensors are not supported in Eigen and I want students to be able to 
    // submit their work in a consistent format
    Eigen::MatrixXd gammaAB_RA(num_3D_dims, num_atoms * num_atoms); 
    // This is the same story, ideally, this would be (3, num_atoms, num_atoms) instead of (3, num_atoms ^ 2)
    Eigen::MatrixXd gradient_nuclear(num_3D_dims, num_atoms);
    Eigen::MatrixXd gradient_electronic(num_3D_dims, num_atoms); 
    Eigen::MatrixXd gradient(num_3D_dims, num_basis_functions); 

    // most of the code goes here 
   



    // You do not need to modify the code below this point 

    // Set print configs
    std::cout << std::fixed << std::setprecision(4) << std::setw(8) << std::right ; 

    // inspect your answer via printing
    std::cout << "Suv_RA" << std::endl;
    std::cout << Suv_RA << std::endl;
    std::cout << "gammaAB_RA" << std::endl; 
    std::cout << gammaAB_RA << std::endl;      
    std::cout << "gradient_nuclear" << std::endl;  
    std::cout << gradient_nuclear << std::endl;  
    std::cout << "gradient_electronic" << std::endl;  
    std::cout <<  gradient_electronic << std::endl;  
    std::cout << "gradient" << std::endl;     
    std::cout << gradient << std::endl; 


    // check that output dir exists
    if (!fs::exists(output_file_path.parent_path())){
        fs::create_directories(output_file_path.parent_path()); 
    }
    
    // delete the file if it does exist (so that no old answers stay there by accident)
    if (fs::exists(output_file_path)){
        fs::remove(output_file_path); 
    }

    // write results to file 
    HighFive::File outfile(output_file_path, HighFive::File::Create); 


    // These transposes are needed because this uses a different internal storage than armadillo (row major vs column major)
    outfile.createDataSet("Suv_RA", Eigen::MatrixXd(Suv_RA.transpose()));
    outfile.createDataSet("gammaAB_RA", Eigen::MatrixXd(gammaAB_RA.transpose()));
    outfile.createDataSet("gradient_nuclear", Eigen::MatrixXd(gradient_nuclear.transpose()));
    outfile.createDataSet("gradient_electronic", Eigen::MatrixXd(gradient_electronic.transpose()));
    outfile.createDataSet("gradient", Eigen::MatrixXd(gradient.transpose()));
    
}  