//
// Created by cheng on 23-1-24.
//
#include "mylib.h"
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <tuple>

#include "gromacs/fileio/xtcio.h"



Sfilter::Sfilter( std::string file_name_in){
    xtc_file_name = file_name_in ;
    const char open_mode[] {'r'};
    xdrFile = open_xtc(xtc_file_name.c_str(), open_mode);
}

std::string Sfilter::get_file_name(){
    return xtc_file_name;
}



real com_z(const std::vector<int> &index, rvec* x){
    real com_z = 0.0;
    for(auto i : index){
        com_z += (*(x+i))[2];
    }
    com_z /= index.size();
    return com_z ;
}

std::tuple<real,real> com_xy(const std::vector<int> &index, rvec* x, real &com_x, real &com_y){
    for(auto i : index){
        com_x += (*(x+i))[0];
        com_y += (*(x+i))[1];
    }
    com_x /= index.size();
    com_y /= index.size();
    return std::make_tuple(com_x,com_y) ;
}


int Sfilter::assign_state_double(const std::vector<int> &S01,
                                 const std::vector<int> &S23,
                                 const std::vector<int> &S45,
                                 const std::vector<int> &center,
                                 const std::vector<int> &ion_index,
                                 const std::vector<int> &wat_index,
                                 const float rad
                                 ){

    matrix     box;
    rvec*      x;
    real       prec;
    gmx_bool   bOK;
    int answer = read_first_xtc(xdrFile,&natoms,&step,&time,box,&x,&prec,&bOK);
    if ( not answer){
        return 1;
    }

    // check the input index is within proper range
    std::vector<std::vector<int>> index_pack {S01,S23,S45,center,ion_index,wat_index} ;
    for(auto j : index_pack){
        //std::cout << "Checking index, Number of atoms :"<< natoms << std::endl ;
        for(auto i : j){
            if(i > natoms-1) return 2 ;
        }
    }

    do
    {
        // compute the COM_z for each boundary
        real com_S01_z = com_z(S01, x);
        real com_S23_z = com_z(S23, x);
        real com_S45_z = com_z(S45, x);
        // compute the COM_xy for the center
        real com_center_x=0.0;
        real com_center_y=0.0;
        com_xy(center, x, com_center_x, com_center_y);
    // loop over ions and assign state(s)
    std::vector<short int> ion_state_frame_vec(ion_index.size(), 2) ;
    for(size_t i=0; i < ion_index.size(); i++){
        int at_index = ion_index.at(i);
        real atom_x = (*(x + at_index))[0] ;
        real atom_y = (*(x + at_index))[1] ;
        real atom_z = (*(x + at_index))[2] ;
        if (atom_z > com_S01_z)
            ion_state_frame_vec.at(i) = 3;
        else if (atom_z < com_S45_z)
            ion_state_frame_vec.at(i) = 4;
        else if (std::sqrt(std::pow(atom_x-com_center_x,2) + std::pow(atom_y-com_center_y,2)) < rad){
            if (atom_z > com_S23_z)
                ion_state_frame_vec.at(i) = 1;
            else
                ion_state_frame_vec.at(i) = 5;
        }
    }
    ion_state_vec.push_back(ion_state_frame_vec);

    // loop over water(O) and assign state(s)
    std::vector<short int> wat_state_frame_vec(wat_index.size(), 2) ;
    for(size_t i=0; i < wat_index.size(); i++){
        int at_index = wat_index.at(i);
        real atom_x = (*(x + at_index))[0] ;
        real atom_y = (*(x + at_index))[1] ;
        real atom_z = (*(x + at_index))[2] ;
        if (atom_z > com_S01_z)
            wat_state_frame_vec.at(i) = 3;
        else if (atom_z < com_S45_z)
            wat_state_frame_vec.at(i) = 4;
        else if (std::sqrt(std::pow(atom_x-com_center_x,2) + std::pow(atom_y-com_center_y,2)) < rad){
            if (atom_z > com_S23_z)
                wat_state_frame_vec.at(i) = 1;
            else
                wat_state_frame_vec.at(i) = 5;
        }
    }
    wat_state_vec.push_back(wat_state_frame_vec);
    } while (read_next_xtc(xdrFile, natoms, &step, &time, box, x, &prec, &bOK) != 0);


    return 0 ;

}


Sfilter::~Sfilter(){
    close_xtc( xdrFile );
    std::cout<< "File closed. Destructor executed" << std::endl ;
}