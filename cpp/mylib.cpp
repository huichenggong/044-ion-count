//
// Created by cheng on 23-1-24.
//
#include "mylib.h"
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "gromacs/fileio/xtcio.h"

namespace py = pybind11;


Sfilter::Sfilter( std::string file_name_in){
    xtc_file_name = file_name_in ;
    const char open_mode[] {'r'};
    xdrFile = open_xtc(xtc_file_name.c_str(), open_mode);
}

std::string Sfilter::get_file_name(){
    return xtc_file_name;
}

int Sfilter::distance(int index_a, int index_b) {
    //std::vector<real> distance_vec ;
    int64_t    step;
    real       time;
    matrix     box;
    rvec*      x;
    real       prec;
    gmx_bool   bOK;
    int answer = read_first_xtc(xdrFile,&natoms,&step,&time,box,&x,&prec,&bOK);
    if ( not answer){
        return 1;
    }
    if (index_a>natoms-1 or index_b>natoms-1){
        return 2;
    }
    do
    {
        real dist_tmp_x = ((*(x+index_a))[0] - (*(x+index_b))[0]);
        real dist_tmp_y = ((*(x+index_a))[1] - (*(x+index_b))[1]);
        real dist_tmp_z = ((*(x+index_a))[2] - (*(x+index_b))[2]);
        real dist_tmp = sqrt(
                dist_tmp_x*dist_tmp_x
                +dist_tmp_y*dist_tmp_y
                +dist_tmp_z*dist_tmp_z);
        distance_vec.push_back(dist_tmp);
    } while (read_next_xtc(xdrFile, natoms, &step, &time, box, x, &prec, &bOK) != 0);
    return 0 ;
}

Eigen::VectorXf Sfilter::get_distance(){
    //std::vector<real> * v = &distance_vec;
    //auto capsule = py::capsule(v, [](void *v) { delete reinterpret_cast<std::vector<real>*>(v); });
    Eigen::Vector3f v2(distance_vec.data());
    return v2 ;
}

int Sfilter::count(){
    distance_vec.at(0) += 100 ;
    return 0 ;
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
        real com_S01_z = 0.0;
        real com_S23_z = 0.0;
        real com_S45_z = 0.0;
        {
            int count = 0;
            for(auto i : S01){
                com_S01_z += (*(x + i))[2] ;
                count += 1;
            }
            com_S01_z /= count ;

            count = 0;
            for(auto i : S23){
                com_S23_z += (*(x + i))[2] ;
                count += 1;
            }
            com_S23_z /= count ;

            count = 0;
            for(auto i : S45){
                com_S45_z += (*(x + i))[2] ;
                count += 1;
            }
            com_S45_z /= count ;
        } // compute the COM_z for each boundary
        //std::cout << "Z boundary :" << com_S01_z << ", " << com_S23_z << ", " << com_S45_z << std::endl ;
        real com_center_x = 0.0;
        real com_center_y = 0.0;
        {
            int count = 0 ;
            for(auto i : S01){
                com_center_x += (*(x + i))[0] ;
                com_center_y += (*(x + i))[1] ;
                count += 1;
            }
            for(auto i : S23){
                com_center_x += (*(x + i))[0] ;
                com_center_y += (*(x + i))[1] ;
                count += 1;
            }
            for(auto i : S45){
                com_center_x += (*(x + i))[0] ;
                com_center_y += (*(x + i))[1] ;
                count += 1;
            }
            com_center_x /= count;
            com_center_y /= count;
        } // compute the COM_xy for center
        //std::cout << "COM x=" << com_center_x << ", y=" << com_center_y << ", z=" << com_center_z << std::endl ;
    // loop over ions and assign state(s)
    std::vector<int> ion_state_frame_vec(ion_index.size(), 2) ;
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
    std::vector<int> wat_state_frame_vec(wat_index.size(), 2) ;
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