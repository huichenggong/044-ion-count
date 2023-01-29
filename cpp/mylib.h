//
// Created by cheng on 23-1-24.
//

#ifndef SELECTIVITY_FILTER_MYLIB_H
#define SELECTIVITY_FILTER_MYLIB_H
#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "gromacs/fileio/xtcio.h"



namespace py = pybind11;

class Sfilter {
private:
    std::string xtc_file_name;
    t_fileio*  xdrFile;
    int        natoms;

public:
    std::vector<real> distance_vec ;
    std::vector<std::vector<int>> ion_state_vec ;
    std::vector<std::vector<int>> wat_state_vec ;

    real       time;
    int64_t    step;

    Sfilter( std::string file_name_in) ;
    std::string get_file_name();
    int distance(int index_a, int index_b);
    int count();
    Eigen::VectorXf get_distance();
    int assign_state_double(const std::vector<int> &S01,
                            const std::vector<int> &S23,
                            const std::vector<int> &S45,
                            const std::vector<int> &center,
                            const std::vector<int> &ion_index,
                            const std::vector<int> &wat_index,
                            const float rad);


    ~Sfilter();
};

#endif //SELECTIVITY_FILTER_MYLIB_H
