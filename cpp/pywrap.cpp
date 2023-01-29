//
// Created by cheng on 23-1-24.
//
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "mylib.h"
#include <pybind11/stl.h>

namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;

PYBIND11_MODULE(PYSfilter, m) {
    m.doc() = "optional module docstring";

    py::class_<Sfilter>(m, "Sfilter")
            .def(py::init<std::string>())
            .def("get_file_name", &Sfilter::get_file_name, py::call_guard<py::gil_scoped_release>())
            .def("distance", &Sfilter::distance, py::call_guard<py::gil_scoped_release>())
            .def("get_distance", &Sfilter::get_distance, py::call_guard<py::gil_scoped_release>()) // remove this later
            .def("count", &Sfilter::count, py::call_guard<py::gil_scoped_release>())
            .def("assign_state_double", &Sfilter::assign_state_double, py::call_guard<py::gil_scoped_release>())
            .def_readonly("distance_vec", &Sfilter::distance_vec, byref) // remove this later
            .def_readonly("ion_state_list", &Sfilter::ion_state_vec, byref)
            .def_readonly("wat_state_list", &Sfilter::wat_state_vec, byref)
            .def_readonly("time", &Sfilter::time, byref)
            .def_readonly("step", &Sfilter::step, byref)
            ;
}

//distance_vecEigen