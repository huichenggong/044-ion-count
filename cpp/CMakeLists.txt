cmake_minimum_required(VERSION 3.10)

project(PYSfilter)
set(CMAKE_CXX_STANDARD 20)
set(PYBIND11_PYTHON_VERSION 3.11)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC -O3 -L/home/cheng/Software/GMX/2022.4-gcc-cuda/lib")

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(gromacs REQUIRED)

pybind11_add_module(${PROJECT_NAME} pywrap.cpp mylib.h mylib.cpp)

target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_include_directories(${PROJECT_NAME} PRIVATE
        ${PYBIND11_INCLUDE_DIRS}
        /home/cheng/Software/Package/GMX/2022.4/gromacs-2022.4/api/legacy/include
        /home/cheng/Software/Package/GMX/2022.4/gromacs-2022.4/src
        )


target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen
        gromacs)
