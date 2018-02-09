# Specify the minimum version for CMake

cmake_minimum_required(VERSION 3.4.1)

add_compile_options(-std=c++14)

project (opticalFlow VERSION 0.1 LANGUAGES CXX)
#Bring the headers into the project
include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})
 
#Bring sources into the project:
file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "include/*.h" "include/*.hpp")

source_group("Headers" FILES ${HEADERS})
source_group("Sources" FILES ${SOURCES})

find_package( OpenCV REQUIRED )
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 found: ${EIGEN3_FOUND}")
message(STATUS "Eigen3 include dir found at ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")

add_executable(OpticalFlow ${SOURCES} ${HEADERS})
target_link_libraries( OpticalFlow ${OpenCV_LIBS} )
