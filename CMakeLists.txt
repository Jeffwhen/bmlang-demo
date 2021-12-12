cmake_minimum_required(VERSION 3.5)

project(bmlangdemo)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_CXX_STANDARD 11)

find_package(bmnn REQUIRED)

add_executable(gen gen.cpp)
target_link_libraries(gen bmnn::bmlang)

add_executable(run run.cpp)
target_link_libraries(run bmnn::bmrt bmnn::bmlib)

