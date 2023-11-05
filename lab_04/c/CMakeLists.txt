cmake_minimum_required(VERSION 3.16)

project(playground VERSION 0.1.0 LANGUAGES CXX)

set(target main)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)



set(src_dir ${CMAKE_SOURCE_DIR}/src)
set(ompi_dir "/home/kkafara/software/dependencies/openmpi/include")

file(GLOB src_files ${src_dir}/*.cpp)

add_executable(${target} ${src_files})

target_include_directories(${target} PRIVATE ${ompi_dir})


