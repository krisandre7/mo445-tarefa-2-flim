cmake_minimum_required(VERSION 2.8.11)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../CMakeLists.txt)

#change to your QT instalation path
set(CMAKE_PREFIX_PATH "~/Qt/5.8/gcc_64")

include_directories(.)
file(GLOB DA_SRC *.cpp)

project(DA)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

find_package(Qt5Widgets)
find_package(Qt5Concurrent)

add_executable(DA ${DA_SRC})

target_link_libraries(DA Qt5::Widgets Qt5::Concurrent ift ${LAPACK_LIBRARIES} ${CBLAS_LIBRARIES} m stdc++ ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES} ${OpenMP_LIBRARIES})
