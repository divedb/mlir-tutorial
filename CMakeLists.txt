cmake_minimum_required(VERSION 3.20)
project(Toy LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LLVM_DIR
  ${CMAKE_SOURCE_DIR}/third_party/llvm-project/build/lib/cmake/llvm
)
set(MLIR_DIR
  ${CMAKE_SOURCE_DIR}/third_party/llvm-project/build/lib/cmake/mlir
)

message(STATUS "${MLIR_DIR}")

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include(HandleLLVMOptions)

include_directories(
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/include
)

add_definitions(${LLVM_DEFINITIONS})
add_subdirectory(toy)
add_subdirectory(basic)