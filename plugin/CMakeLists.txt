message(WARNING "
  =============> SM VERSION SETTING <===============
  By default tiny-tensorrt will compile with compatibility with nvidia's 10x and 20x device(1080ti, 2080 etc, SM_60-SM75), so it might cause compile error under other device like jestion nano(SM_53)
  It would be better that you specify with your device's sm version as shown below, usage:
  cmake -DSM_VERSION=53 ..
  Fermi (CUDA 3.2 until CUDA 8) (deprecated from CUDA 9):
  SM20 or SM_20, compute_30 – Older cards such as GeForce 400, 500, 600, GT-630
  Kepler (CUDA 5 and later):
  SM30 or SM_30, compute_30 – Kepler architecture (generic – Tesla K40/K80, GeForce 700, GT-730)
  Adds support for unified memory programming
  SM35 or SM_35, compute_35 – More specific Tesla K40
  Adds support for dynamic parallelism. Shows no real benefit over SM30 in my experience.
  SM37 or SM_37, compute_37 – More specific Tesla K80
  Adds a few more registers. Shows no real benefit over SM30 in my experience
  Maxwell (CUDA 6 and later):
  SM50 or SM_50, compute_50 – Tesla/Quadro M series
  SM52 or SM_52, compute_52 – Quadro M6000 , GeForce 900, GTX-970, GTX-980, GTX Titan X
  SM53 or SM_53, compute_53 – Tegra (Jetson) TX1 / Tegra X1, Drive CX, Drive PX, Jetson Nano
  Pascal (CUDA 8 and later)
  SM60 or SM_60, compute_60 – Quadro GP100, Tesla P100, DGX-1 (Generic Pascal)
  SM61 or SM_61, compute_61 – GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4, Discrete GPU on the NVIDIA Drive PX2
  SM62 or SM_62, compute_62 – Integrated GPU on the NVIDIA Drive PX2, Tegra (Jetson) TX2
  Volta (CUDA 9 and later)
  SM70 or SM_70, compute_70 – DGX-1 with Volta, Tesla V100, GTX 1180 (GV104), Titan V, Quadro GV100
  SM72 or SM_72, compute_72 – Jetson AGX Xavier, Drive AGX Pegasus, Xavier NX
  Turing (CUDA 10 and later)
  SM75 or SM_75, compute_75 – GTX/RTX Turing – GTX 1660 Ti, RTX 2060, RTX 2070, RTX 2080, Titan RTX, Quadro RTX 4000, Quadro RTX 5000, Quadro RTX 6000, Quadro RTX 8000, Quadro T1000/T2000, Tesla T4
  Ampere (CUDA 11 and later)
  SM80 or SM_80, compute_80 – Tesla A100 (GA100), NVIDIA DGX-A100, RTX Ampere – RTX 3080
  Hopper – currently unknown.
  ======================================================
")

message(STATUS "
  =============> ADDER2D FP16 SETTING <===============
  By default prelu plugin's fp16 mode is disable since it require device that sm version is equal or greater than SM_60, so you need to enable it manully.
  usage cmake -DENABLE_ADDER2D_FP16=ON ..
  ==================================================
")

message(STATUS "
  =============> USAGE <===============
  cmake -DENABLE_ADDER2D_FP16=ON/OFF -DSM_VERSION=xx -DBUILD_PYTHON=ON/OFF -DBUILD_TEST=ON/OFF -DBUILD_KERNEL_TEST=ON/OFF ..
  =====================================
")

cmake_minimum_required(VERSION 3.0)
project(adder2dtrt)
set(CMAKE_CXX_FLAGS "-std=c++11")

# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib CACHE PATH "")
option(BUILD_PYTHON "compile python api" ON)
option(BUILD_TEST "compile plugin test" ON)
option(BUILD_KERNEL_TEST "compile cuda kernel test" ON)

find_package(CUDA REQUIRED)

include(cmake/CUDA_utils.cmake)

set(SM_VERSION "" CACHE STRING "Description")

if(SM_VERSION)
  set(CUDA_targeted_archs ${SM_VERSION})
  CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_targeted_archs})
else()
  # Discover what architectures does nvcc support
  CUDA_find_supported_arch_values(CUDA_supported_archs ${CUDA_known_archs})
  message(STATUS "CUDA supported archs: ${CUDA_supported_archs}")

  set(CUDA_TARGET_ARCHS_SORTED ${CUDA_TARGET_ARCHS})
  list(SORT CUDA_TARGET_ARCHS_SORTED)
  CUDA_find_supported_arch_values(CUDA_targeted_archs ${CUDA_TARGET_ARCHS_SORTED})
  message(STATUS "CUDA targeted archs: ${CUDA_targeted_archs}")
  if (NOT CUDA_targeted_archs)
    message(FATAL_ERROR "None of the provided CUDA architectures ({${CUDA_TARGET_ARCHS}}) is supported by nvcc, use one or more of: ${CUDA_supported_archs}")
  endif()
  CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_targeted_archs})
endif()

# Add ptx & bin flags for cuda
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_gencode_flags}")

option(ENABLE_ADDER2D_FP16 "" OFF)
if(ENABLE_ADDER2D_FP16)
  add_definitions(-DFP16_ADDER2D)
endif()

#include_directories(pybind11/include)
include_directories(./)

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

message(STATUS "
Generated gencode flags: ${CUDA_gencode_flags}
BUILD_PYTHON : ${BUILD_PYTHON}
BUILD_TEST : ${BUILD_TEST}
BUILD_KERNEL_TEST : ${BUILD_KERNEL_TEST}
ENABLE_ADDER2D_FP16 : ${ENABLE_ADDER2D_FP16}
")


file(GLOB_RECURSE adder2dtrt_source
     Adder2dPlugin.cu
     PluginUtils.cpp
     )
cuda_add_library(adder2dtrt SHARED ${adder2dtrt_source})
target_compile_options(adder2dtrt PUBLIC -std=c++11 -Wall -Wfloat-conversion)
set_target_properties(adder2dtrt PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_TEST)
  file(GLOB test_source
      ../test/TestAdder2dPlugin.cpp
      )
  add_executable(TestAdder2dPlugin ${test_source})
  target_compile_options(TestAdder2dPlugin PUBLIC -std=c++11 -Wall -Wfloat-conversion)
  target_link_libraries(TestAdder2dPlugin adder2dtrt)
  target_link_libraries(TestAdder2dPlugin nvinfer)
  target_link_libraries(TestAdder2dPlugin nvinfer_plugin)
  target_link_libraries(TestAdder2dPlugin nvparsers)
  target_link_libraries(TestAdder2dPlugin nvonnxparser)
  target_link_libraries(TestAdder2dPlugin nvcaffe_parser)
  target_link_libraries(TestAdder2dPlugin ${CUDART})
endif()

if(BUILD_KERNEL_TEST)
  file(GLOB kernel_source
      ../test/TestAdderFilterCudaKernel.cu
      )
  cuda_add_executable(TestAdderFilterCudaKernel ${kernel_source})
  target_compile_options(TestAdderFilterCudaKernel PUBLIC -std=c++11 -Wall -Wfloat-conversion)
  target_link_libraries(TestAdderFilterCudaKernel ${CUDART})
endif()

if(BUILD_PYTHON)
  # set(Python3_ROOT_DIR /root/miniconda3/bin)
  # find_package(Python3 REQUIRED)
  include_directories(${PYTHON_INCLUDE_DIRS})
  add_subdirectory(pybind11)
  pybind11_add_module(adder2dpytrt SHARED Adder2dPyTrt.cpp)
  target_link_libraries(adder2dpytrt PRIVATE adder2dtrt)
  target_link_libraries(adder2dpytrt PRIVATE nvinfer)
  target_link_libraries(adder2dpytrt PRIVATE nvinfer_plugin)
  target_link_libraries(adder2dpytrt PRIVATE nvparsers)
  target_link_libraries(adder2dpytrt PRIVATE nvonnxparser)
  target_link_libraries(adder2dpytrt PRIVATE nvcaffe_parser)
endif()
