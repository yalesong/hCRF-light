cmake_minimum_required(VERSION 2.6)
project(hCRF)
#PUBLIC VERSION
ADD_DEFINITIONS(-DUSELBFGS -D_PUBLIC)

# We want -fPIC on linux, to be able to link to dynamic library
# IF(${UNIX})
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
# ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra ")

# For debugging. We also want the extra warning of Effective c++
IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
  ADD_DEFINITIONS(-D_DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weffc++")
ENDIF()

# This is for gnu c++ compiler OPENMP support.
# We have to compile with vec_features (to avoid shared object)
find_package(OpenMP)
if (OPENMP_FOUND)
  message("Compiling with openmp support")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  ADD_DEFINITIONS(-D_VEC_FEATURES)
endif()

if (PROFILING GREATER 0)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

# We need to add other subdirectory
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/../../3rdParty/liblbfgs  lbfgs)
# The file we need to include
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${liblbfgs_SOURCE_DIR}/include)

# This is the list of source files
SET(LIB_SRC src/dataset.cpp
	src/evaluator.cpp
	src/evaluatorCRF.cpp
	src/evaluatorHCRF.cpp
	src/evaluatorHSSHCRF.cpp
	src/evaluatorLDCRF.cpp
	src/evaluatorMVHCRF.cpp
	src/evaluatorMVLDCRF.cpp
	src/featuregenerator.cpp
	src/Features.cpp
	src/gradient.cpp
	src/GradientCRF.cpp
	src/GradientHCRF.cpp
	src/GradientHSSHCRF.cpp
	src/GradientLDCRF.cpp
	src/GradientMVHCRF.cpp
	src/GradientMVLDCRF.cpp
	src/HierarchicalFeatures.cpp
	src/InferenceEngine.cpp
	src/InferenceEngineBP.cpp
	src/InferenceEngineFB.cpp
	src/InferenceEngineJT.cpp
	src/InferenceEngineLoopyBP.cpp
	src/matrix.cpp
	src/matrixSSE2.cpp
	src/model.cpp
	src/MultiviewFeatures.cpp
	src/optimizer.cpp
	src/OptimizerLBFGS.cpp
	src/OptimizerNRBM.cpp
	src/Segment.cpp
	src/toolbox.cpp
	src/ToolboxCRF.cpp
	src/ToolboxHCRF.cpp
	src/ToolboxHSSHCRF.cpp
	src/ToolboxLDCRF.cpp
	src/ToolboxMVHCRF.cpp
	src/ToolboxMVLDCRF.cpp)
add_library(hCRF ${LIB_SRC})
SET(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
INSTALL_TARGETS(/lib hCRF)