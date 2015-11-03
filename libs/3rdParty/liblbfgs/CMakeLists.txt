# We want to have cmake support, for one command build
# Added by hsalamin on 2010/08/07

cmake_minimum_required(VERSION 2.6)
project(liblbfgs)

IF(UNIX)
  IF(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  ENDIF()
ENDIF()

INCLUDE_DIRECTORIES(${liblbfgs_SOURCE_DIR}/include)
add_library(lbfgs lib/lbfgs.c)

IF(UNIX)
  IF(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  ENDIF()
ENDIF()

SET(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
INSTALL_TARGETS(/lib lbfgs)
