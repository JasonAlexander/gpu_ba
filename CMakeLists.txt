##     GPU-BA: Bundle Adjustment on the GPU
##         Ronny Hänsch, Igor Drude
##       Technische Universität Berlin
##  ----1. Intro----
##    For large-sized optimization problems, bundle adjustment has become more relevant over the past
##    years. Several research papers were published, discussing the task of working with big amounts of data,
##    retrieved from various internet photo collections. However, only a few work has been done considering
##    the acceleration properties of GPU systems. At the moment, it is a common approach to address
##    aforementioned problems with the "Levenberg-Marquardt" - algorithm, combined with the conjugate-
##    gradients method for solving the normal equations. This thesis also studies other approaches than that,
##    such as the nonlinear conjugate-gradient or the alterning resection-intersection procedure. Whilst the
##    latter one reveals partly competitive performance on a GPU system, this thesis also explores other ideas
##    to improve the convergence behavior, e.g. a hybrid between the Resection-Itersection and the
##    Levenber-Marquardt - algorithm or the recently suggested embedded point iterations. Furthermore, this
##    elaboration displays the possibilities of time- and space savings, when fitting the implementation
##    strategy to the terms and requirements given by the specific assignment of realizing a bundler on a
##    heterogeneous CPU-GPU system.

# ==============================================================================
#                   GPU-BA CMake
# ==============================================================================

message(STATUS " ~~~ Entering GPU-BA CMake ~~~ ")

cmake_minimum_required(VERSION 3.9)
project(GPU_BA VERSION 1.0.0 LANGUAGES CXX CUDA)
enable_language(CUDA)


# ==============================================================================
#                   Add GPU-BA Targets
# ==============================================================================

add_library(GPU_BA "")

set(CPP_HDR
        ${CMAKE_CURRENT_LIST_DIR}/CalcUtilsGPU.h
        ${CMAKE_CURRENT_LIST_DIR}/DataHandler.h
        ${CMAKE_CURRENT_LIST_DIR}/LMalgorithm.h
        ${CMAKE_CURRENT_LIST_DIR}/MBATimer.h
        ${CMAKE_CURRENT_LIST_DIR}/NCGalgorithm.h
        ${CMAKE_CURRENT_LIST_DIR}/ResInt.h
        ${CMAKE_CURRENT_LIST_DIR}/resource.h )

set(CPP_SRC
        ${CMAKE_CURRENT_LIST_DIR}/DataHandler.cpp
#        ${CMAKE_CURRENT_LIST_DIR}/driver.cpp
        ${CMAKE_CURRENT_LIST_DIR}/LMalgorithm.cpp
        ${CMAKE_CURRENT_LIST_DIR}/NCGalgorithm.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ResInt.cpp )

set(CUDA_HDR
        ${CMAKE_CURRENT_LIST_DIR}/Inverse.cuh
        ${CMAKE_CURRENT_LIST_DIR}/Solve.cuh )

set(CUDA_SRC
        ${CMAKE_CURRENT_LIST_DIR}/CalcUtilsGPU.cu )

target_sources(GPU_BA
        PRIVATE
            ${CPP_HDR}
            ${CPP_SRC}
            ${CUDA_HDR}
            ${CUDA_SRC})

#Add an alias so that library can be used inside the build tree, e.g. when testing
#add_library(CoronaXL::corona ALIAS corona)

# =============== End of Add GPU-BA Targets =================================



# ==============================================================================
#                          Define Properties
# ==============================================================================
#
#include(GPU_BAProperties.cm)
#
# ==================== End of Define Properties ================================



# ==============================================================================
#                          Set Variables
# ==============================================================================

set(CMAKE_CXX_FLAGS_DEBUG "-DGPU_BA_DEBUG")

#Make sure that custom modules like FindRapidJSON or FindCUDA are found
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add if statements as needed for the below defined
#add_definitions( -D__CUDACC__ )

# ====================== End of Set Variables  =================================



# ==============================================================================
#                           Set Options
# ==============================================================================

option(BUILD_TESTING "Enable testing" ON)

# ====================== End of Set Options  =================================


# ==============================================================================
#                          Get Device Stats
# ==============================================================================
#
#set(DeviceRun "${CMAKE_CURRENT_LIST_DIR}/DeviceStats")
#execute_process(COMMAND
#          ${DeviceRun} #-short
#        WORKING_DIRECTORY
#          ${CMAKE_CURRENT_LIST_DIR}
#        OUTPUT_VARIABLE
#          DEVICE_STATS
#        ERROR_VARIABLE
#          DEVICE_ERRORS
#        )
#message("Device Stats: \n ${DEVICE_STATS}")
#
# ==================== End of Get Device Stats =================================



# ==============================================================================
#                       Find 3rd Party Packages
# ==============================================================================
#
#include(GPU_BAFindPackages.cm)
#
# ================= End of Find 3rd Party Packages =============================



# ==============================================================================
#                         Include Directories
# ==============================================================================
#
#include_directories(
#        PRIVATE
#          include
#          src
#          ${PROJECT_SOURCE_DIR}
#)
#
# ===================== End of Include Directories =============================



# ==============================================================================
#                    Add CMake Include Directories
# ==============================================================================
#
#include(src/corona/data_distributors/data_distributors.cm)
#include(src/corona/data_mappers/data_mappers.cm)
#include(src/corona/tasks/tasks.cm)
#include(src/corona/test/test.cm)
#include(src/corona/utilities/utilities.cm)
#include(apps/apps.cm)
#
# ================ End of Add CMake Include Directories ========================



# ==============================================================================
#                          Setup GPU-BA Target
# ==============================================================================
#
#include(GPU_BATargetSetup.cm)
#
# ===================== End of Setup GPU-BA Target =============================



# ==============================================================================
#                         Add GPU-BA GTest
# ==============================================================================
#
#include(GPU_BATests.cm)
#
# ================== End of End of Add GPU-BA GTest ============================



# ==============================================================================
#                      Cmake Install Commands
# ==============================================================================
#
#include(GPU_BAInstall.cm)
#
# ================= End of Cmake Install Commands ==============================


message(STATUS " ~~~ Leaving GPU-BA CMake ~~~ ")

# ==================== End of GPU-BA CMake ==================================
