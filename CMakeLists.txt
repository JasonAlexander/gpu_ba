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

cmake_minimum_required(VERSION 3.15)
project(GPU_BA VERSION 1.0.0 LANGUAGES CXX CUDA)

# ==============================================================================
#                   Add GPU-BA Targets
# ==============================================================================

#add_library(GPU_BA "")
add_executable(GPU_BA "")

message("-- Check for OpenMP")
find_package(OpenMP)
if (OPENMP_FOUND)
    message("-- Found OpenMP ${OpenMP_VERSION}: ${OpenMP_INCLUDE_DIRS}")
else (OPENMP_FOUND)
    message(FATAL_ERROR "Can't find OpenMP.")
endif (OPENMP_FOUND)

message("-- Check for CUDA")
find_package(CUDA REQUIRED)
message("-- Found CUDA ${CUDA_INCLUDE_DIRS_VERSION}: ${CUDA_INCLUDE_DIRS}")
message("    Libraries:  ${CUDA_INCLUDE_DIRS_LIBS}")

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
        ${CMAKE_CURRENT_LIST_DIR}/driver.cpp
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

target_include_directories(GPU_BA
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/>
        PRIVATE
        ${CUDA_INCLUDE_DIRS}
        )

set(GENCODE -arch=sm_30 -gencode=arch=compute_50,code=sm_50)

target_link_options(GPU_BA
        PRIVATE
        $<$<BOOL:OPENMP_FOUND>:${OpenMP_CXX_FLAGS}>
        )

target_compile_options(GPU_BA
        PRIVATE
        # CXX FLAGS
        $<$<BOOL:OPENMP_FOUND>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${OpenMP_CXX_FLAGS}>>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wall>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wextra>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-pedantic>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wno-unused-function>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wno-switch>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wno-unused-parameter>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-std=c++17>
        # CXX RELEASE ONLY FLAGS
        $<$<CONFIG:Release>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Ofast>>
        $<$<CONFIG:Release>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-ffast-math>>
        $<$<CONFIG:Release>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-march=native>>
        $<$<CONFIG:Release>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wno-pedantic>>
        $<$<CONFIG:Release>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wno-unused-result>>
        $<$<CONFIG:Release>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wno-unused-parameter>>
        $<$<CONFIG:Release>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wno-return-type>>
        $<$<CONFIG:Release>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wno-sign-compare>>
        $<$<CONFIG:Release>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wno-unused-variable>>
        $<$<CONFIG:Release>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wno-maybe-uninitialized>>
        $<$<CONFIG:Release>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wno-unused-but-set-variable>>
        # CUDA FLAGS
        $<$<BOOL:OPENMP_FOUND>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${OpenMP_CXX_FLAGS}>>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-std=c++14>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:--compiler-options>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-use_fast_math>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:--default-stream per-thread>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:--expt-extended-lambda>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:--expt-relaxed-constexpr>
        #            $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-dc>
        #            $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Dlink>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-lcudart>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-lcudrand>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-lcudadevrt>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-lcublas_device>
        #CUDA DEBUG ONLY FLAGS
        $<$<CONFIG:Debug>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-O0>>
        #    $<$<CONFIG:Debug>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-G>>
        $<$<CONFIG:Debug>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:${GENCODE}>>
        $<$<CONFIG:Debug>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:--ptxas-options=-v>>
        #CUDA RELEASE ONLY FLAGS, with -DLOCAL_GPU=ON
        $<$<CONFIG:Release>:$<$<STREQUAL:${LOCAL_GPU},ON>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:${GENCODE}>>>
        $<$<CONFIG:Release>:$<$<STREQUAL:${LOCAL_GPU},ON>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:--ptxas-options=-v>>>
        #CUDA RELEASE ONLY FLAGS, with -DLOCAL_GPU=OFF
        $<$<CONFIG:Release>:$<$<STREQUAL:${LOCAL_GPU},OFF>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:--disable-warnings>>>
        $<$<CONFIG:Release>:$<$<STREQUAL:${LOCAL_GPU},OFF>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-gencode=arch=compute_30,code=sm_30>>>
        $<$<CONFIG:Release>:$<$<STREQUAL:${LOCAL_GPU},OFF>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-gencode=arch=compute_50,code=sm_50>>>
        $<$<CONFIG:Release>:$<$<STREQUAL:${LOCAL_GPU},OFF>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-gencode=arch=compute_52,code=sm_52>>>
        $<$<CONFIG:Release>:$<$<STREQUAL:${LOCAL_GPU},OFF>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-gencode=arch=compute_60,code=sm_60>>>
        $<$<CONFIG:Release>:$<$<STREQUAL:${LOCAL_GPU},OFF>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-gencode=arch=compute_61,code=sm_61>>>
        $<$<CONFIG:Release>:$<$<STREQUAL:${LOCAL_GPU},OFF>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-gencode=arch=compute_62,code=sm_62>>>
        $<$<CONFIG:Release>:$<$<STREQUAL:${LOCAL_GPU},OFF>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-gencode=arch=compute_70,code=sm_70>>>
        $<$<CONFIG:Release>:$<$<STREQUAL:${LOCAL_GPU},OFF>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-gencode=arch=compute_72,code=sm_72>>>
        $<$<CONFIG:Release>:$<$<STREQUAL:${LOCAL_GPU},OFF>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-gencode=arch=compute_75,code=sm_75>>>
        )

set_target_properties(GPU_BA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(GPU_BA PROPERTIES POSITION_INDEPENDENT_CODE ON)

message(STATUS " ~~~ Leaving GPU-BA CMake ~~~ ")

# ==================== End of GPU-BA CMake ==================================
