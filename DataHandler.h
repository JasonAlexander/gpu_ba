#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

enum BAmode {metric, focal_radial}; 
enum Solver {LMA, NCG, RI};

// parse input data of the optimization problem //
// partially extracted and modified from PBA (Changchang Wu, 2010)
bool LoadBundlerModelBAL(char* fname, float*& camera_data, float*& point_data,
              float*& measurements, int*& ptidx, int*& camidx, int& n_measurements, int& n_cameras, int& n_points);

// preprocessing data before optimization //
// partially extracted and modified from PBA (Changchang Wu, 2010)
void PreProcessing(float*& camera_data, float*& point_data,
              float*& measurements, int*& ptidx, int*& camidx, int& n_measurements, int& n_cameras, int& n_points, bool fix, bool deleteProj, bool reorder);

// parse configuration file //
bool parseGeneralConfig(char *init, BAmode &im, Solver &sm, char* &path, bool &fix, bool &del, bool &reorder, bool &prprog, bool &trace, bool &smsse);

// parse configuration file //
bool parseSolverConfig(char *init, int &maxI, int &cg_minI, int &cg_maxI, int &epi, float &gradM_stop, float &mse_stop, float &diff_stop, int &il, float &maxL, float &minL,
				 float &tau, float &n_fs, bool &schur, bool &backsub, bool &diagdamping, float &t, float &c1, float &c2, float &start_a, int &b_reset, float &hyb_sw);