#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>
#include "LMalgorithm.h"
#include "NCGalgorithm.h"
#include "ResInt.h"
#include "CalcUtils.h"

using namespace std;

int main(int argc, char* argv[])
{

	// configuration file path //
    char* init_filename  = argv[1];           

	// configuration parameters //
	BAmode im;  // optimize camera motion only or optimize focal+distortion
	Solver sm;	// which solver to use

	char* path;								  // input file path
	bool fix, del, reorder;					  // data perprocessing
	bool print_progress, trace, show_min_sse; // user output
	bool schur, backsub, diagdamping;		  // LMA specific 

	float gradM_stop, mse_stop, diff_stop, hyb_sw;  // stopping thresholds
	int maxI, cg_minI, cg_maxI, epi, il;			// iteration numbers

	float tau, maxL, minL;	// initial lambda scaling
	float n_fs;				// forcing sequence (for CG-solver)
	float t,c1,c2,start_a;  // line search parameters

	int b_reset; // search direction reset intervall for NCG-solver

	// parse config file //
	if(!parseGeneralConfig(init_filename, im, sm, path, fix, del, reorder, print_progress, trace, show_min_sse))
	{
		cout << "Error reading config file " << endl;
		return 0;
	}
	if(!parseSolverConfig(init_filename, maxI, cg_minI, cg_maxI, epi, gradM_stop, mse_stop, diff_stop, il, maxL, minL, tau, n_fs, schur, backsub, diagdamping, 
		t, c1, c2, start_a, b_reset, hyb_sw))
	{
		cout << "Error reading config file " << endl;
		return 0;
	}

	// BA parameters //
	float	*camera_data;			  //camera (input/ouput)
    float	*point_data;			  //3D point(input/output)
    float   *measurements;			  //measurment/projection vector
    int     *camidx, *ptidx;		  //index of camera/point for each projection
    int     *camidx_cs, *ptidx_cs;
	// number of measurements, cameras and points //
	int nproj = 0, ncam = 0, npt = 0;

	// load camera parameter, points and measurements //	
	if (!LoadBundlerModelBAL(path, camera_data, point_data, measurements, ptidx, camidx, nproj, ncam, npt)){
		cout << "Error loading Input File " << endl;
		return 0;
	}

	// preprocessing data 
	PreProcessing(camera_data, point_data, measurements, ptidx, camidx, nproj, ncam, npt, fix, del, reorder);

	int n = (im==metric)?12:18;

	// check compute capability for compatibility issues
	if (!CalcUtilsGPU::checkCudaVersion()) exit(EXIT_SUCCESS);

	if(sm == LMA){

		// start Solver //
		LMalgorithm lma = LMalgorithm(maxI,gradM_stop,mse_stop,diff_stop, tau, maxL, minL, schur, backsub, epi, diagdamping, trace, n_fs, cg_minI, cg_maxI, hyb_sw, show_min_sse, print_progress);
		lma.runMinimizer(im, camera_data, point_data, measurements, ptidx, camidx, nproj, ncam, npt);
	}

	if(sm == NCG){

		// start Solver //
		NCGalgorithm ncga = NCGalgorithm(gradM_stop, mse_stop, print_progress, tau, trace, maxI, b_reset, t, c1, c2, start_a);
		ncga.runMinimizer(im, camera_data, point_data, measurements, ptidx, camidx, nproj, ncam, npt);
	}

	if(sm == RI){

		// start Solver //
		ResInt ri = ResInt(il, maxI, gradM_stop, mse_stop, print_progress, tau, maxL, minL, trace, show_min_sse);
		ri.runMinimizer(im, camera_data, point_data, measurements, ptidx, camidx, nproj, ncam, npt);
	}

	// Write back data to file here if needed (no function provided)
	/*
	for (unsigned int i = 0; i < ncam; i++){ 
		cout << "cam nr. " << i 
			 << ": rx: " << camera_data[i] 
			 << ", ry: " << camera_data[ncam*1+i]
			 << ", rz: " << camera_data[ncam*2+i]
			 << ", tx: " << camera_data[ncam*3+i]
			 << ", ty: " << camera_data[ncam*4+i]
			 << ", tz: " << camera_data[ncam*5+i]
			 << ", f: "  << camera_data[ncam*6+i]
			 << ", k1: " << camera_data[ncam*7+i]
			 << ", k2: " << camera_data[ncam*8+i] 
			 << endl;
	}

	for (unsigned int i = 0; i < npt; i++){ 
		cout << "point nr. " << i 
			 << ": x: " << point_data[i] 
			 << ", y: " << point_data[ncam*1+i]
			 << ", z: " << point_data[ncam*2+i]
			 << endl;
	}
	*/

	return 1;
}
