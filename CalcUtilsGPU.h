#pragma once

#include <stdio.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "device_launch_parameters.h"

#include "DataHandler.h"
#include "Inverse.cuh"
#include "Solve.cuh"

#define typ double 

using namespace std;


namespace CalcUtilsGPU
{

	/////////// initial calls ////////////
	int checkCudaVersion();
	size_t copyDataIntoDevice(float *camera_data, float *point_data,
				float *measurements, int* ptidx, int* camidx, int NM, int NC, int NP, int ncp, Solver s, bool diagdamp);
	int initSolver(Solver solv, int ncp, bool hess, int epi);

	////////// calls fundamental for all optimization strategies ///////////
	void calcErrorVector_d(float &sse, float *c, float *p);
	void buildJacobian_d(int ri, float &lambdaC, float &lambdaP, Solver s, float &gradMag);
	int updateCheck(float sse, float& newSSE, int ri, bool fwd);

	/////////// Levenberg-Marquardt algorithm calls ///////////
	// use for hessian variant
	int solvePCG_Hessian_d(float lambdaC, float lambdaP, float gmag, int maxi, int mini, float fs);
	float calcAlpha_H(float *p, float *s, float *z, int ncp, double& dots, float *JtJp, float lc, float lp);

	// used for schur variant
	int solvePCG_Schur_d(float lambdaC, float lambdaP, float gmag, int maxi, int mini, float fs);
	float calcAlpha_S(float *p, float *s, float *z, int ncp, double& dots, float *Sp, float lc, float lp);
	void backSubstitution();
	int epi(bool backsub, float lambda, int iter, float sse, float &newSSE, float &iSSE);
	void calcRSofRCS(float* d_rcsRS);

	// used for both LMA implementations
	float calcro(float lambdaC, float lambdP);
	void formBdiagHessian(float lambdaC, float lambdaP, int ri);
	void invHessDiagBlocks(float l);
	double calcBeta(float *s, float *z, int ncp, double &dots, int length);

	/////////// nonlinear conjugate gradient calls ///////////
	void calcGDstep_d(float &sse, int ncp, int k, float l, int b_reset, float t, float c1, float c2, float start_a);
	void linesearch(float &a, float &sse, float t, float c1, float c2);

	///////////  resection-intersection calls ///////////
	void resection(float lambda);
	void intersection(float* t_grad, float lambda);

	///////////  extract final solution ///////////
	void wb_result(float *h_cams, float *h_points);



};