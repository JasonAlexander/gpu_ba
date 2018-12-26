#pragma once

#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>

#include "CalcUtils.h"
#include "CalcUtilsGPU.h"

class NCGalgorithm
{

public:

	float gradient_th;          // stop criteria threshold for too small gradient magnitudes; checked in every iteration
	float mse_th;				// stop criteria for a reached specified aim error (error gets below the sse threshold)
	bool trace;					// print detailed results for each iterations at the end
	float tau;                  // damping term initial value
	int maxIter;				// max iterations
	bool print_prog;			// print progress? (requires more time)

	int b_reset;				// reset ncg (b=0) every b_reset iterations
	float t;					// alpha manipulator for each linesearch iteration
	float c1;					// first wolfe condition parameter for the armijo rule 0<c1<1
	float c2;					// second wolfe condition parameter for the curvature rule, 0<c1<c2<1, c=0 -> don't use curvature rule
	float start_a;				// starting value for alpha (start_a > 0) for the first linesearch iteration, start_a = 0 -> use dynamic prediction


	// Constructor //
	NCGalgorithm(float gradM, float mse_stop, bool print_progress, float tau_v, bool tracer, int maxI, int _b_reset, float _t, float _c1, float _c2, float _start_a);

	// start the BA minimizing process //
	bool runMinimizer(BAmode im, float* camera_data, float* point_data,
					float* measurements, int* ptidx, int* camidx, int n_measurements, int n_cameras, int n_points);
	
};

