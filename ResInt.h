#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>

#include "CalcUtils.h"
#include "CalcUtilsGPU.h"

using namespace std;


class ResInt
{

public:

int max_iterationsI;        // max number of inner LM iterations
int max_iterationsO;        // max number of outer RI iterations
float gradient_th;          // stop criteria threshold for too small gradient magnitudes; checked in every iteration
float mse_th;				// stop criteria for a reached specified aim error (error gets below the sse threshold)
float tau;                  // damping term initial value (high => Gradient descent; low => Gauﬂ-Newton)
float maxL;					// max lambda scale factor relative to 1 (if multiplicative damping)
float minL;					// min lambda scale factor relative to 1 (if multiplicative damping)
bool print_prog;			// print progress? (requires more time)
bool trace;					// print detailed results for each iterations at the end
bool show_min_sse;			// if trace is enabled, print the actually lowest computed sse, or the actual computed sse


// Constructor //
ResInt(int maxIi, int maxIo, float stopcritGradient, float stopcritMSE, bool print_progress, float tauArg, float maxLa, float minLa, bool tracer, bool smsse);

// start the BA minimizing process //
bool runMinimizer(BAmode im, float* camera_data, float* point_data,
					float* measurements, int* ptidx, int* camidx, int n_measurements, int n_cameras, int n_points);




};
