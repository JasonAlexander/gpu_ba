#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>

//#include "CalcUtils.h"
#include "CalcUtilsGPU.h"

using namespace std;


class LMalgorithm
{

public:

int max_iterations;         // max number of LM iterations
float gradient_th;          // stop criteria threshold for too small gradient magnitudes; checked in every iteration
float mse_th;				// stop criteria for a reached specified aim error (error gets below the sse threshold)
float tau;                  // damping term initial value (high => Gradient descent; low => Gau�-Newton)
float maxL;					// max lambda scale factor relative to initial lambda diag value (if additive) or 1 (if multiplicative)
float minL;					// min lambda scale factor relative to initial lambda diag value (if additive) or 1 (if multiplicative)
float diff_stop;			// stop criteria, when abs(newSSE/(float)sse - 1) < diff_stop (too small error decreases), has to be very small, or the algorithm might stop too early
bool schur;					// Use Schur complement (=true) or full Hessian (=false)
bool backsub;				// do a Backsubstitution
int epi;					// Embedded Point iteration, 0 means do not use EPIs
bool trace;					// print detailed results for each iterations at the end
bool show_min_sse;			// if trace is enabled, print the actually lowest computed sse, or the actual computed sse
bool print_progress;		// print progress? (requires more time)
bool diagdamping;			// true for multiplicative diagonal damping (like Marquardt suggested), else fixed additional damping
float n_fs;					// error decrease threshold for the CG solver (forcing sequence)
int cg_minI;				// number of min iterations for the CG solver
int cg_maxI;				// number of max iterations for the CG solver
float hyb_switch;			// when to switch from RI to LMA (0 = Hybrid disabled); switch when newSSE/sse < hyb_switch+1

// Constructor //
LMalgorithm(int maxI, float stopcritGradient, float stopcritSSE, float diff_stop, float tauArg, float maxLa, float minLa, bool lmamode, bool backsubst, int epis, bool diagdamp, bool tracer, 
			float fs, int imin, float imax, float hyb_sw, bool sms, bool print_prog);

// start the BA minimizing process //
bool runMinimizer(BAmode im, float*& camera_data, float*& point_data,
					float* measurements, int* ptidx, int* camidx, int n_measurements, int n_cameras, int n_points);




};

