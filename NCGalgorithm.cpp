#include "NCGalgorithm.h"
#include "MBATimer.h"
#include <iostream>
#include <iomanip>
#include <algorithm> 

NCGalgorithm::NCGalgorithm(float gradM, float mse_stop, bool print_progress, float tau_v, bool tracer, int maxI, int _b_reset, float _t, float _c1, float _c2, float _start_a)
{
	gradient_th = gradM;
	mse_th = mse_stop;
	print_prog = print_progress;
	trace = tracer;
	tau = tau_v;
	maxIter = maxI;

	b_reset = _b_reset;
	t = _t;
	c1 = _c1;
	c2 = _c2;
	start_a = _start_a;
}

bool NCGalgorithm::runMinimizer(BAmode im, float* camera_data, float* point_data,
					float* measurements, int* ptidx, int* camidx, int n_measurements, int n_cameras, int n_points)
{


	float gradMag = -1;					// gradient magnitude (J^T * error_vector)
	bool stop = false;					// termination variable for the loop
	float prevsse = 0, sse = 0;			// current Summed Squared Error
	int k = 0;							// iteration counter
	int s = (im == metric)?12:18;		// number of camera parameters * 2
	int pv_num = 
		(n_points*3)+(n_cameras*s/2);	// parameter vector length
	float initsse = 0;

	// for tracer
	vector<double> timestamps;
	vector<float> SSEs;
	vector<float> rel_decr;

	// Jacobian matrix J = [J_c, J_p] stored in a simplyfied BSR format  //
	int mb = n_measurements; 	
	int nb_C = n_cameras;
	int nb_P = n_points;

	// jacobi non-zero elements in ascending order (from left to right for Transposed J_c) coord-major
	float *h_val_P =  (float *)malloc(sizeof(float*) * 6 * n_points);
	float *h_val_C =  (float *)malloc(sizeof(float*) * s * n_cameras);

	// init cuda context and get free memory
	cudaFree(0);

	// place input data into device memory and init cuda  //
	int mem_used1 = CalcUtilsGPU::copyDataIntoDevice(camera_data, point_data, measurements, ptidx, camidx, mb, nb_C, nb_P, s, NCG, true);
	if(mem_used1 == -1){
		cout << "Error: Not enough device storage for input data, jacobian matrix and preconditionier " <<  endl;
		return false;
	}
	// allocate solver specific data
	int mem_used2 = CalcUtilsGPU::initSolver(NCG, s, false, 0);
	if(mem_used2 == -1){
		cout << "Error: Not enough device storage for algorithm specific temporal vectors " <<  endl;
		return false;
	}

	double start_t = get_wall_time();

	// calculate initial vector of errors for each measured point and sse//
  	CalcUtilsGPU::calcErrorVector_d(sse, 0, 0);
	initsse = sse;
	prevsse = sse;
	cout << "Initial MSE: " << (sse/(mb)) << endl;

	// build initial Jacobian matrix part on GPU //
	float l  = tau;	 
	CalcUtilsGPU::buildJacobian_d(-1, l, l, NCG, gradMag);

	///////////////////////////////////////////
	// evaluate initial stop condition
	stop = (  (sse/n_measurements <= mse_th ) ); // (gradMag/n_measurements <= gradient_th) || 

	char* str = (im == metric)?"metric":"focal_radial";
	cout << "Start Nonlinear Conjugate Gradient minimizer with mode: " << str << endl;

	while(!stop && k<maxIter)
	{

		k++;		

		// calculate and apply next descent step //
		CalcUtilsGPU::calcGDstep_d(sse, s, k, l, b_reset, t, c1, c2, start_a);

		if(trace && (k%10 == 0 || k<10)){
			// timestamp
			timestamps.push_back(get_wall_time()-start_t);

			// store sse
			SSEs.push_back(sse);
			
			// write relative decrease in this iteration
			rel_decr.push_back(sse/initsse);
		}
		if(print_prog) cout << "iter: " << k << ", sse: " << ((float)sse/(float)(n_measurements)) << endl;

		// did the sse fall below the threshold ? //
		if( sse/(float)(n_measurements) <= mse_th)
			break;			 
		
		// calculate new gradient g = J^T*e //
		if(k%10 == 0){
			if(prevsse<=sse && abs(sse/prevsse) < 2 ) break;
			prevsse = sse;
		}

		// linearize //
		CalcUtilsGPU::buildJacobian_d(-1, l, l, RI, gradMag);
		

	}

	double time_elapsed = (get_wall_time() - start_t);

	// check device memory for sufficient space //
	size_t free,total;
	cuMemGetInfo(&free, &total);

	cout << ".............................................." << endl;
	cout << "Optimization finished after " << time_elapsed  << "s \n";
	cout << "final RMSE: " << sqrt(sse/(mb)) << ", ";
	cout << "Number of iterations: "  << k << endl;
	cout << ".............................................." << endl;
	cout << "Total Memory available                                       : " << setw(5) << total/1024/1024 << " mb \n";
	cout << "Used Memory for Jacobi/Preconditionier/Input-/Output-Vectors : " << setw(5) << mem_used1  << " mb \n";
	cout << "Used Memory for solver-specific additional temporal data     : " << setw(5) << mem_used2 << " mb \n";
	cout << ".............................................." << endl;

	if(trace){
		cout << endl;
		for(size_t i = 0; i< timestamps.size(); i++){
			printf( "t=%5.3fs, ", timestamps[i] );
			int iter = (i<10)?(i+1):(i-8)*10;
			cout << "iter#" << left << setw(2) << iter;
			cout << ", MSE = " << setw(7) << left << setprecision(5) << ((float)SSEs[i]/(float)(n_measurements)) << " pix, ";
			cout << "rd: " << setw(9) << setprecision(4) << rel_decr[i] << endl; 

		}
		cout << endl;	
	}
/*
	cout.imbue(locale(""));
	if(trace){
		cout << endl;
		for(size_t i = 0; i< timestamps.size(); i++){
			int iter = (i<10)?(i+1):(i-8)*10;
			cout << timestamps[i] << ";";
			cout << setw(7) << left << setprecision(6) << ((float)SSEs[i]/(float)(n_measurements))<< endl;	
		}
		cout << endl;
	}
*/
	CalcUtilsGPU::wb_result(camera_data, point_data);

	return 0;
}
