#include "ResInt.h"
#include "MBATimer.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>


ResInt::ResInt(int maxIi, int maxIo, float stopcritGradient, float stopcritMSE, bool print_progress, float tauArg, float maxLa, float minLa, bool tracer, bool smsse)
{
	max_iterationsI = maxIi;
	max_iterationsO = maxIo;
	gradient_th = stopcritGradient;
	mse_th = stopcritMSE;
	tau = tauArg;
	maxL = maxLa;
	minL = minLa;
	print_prog = print_progress;
	trace = tracer;
	show_min_sse = smsse;			
}

bool ResInt::runMinimizer(BAmode im, float* camera_data, float* point_data,
					float* measurements, int* ptidx, int* camidx, int n_measurements, int n_cameras, int n_points)
{


	int inner = 0;                      // inner iteration counter
	int outer = 0;                      // outer iteration counter
	float gradMag = -1;					// gradient magnitude (J^T * error_vector)
	bool stop = false;					// termination variable for the loop
	float sse = 0;						// current Summed Squared Error
	float newSSE = 0;					// newly computed Summed Squared Error
	float oldsse = 0;
	float v = 1;						// lambda manipulator term
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


	const char* str = (im == metric)?"metric":"focal_radial";
	cout << "Start Resection Intersection LMA minimizer with mode: " << str << endl;

	// jacobi non-zero elements in ascending order (from left to right for Transposed J_c) coord-major
	float *h_val_P_cu =  (float *)malloc(sizeof(float*) * 6 * mb);
	float *h_val_C_cu =  (float *)malloc(sizeof(float*) * s * mb);

	// init cuda context
	cudaFree(0);

	// place input data into device memory and init cuda  //
	int mem_used1 = CalcUtilsGPU::copyDataIntoDevice(camera_data, point_data, measurements, ptidx, camidx, mb, nb_C, nb_P, s, RI, true);
	if(mem_used1 == -1){
		cout << "Error: Not enough device storage for input data, jacobian matrix and preconditionier " <<  endl;
		return false;
	}
	// allocate solver specific data
	int mem_used2 = CalcUtilsGPU::initSolver(RI, s, false, 0);
	if(mem_used2 == -1){
		cout << "Error: Not enough device storage for algorithm specific temporal vectors " <<  endl;
		return false;
	}

	double start_t = get_wall_time();

	// calculate initial vector of errors for each measured point and sse//
  	CalcUtilsGPU::calcErrorVector_d(sse, 0, 0);
	initsse = sse;
	oldsse = sse;
	cout << "Initial MSE: " << (sse/(mb)) << endl;

	// build initial Jacobian matrix part on GPU //
	float lc = tau; //negative to indicate an update is necessary
	float lp = tau; //negative to indicate an update is necessary
	CalcUtilsGPU::buildJacobian_d(-1, lc, lp, RI, gradMag);
	float minlc = (float)(lc/tau)*minL; //minimum lambda value to ensure numerical stability
	float minlp = (float)(lp/tau)*minL; //minimum lambda value to ensure numerical stability

	float maxlc = (float)(lc/tau)*maxL; //minimum lambda value to ensure numerical stability
	float maxlp = (float)(lp/tau)*maxL; //minimum lambda value to ensure numerical stability

	// start outer R-I loop //
	int decreased = true;
	while(!stop && outer < max_iterationsO){

		// stop if outer loop error decrease was too small		
		lc = tau; 
		lp = tau;  
		outer++;
		inner = 0;
		int innerabs = 0;		
		bool istop = false; // inner stop for too small changes

		// stop optimization if 10 consecutive iterations without error decrease
		if(outer%10 == 0){
			if(oldsse<=sse) break;
			oldsse = sse;
		}

		// start inner LM loop //
		while(!stop && inner < max_iterationsI){

			inner++;
			if(istop) break; // stop inner RI loop if error decrease is to small

			// inner LMA loop, do until error gets lower  //
			
			do{		

				innerabs++;
				// solve blockwise normal equation with direct solver (via Cholesky decomposition) //
				if(outer%2==1){
					if(outer>1 || innerabs>1) 
						if(decreased){
							int ri = (max_iterationsI==1)?outer%2:-1;
							CalcUtilsGPU::buildJacobian_d(ri, lc, lp, RI, gradMag); // build new system with new decreased lambda
						}
					CalcUtilsGPU::resection(lc);    // solve only for cam params
				}
				else{
					if(decreased){
						int ri = (max_iterationsI==1)?outer%2:-1;
						CalcUtilsGPU::buildJacobian_d(ri, lc, lp, RI, gradMag); // build new system with new decreased lambda
					}
					CalcUtilsGPU::intersection(0,lp); // solve only for point params
				}
				// if error decreased : apply changes, calc new Jac, err and grad and decrease damping manipulator term  //
				newSSE = 0;
				decreased = CalcUtilsGPU::updateCheck(sse, newSSE, (outer%2), false);
				
				if(print_prog) cout << "iter: " << outer << ", sse: " << ((float)newSSE/(float)(n_measurements)) << endl;

				if(newSSE < sse){ 
				
					// check stopping criteria rmse //
					stop = (sse/(float)(2*n_measurements) <= mse_th );
					sse = newSSE;

					// decrease lambda
					lc = max(minlc, lc/1.1f);
					lp = max(minlp, lp/1.1f);
					v = 1;

					break;
				}
				if(newSSE > sse){
					
					//increase lambda
					v *= 1.1f;
					lc = min(lc*v,maxlc);
					lp = min(lp*v,maxlp);
					
				}

				// stop this inner loop, if error decrease is too small; begin next outer loop with outer++
				if(abs(sse/newSSE - 1) < 1e-9f || innerabs > 10) {
					istop = true;
					//if(lp==maxlp) stop = true;
					break;
				}


			}while(!stop);

		}
		if(trace && (outer%10 == 0 || outer<10)){
			// timestamp
			timestamps.push_back(get_wall_time()-start_t);	

			// store sse
			if(show_min_sse)
				SSEs.push_back(decreased?newSSE:sse);
			else
				SSEs.push_back(newSSE);

			// write relative decrease in this iteration
			rel_decr.push_back(sse/initsse);

		}
	}

	double time_elapsed = (get_wall_time() - start_t);

	
	// get max memory //
	size_t free,total;
//	cuMemGetInfo(&free, &total);

	// print final results //
	cout << ".............................................." << endl;
	cout << "Optimization finished after " << time_elapsed  << "s \n";
	cout << "final RMSE: " << sqrt(sse/(mb)) << ", ";
	cout << "Number of outer iterations: "  << outer << endl;
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