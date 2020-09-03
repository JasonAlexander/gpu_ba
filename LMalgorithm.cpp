#include "LMalgorithm.h"
#include "MBATimer.h"
#include <iostream>
#include <iomanip>
#include <math.h>
#include <algorithm> 


LMalgorithm::LMalgorithm(int maxI, float stopcritGradient, float stopcritMSE, float stopdiff, float tauArg, float maxLa, float minLa, bool lmamode, bool backsubst, int epis, bool diagdamp, bool tracer
						 ,float fs, int imin, float imax, float hyb_sw, bool sms, bool print_prog)
{
	max_iterations = maxI;
	gradient_th = stopcritGradient;
	mse_th = stopcritMSE;
	tau = tauArg;
	maxL = maxLa;
	minL = minLa;
	diff_stop = stopdiff;
	schur = lmamode;				
	backsub = backsubst;			
	epi = epis;				
	trace = tracer;
	diagdamping = diagdamp;
	n_fs = fs;					
	cg_minI = imin;				
	cg_maxI = imax;				
	hyb_switch = hyb_sw;
	show_min_sse = sms;
	print_progress = print_prog;
}


bool LMalgorithm::runMinimizer(BAmode im, float*& camera_data, float*& point_data,
					float* measurements, int* ptidx, int* camidx, int n_measurements, int n_cameras, int n_points)
{

	int k = 0;                          // valid iteration counter
	int solved = 0;						// solved systems counter
	float v = 2;						// lambda manipulator term
	float gradMag = 0;					// gradient magnitude (J^T * error_vector)
	bool stop = false;					// termination variable for the loop
	float sse = 0;						// current Summed Squared Error
	float newSSE = 0;					// newly computed Summed Squared Error
	float oldsse = 0;
	int s = (im == metric)?12:18;		// number of camera parameters * 2
	int pv_num = 
		(n_points*3)+(n_cameras*s/2);	// parameter vector length	

	// for tracer
	vector<double> timestamps;
	vector<float> SSEs;
	vector<float> rel_decr;
	vector<float> d;
	vector<float> cg_iters;

	// Jacobian matrix J = [J_c, J_p] stored in a simplyfied BSR format  //
	int mb = n_measurements; 	
	int nb_C = n_cameras;
	int nb_P = n_points;

	// ensure constraints
	if(!schur){
		backsub = false;
		epi = 0;
	}

    const char* str = (im == metric)?"metric":"focal_radial";
	cout << "Start LMA minimizer with mode: " << str << endl;
	if(!schur)  cout << "Solving " << "full normal equation (hessian-matrix)" <<endl;
	if(schur && epi && backsub)  cout << "Solving " << "rcs-system (schur-matrix) with EPI and back-substitution" <<endl;
	if(schur && epi && !backsub) cout << "Solving " << "rcs-system (schur-matrix) with EPI and no back-substitution" <<endl;
	if(schur && !epi) cout << "Solving " << "schur-system without EPI" <<endl;

	// init cuda context and get free memory
	cudaFree(0);

	// place input data into device memory and init cuda  //
	int mem_used1 = CalcUtilsGPU::copyDataIntoDevice(camera_data, point_data, measurements, ptidx, camidx, mb, nb_C, nb_P, s, LMA, diagdamping);
	if(mem_used1 == -1){
		cout << "Error: Not enough device storage for input data, jacobian matrix and preconditionier matrix" <<  endl;
		return false;
	}
	// allocate solver specific data
	int mem_used2 = CalcUtilsGPU::initSolver(LMA, s, schur, epi);
	if(mem_used2 == -1){
		cout << "Error: Not enough device storage for algorithm specific temporal vectors " <<  endl;
		return false;
	}

	double start_t = get_wall_time();

	// calculate initial vector of errors for each measured point and sse //
  	CalcUtilsGPU::calcErrorVector_d(sse, 0, 0);
	float initsse = sse;
	float interSSE = 0;
	newSSE = sse;
	oldsse = sse;
	cout << "Initial MSE: " << (sse/(mb)) << endl;

	// define lambda //
	float minlc, minlp, maxlp, maxlc;
	float lc = (diagdamping)? tau: -tau;
	float lp = (diagdamping)? tau: -tau;

	// build initial Jacobian matrix part on device //
	if(hyb_switch<=0){
		CalcUtilsGPU::buildJacobian_d(-1, lc, lp, LMA, gradMag);
		CalcUtilsGPU::invHessDiagBlocks(lp);
	}else{
		CalcUtilsGPU::buildJacobian_d(-1, lc, lp, RI, gradMag);
	}
	minlc = (float)(lc/tau)*minL; //minimum lambda value to ensure numerical stability
	minlp = (float)(lp/tau)*minL; //minimum lambda value to ensure numerical stability

	maxlc = (float)(lc/tau)*maxL; //maximum lambda value to ensure numerical stability
	maxlp = (float)(lp/tau)*maxL; //maximum lambda value to ensure numerical stability


	// evaluate initial stop conditions
	stop = ( (gradMag/pv_num) <= gradient_th) || (sse/(2*n_measurements) <= mse_th );

	// initial resection-intersection //
	int h = 0;
	if(hyb_switch>0){
		h++;
		bool decreased = true;		
		while(true){
			if(h%2==1){ 
				if(h>1 && decreased){
						CalcUtilsGPU::buildJacobian_d(h%2, lc, lp, RI, gradMag); // build new system with new decreased lambda
				}
				CalcUtilsGPU::resection(lc);    // solve only for cam params
			}
			else{
				if(decreased){
					CalcUtilsGPU::buildJacobian_d(h%2, lc, lp, RI, gradMag); // build new system with new decreased lambda
				}
				CalcUtilsGPU::intersection(0,lp); // solve only for point params
			}
			newSSE = 0;
			decreased = CalcUtilsGPU::updateCheck(sse, newSSE, (h%2), false);
			// modify lambda
			if(newSSE > sse){

				if(lp==maxlp) {
					lc = tau; 
					lp = tau;
					break;
				}
				v *= 1.33333;
				lc = min(lc*v,maxlc);
				lp = min(lp*v,maxlp);
				h--;
				
			}else{

				if(trace){

					// timestamp				
					timestamps.push_back(get_wall_time()-start_t);				

					// store sse
					if(show_min_sse)
						SSEs.push_back(decreased?newSSE:sse);
					else
						SSEs.push_back(newSSE);

					// write relative decrease in this iteration
					rel_decr.push_back(newSSE/initsse);

					// did this iteration decrease the error?
					d.push_back(decreased);

					// how many cg iterations were performed?
					cg_iters.push_back(0);
				}
				lc = tau; 
				lp = tau;
				if((h%2==0 && abs(newSSE/(float)sse - 1) < hyb_switch)) break;
				sse = newSSE;
			}
			h++;
		}
		CalcUtilsGPU::buildJacobian_d(-1, lc, lp, LMA, gradMag);
		CalcUtilsGPU::invHessDiagBlocks(lp);
		v = 1;
		cout << "Hybrid: switched from RI to LMA after " << h << " iterations." << endl;
	}
	/////////////////////////////////////////////////////////////////////////

	int cgi = 0;		
	// start LM loop //
	while(!stop && solved < max_iterations){

		k++;

		// stop optimization if 10 consecutive iterations without error decrease
		if(k%10 == 0){
			if(oldsse<=sse) break;
			oldsse = sse;
		}

		// inner LMA loop , do until error gets lower  //
		do{	
			int decreased = 0;			

			if(!schur){
				// solve normal equation with pcg: get dp //
				cgi = CalcUtilsGPU::solvePCG_Hessian_d(lp, lp, gradMag, cg_maxI, cg_minI, n_fs);
			}else{
				// solve reduced camera system with pcg: get dp //	 
				cgi = CalcUtilsGPU::solvePCG_Schur_d(lp, lp, gradMag, cg_maxI, cg_minI, n_fs);
				if(backsub) 
					CalcUtilsGPU::backSubstitution();
			}

			// used for gain ratio calculations
			float ro = CalcUtilsGPU::calcro(lp, lp);

			// evaluate new parameter vector, after optionally applying EPIs
			if(epi > 0)
				decreased = CalcUtilsGPU::epi(backsub, lp, epi, sse, newSSE, interSSE);				
			else
				decreased = CalcUtilsGPU::updateCheck(sse, newSSE, -1, false);

			solved++;

			if(print_progress) cout << "iter: " << k << ", sse: " << ((float)sse/(float)(n_measurements)) << endl;

			if(trace){

				// timestamp				
				timestamps.push_back(get_wall_time()-start_t);				

				// store sse
				if(show_min_sse)
					SSEs.push_back(decreased?newSSE:sse);
				else
					SSEs.push_back(newSSE);

				// write relative decrease in this iteration
				rel_decr.push_back(newSSE/initsse);

				// did this iteration decrease the error?
				d.push_back(decreased);

				// how many cg iterations were performed?
				cg_iters.push_back(cgi);
			}

			// if error decreased ...  //
			if(decreased){ 

				// calculate gain ratio & decrease lambda for the next iteration
				if(epi <= 0){										
					ro = (sse-newSSE)/ro;
					float t = 1-((2*ro-1)*(2*ro-1)*(2*ro-1));
					lp *= (t<(0.3333333f))?(0.3333333f):t;
					lp = max(lp,minlp);

				}else{
					if(!backsub){
						lp *= 0.4f;
						lp = max(lp,minlp);
					}else{	
					
						ro = (sse-interSSE)/ro;
						float t = 1-((2*ro-1)*(2*ro-1)*(2*ro-1));
						if(ro<0) t = 2;
						lp *= (t<(0.3333333f))?(0.3333333f):t;
						lp = max(lp,minlp);
						
					}
				}

				
				// reset lambda manipulator				
				v = 2;

				// calculate new Jacobian Matrix and Gradient 	
				gradMag = 0;

				// linearize
				CalcUtilsGPU::buildJacobian_d(-1, lp, lp, LMA, gradMag);

				// invert Hessian block diagonals
				CalcUtilsGPU::invHessDiagBlocks(lp);		

				// check stopping criteria mse and gradient : sse and gradient magnitude are updated now //	
				stop = (sqrt(gradMag/(float)pv_num) <= gradient_th) || (newSSE/(float)(2*n_measurements) <= mse_th ) || abs(newSSE/(float)sse - 1) < diff_stop;
				sse = newSSE;
				break;

			}else{  // else raise damping term, raise lambda manipulator

				// increase lambda
				lp *= v;
				lc *= v;
				lp = min(lp,maxlp);
				lc = min(lc,maxlc);
				v *= 2;

				CalcUtilsGPU::formBdiagHessian(lp, lp, -1); // update Hessian block diagonals with new lambda explicitly
				CalcUtilsGPU::invHessDiagBlocks(lp); // invert Hessian block diagonals	

				// terminate due to small changes
				stop = (abs(sse/(float)newSSE - 1) < diff_stop) && sse != newSSE;

				// abort when there is no error decrease with max lambda
				if(lp==maxlp)
					stop=true;
				
			}

		}while(!stop && solved < max_iterations);

	}

	double time_elapsed = (get_wall_time() - start_t);

	// check device memory for sufficient space //
	size_t free,total;
//	cuMemGetInfo(&free, &total);

	// print final results //
	cout << ".............................................." << endl;
	cout << "Optimization finished after " << time_elapsed  << "s \n";
	cout << "final RMSE: " << sqrt(sse/(mb)) << ", ";
	cout << "Number of valid/total iterations: "  << k-1 << "/" << solved << endl;
	cout << ".............................................." << endl;
	cout << "Total Memory available                                       : " << setw(5) << total/1024/1024 << " mb \n";
	cout << "Used Memory for Jacobi/Preconditionier/Input-/Output-Vectors : " << setw(5) << mem_used1  << " mb \n";
	cout << "Used Memory for solver-specific additional temporal data     : " << setw(5) << mem_used2 << " mb \n";
	cout << ".............................................." << endl;

	if(trace){
		cout << endl;
		for(size_t i = 0; i< timestamps.size(); i++){
			printf( "t=%5.3fs, ", timestamps[i] );
			cout << "iter#" << left << setw(2) << (i+1);
			cout << ", cgi: "<< setw(2) << right << cg_iters[i];
			cout << ", MSE = " << setw(7) << left << setprecision(5) << ((float)SSEs[i]/(float)(n_measurements)) << " pix, ";
			cout << "rd: " << setw(9) << setprecision(4) << rel_decr[i] << "d?: " << d[i] << endl; 

		}
		cout << endl;	
	}

/*
	cout.imbue(locale(""));
	if(trace){
		cout << endl;
		for(size_t i = 0; i< timestamps.size(); i++){
			cout << timestamps[i] << ";";
			cout << setw(7) << left << setprecision(6) << ((float)SSEs[i]/(float)(n_measurements))<< endl;	
		}
		cout << endl;
	}
*/

	CalcUtilsGPU::wb_result(camera_data, point_data);

	return 0;
	
}


