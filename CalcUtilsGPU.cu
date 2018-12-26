#include "CalcUtilsGPU.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <windows.h>

#include "device_launch_parameters.h"

#define BLOCKSIZE 128


using namespace std;

// device input  //
float *d_cams;
float *d_points;
float *d_obs;
int *d_pidx;
int *d_cidx;
int n_obs, n_c, n_p;

// global variables //
int pv_num;		  // num of parameter vector entries
int ov_num;		  // num of measurement vector entries
int n_cam_params; // num of camera parameters * 2 (for each measurement dimension)
double prevdot;	  // temporal variable used for NCG
bool diagdamping; // use multiplicative (true) or additive damping (false)

// intermediate vector
float *d_err;        // error vector device pointer
float *d_val_C;		 // J_C vector device pointer
float *d_val_P;      // J_P vector device pointer
float *d_grad;       // gradient camera vector device pointer 
float *d_dP;		 // parameter correction vector device pointer
float *d_prec_c;	 // preconditioner for CG
float *d_prec_p;	 // preconditioner for CG
float *d_p;			 // NLCG  search direction
float *as;			 // temporal vector for gain ratio calculations

// intermediate parameter vector for temporal error function evaluations
float *Pc_new;
float *Pp_new;
float *grad_new;

// vectors for adding blocks after gpu reductions
float* d_parts;

// diagonal elements of the Hessian matrix
float * d_diagE;

//conjugate gradient temporal vectors 
float *s,*z,*p;

//... for Hessian mode
float *JtJp;

//... for Schur mode
float *Sp,*tc,*tp,*pv;


#pragma region init

int CalcUtilsGPU::checkCudaVersion()
{
	int device;     
	if( cudaGetDevice(&device) != cudaSuccess) return 0;
    cudaDeviceProp deviceProp;
   
	if(cudaGetDeviceProperties(&deviceProp, device) == cudaSuccess)
	{
		/* Statistics about the GPU device */
		printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
        deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

		int version = (deviceProp.major * 0x10 + deviceProp.minor);

		if (version < 0x30)
		{
			printf("%s: requires a minimum CUDA compute 3.0 capability\n", "ModernBA");

			cudaDeviceReset();
			return 0;
	    }
		
		return 1;
	}
	else 
		return 0;

}

size_t CalcUtilsGPU::copyDataIntoDevice(float *camera_data, float *point_data,
				float *measurements, int* ptidx, int* camidx, int NM, int NC, int NP, int ncp, Solver solv, bool diagdamp)
{	

	// input data //
	n_obs = NM;
	n_c = NC;
	n_p = NP;
	pv_num = (n_p*3)+(n_c*ncp/2);
	ov_num = 2*n_obs;
	n_cam_params = ncp;
	prevdot = 0;
	diagdamping = diagdamp;
	

	size_t used, total;
	cuMemGetInfo(&used, &total);

	vector<cudaError_t> err;
	// allocate input device data
	err.push_back(cudaMalloc((void**)&d_cams, n_c * sizeof(float) * 9 ));
	err.push_back(cudaMalloc((void**)&d_points, n_p * sizeof(float) * 3 ));
	err.push_back(cudaMalloc((void**)&d_obs, n_obs * sizeof(float) * 2));
	err.push_back(cudaMalloc((void**)&d_cidx, n_obs * sizeof(int)));
	err.push_back(cudaMalloc((void**)&d_pidx, n_obs * sizeof(int)));

	// allocate output device data
	err.push_back(cudaMalloc((void**)&d_err,sizeof(float)*ov_num));
	err.push_back(cudaMalloc((void**)&d_dP, sizeof(float)*pv_num));
	err.push_back(cudaMalloc((void**)&d_grad, sizeof(float)*pv_num));

	// allocate blockdiag of Hessian (Preconditioner) device data
	int tcp = (ncp==12)?21:45;
	err.push_back(cudaMalloc((void**)&d_prec_c, sizeof(float)*n_c*(ncp/2)*(ncp/2)));
	err.push_back(cudaMalloc((void**)&d_prec_p, sizeof(float)*n_p*6));	

	if(solv == LMA){
		//  stride for the val vector needed for 1D -> 2D access conversion  //
		size_t pitchC;
		size_t pitchP;
		err.push_back(cudaMallocPitch((void **)&d_val_C, &pitchC, n_obs * sizeof(float), n_cam_params));
		err.push_back(cudaMallocPitch((void **)&d_val_P, &pitchP, n_obs * sizeof(float), 6));
		err.push_back(cudaMalloc((void**)&as, sizeof(float)*pv_num));
	}

	size_t free;
	cuMemGetInfo(&free, &total);
	int mem_used = (used-free)/1024/1024;

	// copy to device
	cudaMemcpy(d_cams, camera_data, sizeof(float) * 9 * n_c, cudaMemcpyHostToDevice);
	cudaMemcpy(d_points, point_data, sizeof(float) * 3 * n_p, cudaMemcpyHostToDevice);
	cudaMemcpy(d_obs, measurements, sizeof(float) * 2 * n_obs, cudaMemcpyHostToDevice);
	cudaMemcpy(d_cidx, camidx, sizeof(int) * n_obs, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pidx, ptidx, sizeof(int) * n_obs, cudaMemcpyHostToDevice);

	// set used memory to 0 mb to indicate errors
	for(int i = 0; i< (int)err.size(); i++) 
		if(err[i] != cudaSuccess)
			mem_used = -1;

	return mem_used;

}

int CalcUtilsGPU::initSolver(Solver solv, int ncp, bool schur, int epi){

	vector<cudaError_t> err;

	int n_block_pv   = (int)( ( (float)(pv_num)/(float)BLOCKSIZE ) +1 );
	int n_block_pc   = (int)( ( (float)(n_c*n_cam_params/2)/(float)BLOCKSIZE ) +1 );
	int n_block_pp   = (int)( ( (float)(n_p*3)/(float)BLOCKSIZE ) +1 );
	int n_block_ovpv = (int)( ( (float)(ov_num + pv_num)/(float)BLOCKSIZE ) +1 ); 
	int n_block_opv	 = (int)( ( (float)(n_obs + pv_num)/(float)BLOCKSIZE ) +1 ); 
	int n_block_o    = (int)( ( (float)(ov_num)/(float)BLOCKSIZE ) +1 );

	size_t used, total;
	cuMemGetInfo(&used, &total);

	// intermediate parameter vector for temporal error function evaluations
	err.push_back(cudaMalloc((void**)&Pc_new, sizeof(float)*(9*n_c)));  
	err.push_back(cudaMalloc((void**)&Pp_new, sizeof(float)*(3*n_p)));	

	// vectors for adding blocks after gpu reductions
	err.push_back(cudaMalloc((void**)&d_parts, sizeof(float) * 32 ));

	//conjugate gradient temporal vectors 
	if(solv == LMA){

		if(diagdamping)
			err.push_back(cudaMalloc((void**)&d_diagE, sizeof(float)*(pv_num)));

		if(schur == true){
			err.push_back(cudaMalloc((void**)&s, sizeof(float)*(n_c*n_cam_params/2)));
			err.push_back(cudaMalloc((void**)&p, sizeof(float)*(n_c*n_cam_params/2)));
			err.push_back(cudaMalloc((void**)&z, sizeof(float)*(n_c*n_cam_params/2)));
			err.push_back(cudaMalloc((void**)&Sp, sizeof(float)*(n_c*n_cam_params/2)));
			err.push_back(cudaMalloc((void**)&tc, sizeof(float)*(n_c*n_cam_params/2)));
			err.push_back(cudaMalloc((void**)&tp, sizeof(float)*(n_p*3)));
			err.push_back(cudaMalloc((void**)&pv, sizeof(float)*(pv_num)));
			if(epi>0) err.push_back(cudaMalloc((void**)&grad_new, sizeof(float)*(pv_num)));
		}else{	
			err.push_back(cudaMalloc((void**)&s, sizeof(float)*(pv_num)));
			err.push_back(cudaMalloc((void**)&p, sizeof(float)*(pv_num)));
			err.push_back(cudaMalloc((void**)&z, sizeof(float)*(pv_num)));
			err.push_back(cudaMalloc((void**)&JtJp, sizeof(float)*(pv_num))); 
		}
	}

	// increase L1 Cache size for register spilling (for less shared memory)
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);	

	// nonlinear conjugate gradient variables
	if(solv == NCG){
		err.push_back(cudaMalloc((void**)&d_p, sizeof(float)*(pv_num)));
		err.push_back(cudaMalloc((void**)&z, sizeof(float)*(pv_num)));
		err.push_back(cudaMalloc((void**)&grad_new, sizeof(float)*(pv_num)));
	}

	size_t free;
	cuMemGetInfo(&free, &total);
	int mem_used = (used-free)/1024/1024;

	// set used memory to 0 mb to indicate errors
	for(int i = 0; i< (int)err.size(); i++) 
		if(err[i] != cudaSuccess)
			mem_used = -1;

	return mem_used;
}

#pragma endregion

#pragma region cuda kernels

__global__ void 
	calcErrorVector_kernel(float *errors, float *cams, float *points, float *obs, int *cidx, int *pidx, int NM, int NC, int NP){
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	if(idx>=NM) return;
	int c_idx = cidx[idx];
	int p_idx = pidx[idx];

	// rotation vector related to the measurement
	float r0 = cams[c_idx];
	float r1 = cams[NC+c_idx];
	float r2 = cams[2*NC+c_idx];

	// point related to the measurement
	float pxyz0 = points[0*NP+p_idx];
	float pxyz1 = points[1*NP+p_idx];
	float pxyz2 = points[2*NP+p_idx];


	float dot = r0*r0+r1*r1+r2*r2;
	float a = __fsqrt_rn(dot);

	float ct = a==0?0.5:__fdividef( (1.0 -__cosf(a)) , (a*a) );
    float st = a==0?1:( __fdividef(__sinf(a),a) );

	// rotated and translated point
	float XX0 =	   pxyz0*(1.0 - (r1*r1 + r2*r2)*ct) +  pxyz1*(r0*r1*ct - r2*st) + pxyz2*(r2*r0*ct + r1*st) + cams[3*NC+c_idx];
	float XX1 = -  (pxyz0*(r0*r1*ct + r2*st) +  pxyz1*(1.0 - (r2*r2 + r0*r0)*ct) + pxyz2*(r1*r2*ct - r0*st)) + cams[4*NC+c_idx];
	float XX2 = -  (pxyz0*(r2*r0*ct - r1*st) +  pxyz1*(r1*r2*ct + r0*st) + pxyz2*(1.0 - (r0*r0 + r1*r1)*ct)) + cams[5*NC+c_idx];
 
	// perspective division
	pxyz0 = __fdividef(XX0,XX2);
	pxyz1 = __fdividef(XX1,XX2);

	// calculate and apply distortions/ focal length
	float n2 = pxyz0*pxyz0 + pxyz1*pxyz1;
	float rr = 1.0f + n2*cams[7*NC+c_idx] + n2*n2*cams[8*NC+c_idx];
	float f = cams[6*NC+c_idx];

	// calculate partial differences //
	errors[idx] = (obs[idx] - rr*pxyz0*f);
    errors[idx+NM] = (obs[NM + idx] - rr*pxyz1*f);

}

template<int in, int out> __global__ void 
	__launch_bounds__(BLOCKSIZE,16)
	calcHp12_kernel(float *d_val_C, float *d_val_P, float *d_v, float *res, int *indC, int *indP, int nC, int nP, int N)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;


	if(idx >= N) return;
	
	float t0;
	float t1;

	int cidx = indC[idx];
	int pidx = indP[idx];

	float dvP0 = d_val_P[idx];
	float dvP1 = d_val_P[N + idx];
	float dvP2 = d_val_P[2*N + idx];
	float dvP3 = d_val_P[3*N + idx];
	float dvP4 = d_val_P[4*N + idx];
	float dvP5 = d_val_P[5*N + idx];

	float dC[5];
	dC[0] = d_val_C[idx];
	dC[1] = d_val_C[N + idx];
	__shared__ float s_dC[BLOCKSIZE*2];
	s_dC[threadIdx.x*2]  = d_val_C[2*N + idx];
	s_dC[threadIdx.x*2+1] = d_val_C[3*N + idx];

	// calculate J*p //
	if(in==-1){
		// input is a full parameter vector

		float dv0 = d_v[nC*6 + pidx];
		float dv1 = d_v[nC*6 + nP + pidx];
		float dv2 = d_v[nC*6 + nP*2 + pidx];

		float dvc0 = d_v[cidx];
		float dvc1 = d_v[nC + cidx];
		float dvc2 = d_v[nC*2 + cidx];
		float dvc3 = d_v[nC*3 + cidx];
		float dvc4 = d_v[nC*4 + cidx];
		float dvc5 = d_v[nC*5 + cidx];
				
		t0 = dC[0] * dvc0 + dC[1] * dvc1 + s_dC[threadIdx.x*2] * dvc2 + 
			 s_dC[threadIdx.x*2+1] * dvc3 + d_val_C[4*N + idx] * dvc4 + d_val_C[5*N + idx] * dvc5 + 
			 dvP0 * dv0 + dvP1 * dv1 + dvP2 * dv2;
		t1 = d_val_C[6*N + idx] * dvc0 + d_val_C[7*N + idx] * dvc1 + d_val_C[8*N + idx] * dvc2 + 
			 d_val_C[9*N + idx]  * dvc3 + d_val_C[10*N + idx] * dvc4 + d_val_C[11*N + idx] * dvc5 +
			 dvP3 * dv0 + dvP4 * dv1 + dvP5 * dv2;
	}

	if(in==0){
		// input is a point only parameter vector

		float dv0 = d_v[pidx];
		float dv1 = d_v[nP + pidx];
		float dv2 = d_v[nP*2 + pidx];

		
		t0 = dvP0 * dv0 + dvP1 * dv1 + dvP2 * dv2;
		t1 = dvP3 * dv0 + dvP4 * dv1 + dvP5 * dv2;
	}

	if(in==1){
		// input is a camera only parameter vector

		float dvc0 = d_v[cidx];
		float dvc1 = d_v[nC + cidx];
		float dvc2 = d_v[nC*2 + cidx];
		float dvc3 = d_v[nC*3 + cidx];
		float dvc4 = d_v[nC*4 + cidx];
		float dvc5 = d_v[nC*5 + cidx];

		
		t0 = dC[0] * dvc0 + dC[1] * dvc1 + s_dC[threadIdx.x*2] * dvc2 + 
			    s_dC[threadIdx.x*2+1] * dvc3 +  d_val_C[4*N + idx] * dvc4 + d_val_C[5*N + idx] * dvc5;
		t1 = d_val_C[6*N + idx] * dvc0 + d_val_C[7*N + idx] * dvc1 + d_val_C[8*N + idx] * dvc2 + 
			   d_val_C[9*N + idx] * dvc3 + d_val_C[10*N + idx] * dvc4 + d_val_C[11*N + idx] * dvc5;
	}

	// calculate Jt*J*p //

	if(out==-1){
		// output is a full parameter vector
		
		// M^T*v , M stored in row-major //		
		atomicAdd(&res[cidx],	     dC[0] * t0 + d_val_C[6*N + idx] * t1);
		atomicAdd(&res[nC + cidx],   dC[1] * t0 + d_val_C[7*N + idx] * t1);
		atomicAdd(&res[2*nC + cidx], s_dC[threadIdx.x*2] * t0 + d_val_C[8*N + idx] * t1);
		atomicAdd(&res[3*nC + cidx], s_dC[threadIdx.x*2+1] * t0 + d_val_C[9*N + idx] * t1);
		atomicAdd(&res[4*nC + cidx], d_val_C[4*N + idx] * t0 + d_val_C[10*N + idx] * t1);	
		atomicAdd(&res[5*nC + cidx], d_val_C[5*N + idx] * t0 + d_val_C[11*N + idx] * t1);

		// M^T*v , M stored in row-major //				
		atomicAdd(&res[6*nC + pidx], dvP0 * t0 + dvP3 * t1);
		atomicAdd(&res[6*nC + nP + pidx], dvP1 * t0 + dvP4 * t1);
		atomicAdd(&res[6*nC + 2*nP + pidx], dvP2 * t0 + dvP5 * t1);

	}

	if(out==0){
		// output is a point only parameter vector

		// M^T*v , M stored in row-major //
		atomicAdd(&res[pidx], dvP0 * t0 + dvP3 * t1);
		atomicAdd(&res[nP + pidx], dvP1 * t0 + dvP4 * t1);
		atomicAdd(&res[2*nP + pidx], dvP2 * t0 + dvP5 * t1);

	}

	if(out==1){
		// output is a camera only parameter vector

		// M^T*v , M stored in row-major //
		atomicAdd(&res[cidx],  dC[0] * t0 + d_val_C[6*N + idx] * t1);
		atomicAdd(&res[nC + cidx], dC[1] * t0 + d_val_C[7*N + idx] * t1);
		atomicAdd(&res[2*nC + cidx], s_dC[threadIdx.x*2] * t0 + d_val_C[8*N + idx] * t1);
		atomicAdd(&res[3*nC + cidx], s_dC[threadIdx.x*2+1] * t0 + d_val_C[9*N + idx] * t1);
		atomicAdd(&res[4*nC + cidx], d_val_C[4*N + idx] * t0 + d_val_C[10*N + idx] * t1);	
		atomicAdd(&res[5*nC + cidx], d_val_C[5*N + idx] * t0 + d_val_C[11*N + idx] * t1);

	}



}

template<int in, int out>  __global__ void 
	__launch_bounds__(BLOCKSIZE,16)
	calcHp18_kernel(float *d_val_C, float *d_val_P, float *d_v, float *res, int *indC, int *indP, int nC, int nP, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx >= N) return;

	float t[2];
	int cidx = indC[idx];
	int pidx = indP[idx];

	float dvP0 = d_val_P[idx];
	float dvP1 = d_val_P[N + idx];
	float dvP2 = d_val_P[2*N + idx];
	float dvP3 = d_val_P[3*N + idx];
	float dvP4 = d_val_P[4*N + idx];
	float dvP5 = d_val_P[5*N + idx];

	// calculate J*p //
	
	if(in==-1){
		// input is a full parameter vector

		float dv0 = d_v[nC*9 + pidx];
		float dv1 = d_v[nC*9 + nP + pidx];
		float dv2 = d_v[nC*9 + nP*2 + pidx];

		float dvc0 = d_v[cidx];
		float dvc1 = d_v[nC + cidx];
		float dvc2 = d_v[nC*2 + cidx];
		float dvc3 = d_v[nC*3 + cidx];
		float dvc4 = d_v[nC*4 + cidx];
		float dvc5 = d_v[nC*5 + cidx];
		float dvc6 = d_v[nC*6 + cidx];
		float dvc7 = d_v[nC*7 + cidx];
		float dvc8 = d_v[nC*8 + cidx];

		
		t[0]  = d_val_C[idx] * dvc0 + d_val_C[N + idx] * dvc1 + d_val_C[2*N + idx] * dvc2 + 
				d_val_C[3*N + idx] * dvc3 + d_val_C[4*N + idx] * dvc4 + d_val_C[5*N + idx] * dvc5 +
				d_val_C[6*N + idx] * dvc6 + d_val_C[7*N + idx] * dvc7 + d_val_C[8*N + idx] * dvc8 +
				dvP0 * dv0 + dvP1 * dv1 + dvP2 * dv2;

		t[1]  = d_val_C[9*N + idx]  * dvc0        + d_val_C[10*N + idx] * dvc1  + d_val_C[11*N + idx] * dvc2 + 
				d_val_C[12*N + idx] * dvc3 + d_val_C[13*N + idx] * dvc4 + d_val_C[14*N + idx] * dvc5 +
				d_val_C[15*N + idx] * dvc6 + d_val_C[16*N + idx] * dvc7 + d_val_C[17*N + idx] * dvc8 +
				dvP3 * dv0 + dvP4 * dv1 + dvP5 * dv2;
	}

	if(in==0){
		// input is a point only parameter vector

		float dv0 = d_v[pidx];
		float dv1 = d_v[nP + pidx];
		float dv2 = d_v[nP*2 + pidx];

		
		t[0] = dvP0 * dv0 + dvP1 * dv1 + dvP2 * dv2;
		t[1] = dvP3 * dv0 + dvP4 * dv1 + dvP5 * dv2;
	}

	if(in==1){
		// input is a camera only parameter vector

		float dvc0 = d_v[cidx];
		float dvc1 = d_v[nC + cidx];
		float dvc2 = d_v[nC*2 + cidx];
		float dvc3 = d_v[nC*3 + cidx];
		float dvc4 = d_v[nC*4 + cidx];
		float dvc5 = d_v[nC*5 + cidx];
		float dvc6 = d_v[nC*6 + cidx];
		float dvc7 = d_v[nC*7 + cidx];
		float dvc8 = d_v[nC*8 + cidx];
		
		t[0]  = d_val_C[idx] * dvc0 + d_val_C[N + idx] * dvc1 + d_val_C[2*N + idx] * dvc2 + 
				d_val_C[3*N + idx] * dvc3 + d_val_C[4*N + idx] * dvc4 + d_val_C[5*N + idx] * dvc5 +
				d_val_C[6*N + idx] * dvc6 + d_val_C[7*N + idx] * dvc7 + d_val_C[8*N + idx] * dvc8;

		t[1]  = d_val_C[9*N + idx]  * dvc0        + d_val_C[10*N + idx] * dvc1  + d_val_C[11*N + idx] * dvc2 + 
				d_val_C[12*N + idx] * dvc3 + d_val_C[13*N + idx] * dvc4 + d_val_C[14*N + idx] * dvc5 +
				d_val_C[15*N + idx] * dvc6 + d_val_C[16*N + idx] * dvc7 + d_val_C[17*N + idx] * dvc8;
	}

	// calculate Jt*J*p //

	if(out==-1){
		// output is a full parameter vector

		// M^T*v , M stored in row-major //
		atomicAdd(&res[cidx],        d_val_C[idx] *       t[0] + d_val_C[9*N + idx] *  t[1]);
		atomicAdd(&res[nC + cidx],   d_val_C[N + idx] *   t[0] + d_val_C[10*N + idx] * t[1]);
		atomicAdd(&res[2*nC + cidx], d_val_C[2*N + idx] * t[0] + d_val_C[11*N + idx] * t[1]);
		atomicAdd(&res[3*nC + cidx], d_val_C[3*N + idx] * t[0] + d_val_C[12*N + idx] * t[1]);
		atomicAdd(&res[4*nC + cidx], d_val_C[4*N + idx] * t[0] + d_val_C[13*N + idx] * t[1]);	
		atomicAdd(&res[5*nC + cidx], d_val_C[5*N + idx] * t[0] + d_val_C[14*N + idx] * t[1]);
		atomicAdd(&res[6*nC + cidx], d_val_C[6*N + idx] * t[0] + d_val_C[15*N + idx] * t[1]);
		atomicAdd(&res[7*nC + cidx], d_val_C[7*N + idx] * t[0] + d_val_C[16*N + idx] * t[1]);	
		atomicAdd(&res[8*nC + cidx], d_val_C[8*N + idx] * t[0] + d_val_C[17*N + idx] * t[1]);

		// M^T*v , M stored in row-major //
		atomicAdd(&res[9*nC + pidx],        dvP0 * t[0] + dvP3 * t[1]);
		atomicAdd(&res[9*nC + nP + pidx],   dvP1 * t[0] + dvP4 * t[1]);
		atomicAdd(&res[9*nC + 2*nP + pidx], dvP2 * t[0] + dvP5 * t[1]);

	}

	if(out==0){
		// output is a point only parameter vector

		// M^T*v , M stored in row-major //
		atomicAdd(&res[pidx],        dvP0 * t[0] + dvP3 * t[1]);
		atomicAdd(&res[nP + pidx],   dvP1 * t[0] + dvP4 * t[1]);
		atomicAdd(&res[2*nP + pidx], dvP2 * t[0] + dvP5 * t[1]);

	}

	if(out==1){
		// output is a camera only parameter vector

		// M^T*v , M stored in row-major //
		atomicAdd(&res[cidx],        d_val_C[idx] *       t[0] + d_val_C[9*N + idx] *  t[1]);
		atomicAdd(&res[nC + cidx],   d_val_C[N + idx] *   t[0] + d_val_C[10*N + idx] * t[1]);
		atomicAdd(&res[2*nC + cidx], d_val_C[2*N + idx] * t[0] + d_val_C[11*N + idx] * t[1]);
		atomicAdd(&res[3*nC + cidx], d_val_C[3*N + idx] * t[0] + d_val_C[12*N + idx] * t[1]);
		atomicAdd(&res[4*nC + cidx], d_val_C[4*N + idx] * t[0] + d_val_C[13*N + idx] * t[1]);	
		atomicAdd(&res[5*nC + cidx], d_val_C[5*N + idx] * t[0] + d_val_C[14*N + idx] * t[1]);
		atomicAdd(&res[6*nC + cidx], d_val_C[6*N + idx] * t[0] + d_val_C[15*N + idx] * t[1]);
		atomicAdd(&res[7*nC + cidx], d_val_C[7*N + idx] * t[0] + d_val_C[16*N + idx] * t[1]);	
		atomicAdd(&res[8*nC + cidx], d_val_C[8*N + idx] * t[0] + d_val_C[17*N + idx] * t[1]);
	}
	
}

__global__ void 
	calcVectorDot_kernel(const float* v1, const float* v2, int len, int b_len, float* res)
{
	int tid = threadIdx.x;

    __shared__ float block[BLOCKSIZE];
    int b_start = b_len * blockIdx.x;
    int start = b_start + threadIdx.x;
    int end   = min(len, b_start + b_len);

	// block-stride addition
    float temp = 0;
    for(int i = start; i < end; i += blockDim.x) 
		temp += (v1[i] * v2[i]);
    block[threadIdx.x] = temp;

    // parallel reduction
	for(unsigned int s = 1; s < BLOCKSIZE; s *=2){
		if(tid % (2*s) == 0){
			block[tid] += block[tid + s];
		}
		__syncthreads();
	}

    // store result
    if (tid  == 0) 
		res[blockIdx.x] = (block[0]);
}

__global__ void 
	scaleAdd_kernel(float *res, float *V1, float *V2, float alpha, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx >= N) return;

	// apply saxpy operation
	if(V2 != 0){		
		res[idx] = V1[idx]*alpha + V2[idx];
	}
	else{
		res[idx] = V1[idx]*alpha;
	}
/*	
	// check for NaN or Infinity
	if(isnan(r) || isinf(r)) 
		res[idx] = 0;
	else
		res[idx] = r;
*/	
}

__global__ void
	vector_mult(float *v1, float *v2, float *res, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) return;

	// elementwise vector multiplication
	res[idx] = v1[idx]*v2[idx];
}

__global__ void
	__launch_bounds__(BLOCKSIZE,12)
	linearize_kernelCams12(float *d_val_C, float *Pr_c, float *grad, float *err, float *cams, float *points, int *pidx, int *cidx, int NM, int NC, int NP, bool store_jac)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx>=NM) return;

	int c_idx = cidx[idx];
	int p_idx = pidx[idx];

	// rotation vector related to the measurement
	float r0 = cams[c_idx];
	float r1 = cams[NC+c_idx];
	float r2 = cams[2*NC+c_idx];

	// point related to the measurement
	//float pxyz[3];
	float pxyz0 = points[0*NP+p_idx];
	float pxyz1 = points[1*NP+p_idx];
	float pxyz2 = points[2*NP+p_idx];

	float m2 = r0*r0 + r1*r1 + r2*r2;
	float m = __fsqrt_rn(m2);  // Rodrigues magnitude = rotation angle
    float c = __cosf(m);
    float s = __sinf(m);
 
    // common trig terms
    float ct = __fdividef((1-c),m2);
    float st = __fdividef(s,m);
 
    // derivative coefficients for common trig terms
    float dct = __fdividef(__fdividef(s,m),m2);
    float dst = __fdividef(c,m2) - dct;
    dct -=  __fdividef(2*(1-c),(m2*m2));
 
    float utc = pxyz2*r0*r2 + pxyz1*r0*r1 - pxyz0*(r1*r1+r2*r2);
    float uts = pxyz2*r1 - pxyz1*r2;
    float vtc = -(pxyz0*r0*r1 + pxyz2*r1*r2 - pxyz1*(r0*r0+r2*r2));
    float vts = -(pxyz0*r2 - pxyz2*r0);
    float wtc = -(pxyz0*r0*r2 + pxyz1*r1*r2 - pxyz2*(r0*r0+r1*r1));
    float wts = -(pxyz1*r0 - pxyz0*r1);   

	float XX2 = ct*wtc + st*wts - pxyz2 + cams[5*NC+c_idx];

	// matrix derivate of xu w.r.t. XX	
	float f = cams[6*NC+c_idx];
    float dxu_dXX00 = f * __fdividef(1.0f,XX2); 			
	float dxu_dXX02 = f * __fdividef(-(ct*utc + st*uts + pxyz0 + cams[3*NC+c_idx]),(XX2*XX2));						
	float dxu_dXX12 = f * __fdividef(-(ct*vtc + st*vts - pxyz1 + cams[4*NC+c_idx]),(XX2*XX2));


	// calculate Jacobian block entry into register file //
	float J_c_reg0 = (ct*(r2*pxyz2 + r1*pxyz1) + r0*dct*utc + r0*dst*uts) *  dxu_dXX00 +
						 //(ct*(-r1*pxyz0 + 2*r0*pxyz1) + r0*dct*vtc + st*pxyz2 + r0*dst*vts) * dp_dxu01 * dxu_dXX00 + 
				     (ct*(-r2*pxyz0 + 2*r0*pxyz2) + r0*dct*wtc - st*pxyz1 + r0*dst*wts) *( dxu_dXX02);
	float J_c_reg1 = (ct*(r0*pxyz1 - 2*r1*pxyz0) + r1*dct*utc + st*pxyz2 + r1*dst*uts) *  dxu_dXX00 +
						 //(ct*(-r0*pxyz0 -   r2*pxyz2) + r1*dct*vtc + r1*dst*vts) * dp_dxu01 * dxu_dXX00 + 
					 (ct*(-r2*pxyz1 + 2*r1*pxyz2) + r1*dct*wtc + st*pxyz0 + r1*dst*wts) * (dxu_dXX02);
	float J_c_reg2 = (ct*(r0*pxyz2 - 2*r2*pxyz0) + r2*dct*utc - st*pxyz1 + r2*dst*uts) * dxu_dXX00 +
						 //(ct*(-r1*pxyz2 + 2*r2*pxyz1) + r2*dct*vtc - st*pxyz0 + r2*dst*vts) * dp_dxu01 * dxu_dXX00 + 
					 (ct*(-r0*pxyz0 -   r1*pxyz1) + r2*dct*wtc + r2*dst*wts) * (dxu_dXX02);
	float J_c_reg3 = dxu_dXX00;
	//float J_c_reg4 = 0;
	float J_c_reg5 = dxu_dXX02;
	float J_c_reg6 = //(ct*(r2*pxyz2 + r1*pxyz1) + r0*dct*utc + r0*dst*uts) * dp_dxu01 * dxu_dXX00 +
						 (ct*(-r1*pxyz0 + 2*r0*pxyz1) + r0*dct*vtc + st*pxyz2 + r0*dst*vts)  * dxu_dXX00 + 
						 (ct*(-r2*pxyz0 + 2*r0*pxyz2) + r0*dct*wtc - st*pxyz1 + r0*dst*wts) * (dxu_dXX12);
	float J_c_reg7 =  //(ct*(r0*pxyz1 - 2*r1*pxyz0) + r1*dct*utc + st*pxyz2 + r1*dst*uts) * dp_dxu01 * dxu_dXX00 +
						 (ct*(-r0*pxyz0 -   r2*pxyz2) + r1*dct*vtc + r1*dst*vts) * dxu_dXX00 + 
						 (ct*(-r2*pxyz1 + 2*r1*pxyz2) + r1*dct*wtc + st*pxyz0 + r1*dst*wts) * (dxu_dXX12);
	float J_c_reg8 = //(ct*(r0*pxyz2 - 2*r2*pxyz0) + r2*dct*utc - st*pxyz1 + r2*dst*uts) * dp_dxu01 * dxu_dXX00 +
						 (ct*(-r1*pxyz2 + 2*r2*pxyz1) + r2*dct*vtc - st*pxyz0 + r2*dst*vts) * dxu_dXX00 + 
						 (ct*(-r0*pxyz0 -   r1*pxyz1) + r2*dct*wtc + r2*dst*wts) * (dxu_dXX12) ;
	//float J_c_reg9 = 0;
	float J_c_reg10 = dxu_dXX00;
	float J_c_reg11 = dxu_dXX12;

	// store Jacobian?
	if(store_jac){

		d_val_C[0*NM + idx] = J_c_reg0;
		d_val_C[1*NM + idx] = J_c_reg1;
		d_val_C[2*NM + idx] = J_c_reg2;
		d_val_C[3*NM + idx] = J_c_reg3;
		d_val_C[4*NM + idx] = 0;
		d_val_C[5*NM + idx] = J_c_reg5;
		d_val_C[6*NM + idx] = J_c_reg6;
		d_val_C[7*NM + idx] = J_c_reg7;
		d_val_C[8*NM + idx] = J_c_reg8;
		d_val_C[9*NM + idx] = 0;
		d_val_C[10*NM + idx] = J_c_reg10;
		d_val_C[11*NM + idx] = J_c_reg11;
	}

	// calc+store Preconditionier
	atomicAdd(&Pr_c[(0) *NC + c_idx], J_c_reg0 * J_c_reg0 + J_c_reg6 * J_c_reg6);
	atomicAdd(&Pr_c[(1) *NC + c_idx], J_c_reg0 * J_c_reg1 + J_c_reg6 * J_c_reg7);
	atomicAdd(&Pr_c[(2) *NC + c_idx], J_c_reg0 * J_c_reg2 + J_c_reg6 * J_c_reg8);
	atomicAdd(&Pr_c[(3) *NC + c_idx], J_c_reg0 * J_c_reg3);
	atomicAdd(&Pr_c[(4) *NC + c_idx], J_c_reg6 * J_c_reg10);
	atomicAdd(&Pr_c[(5) *NC + c_idx], J_c_reg0 * J_c_reg5 + J_c_reg6 * J_c_reg11);

	atomicAdd(&Pr_c[(6) *NC + c_idx], J_c_reg1 * J_c_reg1 + J_c_reg7 * J_c_reg7);
	atomicAdd(&Pr_c[(7) *NC + c_idx], J_c_reg1 * J_c_reg2 + J_c_reg7 * J_c_reg8);
	atomicAdd(&Pr_c[(8) *NC + c_idx], J_c_reg1 * J_c_reg3);
	atomicAdd(&Pr_c[(9) *NC + c_idx], J_c_reg7 * J_c_reg10);
	atomicAdd(&Pr_c[(10) *NC + c_idx], J_c_reg1 * J_c_reg5 + J_c_reg7 * J_c_reg11);

	atomicAdd(&Pr_c[(11) *NC + c_idx], J_c_reg2 * J_c_reg2 + J_c_reg8 * J_c_reg8);
	atomicAdd(&Pr_c[(12) *NC + c_idx], J_c_reg2 * J_c_reg3);
	atomicAdd(&Pr_c[(13) *NC + c_idx], J_c_reg8 * J_c_reg10);
	atomicAdd(&Pr_c[(14) *NC + c_idx], J_c_reg2 * J_c_reg5 + J_c_reg8 * J_c_reg11);

	atomicAdd(&Pr_c[(15) *NC + c_idx], J_c_reg3 * J_c_reg3);
	//atomicAdd(&Pr_c[(16) *NC + c_idx], 0);
	atomicAdd(&Pr_c[(17) *NC + c_idx], J_c_reg3 * J_c_reg5);

	atomicAdd(&Pr_c[(18) *NC + c_idx],J_c_reg10 * J_c_reg10);
	atomicAdd(&Pr_c[(19) *NC + c_idx],J_c_reg10 * J_c_reg11);

	atomicAdd(&Pr_c[(20) *NC + c_idx], J_c_reg5 * J_c_reg5 + J_c_reg11 * J_c_reg11);

	// calc+store gradient
	float err0 = err[idx];
	float err1 = err[idx+NM];
	atomicAdd(&grad[c_idx],		   J_c_reg0 * err0 + J_c_reg6  * err1);
	atomicAdd(&grad[NC + c_idx],   J_c_reg1 * err0 + J_c_reg7  * err1);
	atomicAdd(&grad[2*NC + c_idx], J_c_reg2 * err0 + J_c_reg8  * err1);
	atomicAdd(&grad[3*NC + c_idx], J_c_reg3 * err0);
	atomicAdd(&grad[4*NC + c_idx], J_c_reg10* err1);	
	atomicAdd(&grad[5*NC + c_idx], J_c_reg5 * err0 + J_c_reg11 * err1);
}

__global__ void
	__launch_bounds__(BLOCKSIZE,12)
	linearize_kernelCams18(float *d_val_C, float *Pr_c, float *grad, float *err, float *cams, float *points, int *pidx, int *cidx, int NM, int NC, int NP, bool store_jac)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= NM) return;

	int c_idx = cidx[idx];
	int p_idx = pidx[idx];

	// rotation vector related to the measurement
	float r[3];
	r[0] = cams[c_idx];
	r[1] = cams[NC+c_idx];
	r[2] = cams[2*NC+c_idx];

	// point related to the measurement
	float pxyz[3];
	pxyz[0] = points[0*NP+p_idx];
	pxyz[1] = points[1*NP+p_idx];
	pxyz[2] = points[2*NP+p_idx];

	float m2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
	float m = __fsqrt_rn(m2);  // Rodrigues magnitude = rotation angle
    float c = __cosf(m);
    float s = __sinf(m);
 
    // common trig terms
    float ct = __fdividef((1-c),m2);
    float st = __fdividef(s,m);
 
    // derivative coefficients for common trig terms
    float dct = __fdividef(__fdividef(s,m),m2);
    float dst = __fdividef(c,m2) - dct;
    dct -=  __fdividef(2*(1-c),(m2*m2));
 
    float utc = pxyz[2]*r[0]*r[2] + pxyz[1]*r[0]*r[1] - pxyz[0]*(r[1]*r[1]+r[2]*r[2]);
    float uts = pxyz[2]*r[1] - pxyz[1]*r[2];
    float vtc = -(pxyz[0]*r[0]*r[1] + pxyz[2]*r[1]*r[2] - pxyz[1]*(r[0]*r[0]+r[2]*r[2]));
    float vts = -(pxyz[0]*r[2] - pxyz[2]*r[0]);
    float wtc = -(pxyz[0]*r[0]*r[2] + pxyz[1]*r[1]*r[2] - pxyz[2]*(r[0]*r[0]+r[1]*r[1]));
    float wts = -(pxyz[1]*r[0] - pxyz[0]*r[1]);   

	// rotate vector (Rodridues) + translate
	float XX0 = ct*utc + st*uts + pxyz[0] + cams[3*NC+c_idx];
	float XX1 = ct*vtc + st*vts - pxyz[1] + cams[4*NC+c_idx];
	float XX2 = ct*wtc + st*wts - pxyz[2] + cams[5*NC+c_idx];

	// perspective division
	float xu[2]; 
	xu[0] = __fdividef(XX0,XX2);
	xu[1] = __fdividef(XX1,XX2);

	// matrix derivate of xu w.r.t. XX	
    float dxu_dXX00 = __fdividef(1.0f,XX2); 			
	float dxu_dXX02 = __fdividef(-XX0,(XX2*XX2));						
	float dxu_dXX12 = __fdividef(-XX1,(XX2*XX2));

	// matrix derivate of xd w.r.t. xu
	float f = cams[6*NC+c_idx];
	float n2 = xu[0]*xu[0] + xu[1]*xu[1];
	float rr = 1.0f + n2*cams[7*NC+c_idx] + n2*n2*cams[8*NC+c_idx];
	float dr = 2*cams[7*NC+c_idx] + 4*cams[8*NC+c_idx]*n2;

	float dp_dxu00 = (rr + xu[0] * xu[0] * dr)*f;
	float dp_dxu01 = xu[0] * xu[1] * dr * f;
	float dp_dxu11 = (rr + xu[1] * xu[1] * dr)*f;

	// calculate Jacobian block entry into register file //
	float J_c_reg[18];

	J_c_reg[0] = (ct*(r[2]*pxyz[2] + r[1]*pxyz[1]) + r[0]*dct*utc + r[0]*dst*uts) * dp_dxu00 * dxu_dXX00 +
						 (ct*(-r[1]*pxyz[0] + 2*r[0]*pxyz[1]) + r[0]*dct*vtc + st*pxyz[2] + r[0]*dst*vts) * dp_dxu01 * dxu_dXX00 + 
						 (ct*(-r[2]*pxyz[0] + 2*r[0]*pxyz[2]) + r[0]*dct*wtc - st*pxyz[1] + r[0]*dst*wts) *( dxu_dXX02*dp_dxu00 + dxu_dXX12*dp_dxu01);
	J_c_reg[1] = (ct*(r[0]*pxyz[1] - 2*r[1]*pxyz[0]) + r[1]*dct*utc + st*pxyz[2] + r[1]*dst*uts) * dp_dxu00 * dxu_dXX00 +
						 (ct*(-r[0]*pxyz[0] -   r[2]*pxyz[2]) + r[1]*dct*vtc + r[1]*dst*vts) * dp_dxu01 * dxu_dXX00 + 
						 (ct*(-r[2]*pxyz[1] + 2*r[1]*pxyz[2]) + r[1]*dct*wtc + st*pxyz[0] + r[1]*dst*wts) * (dxu_dXX02*dp_dxu00 + dxu_dXX12*dp_dxu01);
	J_c_reg[2] = (ct*(r[0]*pxyz[2] - 2*r[2]*pxyz[0]) + r[2]*dct*utc - st*pxyz[1] + r[2]*dst*uts) * dp_dxu00 * dxu_dXX00 +
						 (ct*(-r[1]*pxyz[2] + 2*r[2]*pxyz[1]) + r[2]*dct*vtc - st*pxyz[0] + r[2]*dst*vts) * dp_dxu01 * dxu_dXX00 + 
						 (ct*(-r[0]*pxyz[0] -   r[1]*pxyz[1]) + r[2]*dct*wtc + r[2]*dst*wts) * (dxu_dXX02*dp_dxu00 + dxu_dXX12*dp_dxu01);
	J_c_reg[3] = dp_dxu00 * dxu_dXX00;
	J_c_reg[4] = dp_dxu01 * dxu_dXX00;
	J_c_reg[5] = dxu_dXX02 * dp_dxu00 + dxu_dXX12 * dp_dxu01;
	J_c_reg[6] = rr*xu[0];
	J_c_reg[7] = xu[0]*n2*f;
	J_c_reg[8] = xu[0]*n2*n2*f;

	J_c_reg[9] = (ct*(r[2]*pxyz[2] + r[1]*pxyz[1]) + r[0]*dct*utc + r[0]*dst*uts) * dp_dxu01 * dxu_dXX00 +
						 (ct*(-r[1]*pxyz[0] + 2*r[0]*pxyz[1]) + r[0]*dct*vtc + st*pxyz[2] + r[0]*dst*vts) * dp_dxu11 * dxu_dXX00 + 
						 (ct*(-r[2]*pxyz[0] + 2*r[0]*pxyz[2]) + r[0]*dct*wtc - st*pxyz[1] + r[0]*dst*wts) * (dxu_dXX02*dp_dxu01 + dxu_dXX12*dp_dxu11);
	J_c_reg[10] =  (ct*(r[0]*pxyz[1] - 2*r[1]*pxyz[0]) + r[1]*dct*utc + st*pxyz[2] + r[1]*dst*uts) * dp_dxu01 * dxu_dXX00 +
						 (ct*(-r[0]*pxyz[0] -   r[2]*pxyz[2]) + r[1]*dct*vtc + r[1]*dst*vts) * dp_dxu11 * dxu_dXX00 + 
						 (ct*(-r[2]*pxyz[1] + 2*r[1]*pxyz[2]) + r[1]*dct*wtc + st*pxyz[0] + r[1]*dst*wts) * (dxu_dXX02*dp_dxu01 + dxu_dXX12*dp_dxu11);
	J_c_reg[11] = (ct*(r[0]*pxyz[2] - 2*r[2]*pxyz[0]) + r[2]*dct*utc - st*pxyz[1] + r[2]*dst*uts) * dp_dxu01 * dxu_dXX00 +
						 (ct*(-r[1]*pxyz[2] + 2*r[2]*pxyz[1]) + r[2]*dct*vtc - st*pxyz[0] + r[2]*dst*vts) * dp_dxu11 * dxu_dXX00 + 
						 (ct*(-r[0]*pxyz[0] -   r[1]*pxyz[1]) + r[2]*dct*wtc + r[2]*dst*wts) * (dxu_dXX02*dp_dxu01 + dxu_dXX12*dp_dxu11) ;
	J_c_reg[12] = dp_dxu01 * dxu_dXX00;
	J_c_reg[13] = dp_dxu11 * dxu_dXX00;
	J_c_reg[14] = dxu_dXX02 * dp_dxu01 + dxu_dXX12 * dp_dxu11;
	J_c_reg[15] = rr*xu[1];
	J_c_reg[16] = xu[1]*n2*f;
	J_c_reg[17] = xu[1]*n2*n2*f; 

	// store Jacobian?
	if(store_jac){

		d_val_C[0*NM + idx] = J_c_reg[0];
		d_val_C[1*NM + idx] = J_c_reg[1];
		d_val_C[2*NM + idx] = J_c_reg[2];
		d_val_C[3*NM + idx] = J_c_reg[3];
		d_val_C[4*NM + idx] = J_c_reg[4];
		d_val_C[5*NM + idx] = J_c_reg[5];
		d_val_C[6*NM + idx] = J_c_reg[6];
		d_val_C[7*NM + idx] = J_c_reg[7];
		d_val_C[8*NM + idx] = J_c_reg[8];

		d_val_C[9*NM + idx] = J_c_reg[9];
		d_val_C[10*NM + idx] = J_c_reg[10];
		d_val_C[11*NM + idx] = J_c_reg[11];
		d_val_C[12*NM + idx] = J_c_reg[12];
		d_val_C[13*NM + idx] = J_c_reg[13];
		d_val_C[14*NM + idx] = J_c_reg[14];
		d_val_C[15*NM + idx] = J_c_reg[15]; 
		d_val_C[16*NM + idx] = J_c_reg[16];
		d_val_C[17*NM + idx] = J_c_reg[17];
	}

	// calculate/store preconditioner
	int co = 0;
	for (int i = 0; i < 9; i++) {
		for (int j = i; j < 9; j++) {
			float temp = 0;
			for (int inner = 0; inner < 2; inner++) {
               
				temp += J_c_reg[(i+inner*9)] * J_c_reg[(j+inner*9)];				
			}

			atomicAdd(&Pr_c[(co) *NC + c_idx], temp);
			co++;
		}
	}
	// calculate + store gradient
	float err0 = err[idx];
	float err1 = err[idx+NM];
	atomicAdd(&grad[c_idx],        J_c_reg[0] * err0 + J_c_reg[9] * err1);
	atomicAdd(&grad[NC + c_idx],   J_c_reg[1] * err0 + J_c_reg[10] * err1);
	atomicAdd(&grad[2*NC + c_idx], J_c_reg[2] * err0 + J_c_reg[11] * err1);
	atomicAdd(&grad[3*NC + c_idx], J_c_reg[3] * err0 + J_c_reg[12] * err1);
	atomicAdd(&grad[4*NC + c_idx], J_c_reg[4] * err0 + J_c_reg[13] * err1);	
	atomicAdd(&grad[5*NC + c_idx], J_c_reg[5] * err0 + J_c_reg[14] * err1);
	atomicAdd(&grad[6*NC + c_idx], J_c_reg[6] * err0 + J_c_reg[15] * err1);
	atomicAdd(&grad[7*NC + c_idx], J_c_reg[7] * err0 + J_c_reg[16] * err1);	
	atomicAdd(&grad[8*NC + c_idx], J_c_reg[8] * err0 + J_c_reg[17] * err1);

}

__global__ void
	linearize_kernelPoints18(float *d_val_P, float *Pr_p, float *grad, float *err, float *cams, float *points, int *pidx, int *cidx, int NM, int NC, int NP, bool store_jac)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= NM) return;

	int c_idx = cidx[idx];
	int p_idx = pidx[idx];
	// compute Jacobian sub matrices J_c[i,j] and J_p[i,j]  //

	// rotation vector related to the measurement
	float r[3];
	r[0] = cams[c_idx];
	r[1] = cams[NC+c_idx];
	r[2] = cams[2*NC+c_idx];

	float a = __fsqrt_rn(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
    float ct = a==0?0.5:__fdividef((1.0-__cosf(a)),(a*a));
    float st = a==0?1:__fdividef(__sinf(a),a);

	// rotation matrix entries
    float R00 =float(1.0 - (r[1]*r[1] + r[2]*r[2])*ct);
    float R01 =float(r[0]*r[1]*ct - r[2]*st);
    float R02 =float(r[2]*r[0]*ct + r[1]*st);
    float R10 =-float(r[0]*r[1]*ct + r[2]*st);
    float R11 =-float(1.0 - (r[2]*r[2] + r[0]*r[0])*ct);
    float R12 =-float(r[1]*r[2]*ct - r[0]*st);
    float R20 =-float(r[2]*r[0]*ct - r[1]*st);
    float R21 =-float(r[1]*r[2]*ct + r[0]*st);
    float R22 =-float(1.0 - (r[0]*r[0] + r[1]*r[1])*ct);

	// point related to the measurement
	float pxyz[3];
	pxyz[0] = points[0*NP+p_idx];
	pxyz[1] = points[1*NP+p_idx];
	pxyz[2] = points[2*NP+p_idx];
	
	// apply rotation + translation
	float XX[3];
	XX[0] = R00 * pxyz[0] + R01 * pxyz[1] + R02 * pxyz[2] + cams[3*NC+c_idx];
	XX[1] = R10 * pxyz[0] + R11 * pxyz[1] + R12 * pxyz[2] + cams[4*NC+c_idx];
	XX[2] = R20 * pxyz[0] + R21 * pxyz[1] + R22 * pxyz[2] + cams[5*NC+c_idx];

	// perspective division
	float xu[2]; 
	xu[0] = __fdividef(XX[0],XX[2]);
	xu[1] = __fdividef(XX[1],XX[2]);

	// matrix derivate of xu w.r.t. XX	
    float dxu_dXX00 = __fdividef(1.0f,XX[2]); 			
	float dxu_dXX02 = __fdividef(-XX[0],(XX[2]*XX[2]));						
	float dxu_dXX12 = __fdividef(-XX[1],(XX[2]*XX[2]));

	// matrix derivate of xd w.r.t. xu	
	float f = cams[6*NC+c_idx];
	float n2 = xu[0]*xu[0] + xu[1]*xu[1];
	float rr = 1.0f + n2*cams[7*NC+c_idx] + n2*n2*cams[8*NC+c_idx];
	float dr = 2*cams[7*NC+c_idx] + 4*cams[8*NC+c_idx]*n2;

	float dp_dxu[3];
	dp_dxu[0] = (rr + xu[0] * xu[0] * dr)*f;
	dp_dxu[1] = xu[0] * xu[1] * dr * f;
	dp_dxu[2] = (rr + xu[1] * xu[1] * dr)*f;

	// calculate Jacobian block entry into register file //
	float J_p_reg0 = R00 * dp_dxu[0] * dxu_dXX00 + R10 * dp_dxu[1] * dxu_dXX00 + 
					       R20 * (dxu_dXX02*dp_dxu[0] + dxu_dXX12*dp_dxu[1]);
	float J_p_reg1 = R01 * dp_dxu[0] * dxu_dXX00 + R11 * dp_dxu[1] * dxu_dXX00 + 
						   R21 * (dxu_dXX02*dp_dxu[0] + dxu_dXX12*dp_dxu[1]);
	float J_p_reg2 = R02 * dp_dxu[0] * dxu_dXX00 + R12 * dp_dxu[1] * dxu_dXX00 + 
						   R22 * (dxu_dXX02*dp_dxu[0] + dxu_dXX12*dp_dxu[1]);
	float J_p_reg3 = R00 * dp_dxu[1] * dxu_dXX00 + R10 * dp_dxu[2] * dxu_dXX00 + 
						   R20 * (dxu_dXX02*dp_dxu[1] + dxu_dXX12*dp_dxu[2]);
	float J_p_reg4 = R01 * dp_dxu[1] * dxu_dXX00 + R11 * dp_dxu[2] * dxu_dXX00 + 
						   R21 * (dxu_dXX02*dp_dxu[1] + dxu_dXX12*dp_dxu[2]);
	float J_p_reg5 = R02 * dp_dxu[1] * dxu_dXX00 + R12 * dp_dxu[2] * dxu_dXX00 + 
						   R22 * (dxu_dXX02*dp_dxu[1] + dxu_dXX12*dp_dxu[2]);

	// store Jacobian in DRAM?
	if(store_jac){
		d_val_P[idx] =  J_p_reg0;
		d_val_P[NM + idx] =  J_p_reg1;
		d_val_P[2*NM + idx] =  J_p_reg2;
		d_val_P[3*NM + idx] =  J_p_reg3;
		d_val_P[4*NM + idx] =  J_p_reg4;
		d_val_P[5*NM + idx] =  J_p_reg5;
	}
				
	// calculate + store Preconditioner	
	atomicAdd(&Pr_p[(0) *NP + p_idx], J_p_reg0 * J_p_reg0 + J_p_reg3 * J_p_reg3);
	atomicAdd(&Pr_p[(1) *NP + p_idx], J_p_reg0 * J_p_reg1 + J_p_reg3 * J_p_reg4);
	atomicAdd(&Pr_p[(2) *NP + p_idx], J_p_reg0 * J_p_reg2 + J_p_reg3 * J_p_reg5);
	atomicAdd(&Pr_p[(3) *NP + p_idx], J_p_reg1 * J_p_reg1 + J_p_reg4 * J_p_reg4);
	atomicAdd(&Pr_p[(4) *NP + p_idx], J_p_reg1 * J_p_reg2 + J_p_reg4 * J_p_reg5);
	atomicAdd(&Pr_p[(5) *NP + p_idx], J_p_reg2 * J_p_reg2 + J_p_reg5 * J_p_reg5);

	// calculate + store gradient
	float err0 = err[idx];
	float err1 = err[idx+NM];
	atomicAdd(&grad[p_idx],        J_p_reg0 * err0 + J_p_reg3 * err1);
	atomicAdd(&grad[NP + p_idx],   J_p_reg1 * err0 + J_p_reg4 * err1);
	atomicAdd(&grad[2*NP + p_idx], J_p_reg2 * err0 + J_p_reg5 * err1);

}

__global__ void
	linearize_kernelPoints12(float *d_val_P, float *Pr_p, float *grad, float *err, float *cams, float *points, int *pidx, int *cidx, int NM, int NC, int NP, bool store_jac)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= NM) return;

	int c_idx = cidx[idx];
	int p_idx = pidx[idx];
	// compute Jacobian sub matrices J_c[i,j] and J_p[i,j]  //

	// rotation vector related to the measurement
	float r0 = cams[c_idx];
	float r1 = cams[NC+c_idx];
	float r2 = cams[2*NC+c_idx];

	float a = __fsqrt_rn(r0*r0+r1*r1+r2*r2);
    float ct = a==0?0.5:__fdividef((1.0-__cosf(a)),(a*a));
    float st = a==0?1:__fdividef(__sinf(a),a);

	// rotation matrix entries
    float R00 =float(1.0 - (r1*r1 + r2*r2)*ct);
    float R01 =float(r0*r1*ct - r2*st);
    float R02 =float(r2*r0*ct + r1*st);
    float R10 =-float(r0*r1*ct + r2*st);
    float R11 =-float(1.0 - (r2*r2 + r0*r0)*ct);
    float R12 =-float(r1*r2*ct - r0*st);
    float R20 =-float(r2*r0*ct - r1*st);
    float R21 =-float(r1*r2*ct + r0*st);
    float R22 =-float(1.0 - (r0*r0 + r1*r1)*ct);

	// point related to the measurement
	float pxyz[3];
	pxyz[0] = points[0*NP+p_idx];
	pxyz[1] = points[1*NP+p_idx];
	pxyz[2] = points[2*NP+p_idx];
	
	// apply rotation + translation
	//float XX[3];
	//float XX0 = R00 * pxyz[0] + R01 * pxyz[1] + R02 * pxyz[2] + cams[3*NC+c_idx];
	//float XX1 = R10 * pxyz[0] + R11 * pxyz[1] + R12 * pxyz[2] + cams[4*NC+c_idx];
	float XX2 = R20 * pxyz[0] + R21 * pxyz[1] + R22 * pxyz[2] + cams[5*NC+c_idx];

	float f = cams[6*NC+c_idx];
	// matrix derivate of xu w.r.t. XX	
    float dxu_dXX00 = f * __fdividef(1.0f,XX2); 			
	float dxu_dXX02 = f * __fdividef(-(R00 * pxyz[0] + R01 * pxyz[1] + R02 * pxyz[2] + cams[3*NC+c_idx]),(XX2*XX2));						
	float dxu_dXX12 = f * __fdividef(-(R10 * pxyz[0] + R11 * pxyz[1] + R12 * pxyz[2] + cams[4*NC+c_idx]),(XX2*XX2));

	// calculate Jacobian block entry into register file //
	float J_p_reg0 = R00 * dxu_dXX00 + R20 * (dxu_dXX02);
	float J_p_reg1 = R01 * dxu_dXX00 + R21 * (dxu_dXX02);
	float J_p_reg2 = R02 * dxu_dXX00 + R22 * (dxu_dXX02);
	float J_p_reg3 = R10 * dxu_dXX00 + R20 * (dxu_dXX12);
	float J_p_reg4 = R11 * dxu_dXX00 + R21 * (dxu_dXX12);
	float J_p_reg5 = R12 * dxu_dXX00 + R22 * (dxu_dXX12);
				
	// calculate + store Preconditioner	
	atomicAdd(&Pr_p[(0) *NP + p_idx], J_p_reg0 * J_p_reg0 + J_p_reg3 * J_p_reg3);
	atomicAdd(&Pr_p[(1) *NP + p_idx], J_p_reg0 * J_p_reg1 + J_p_reg3 * J_p_reg4);
	atomicAdd(&Pr_p[(2) *NP + p_idx], J_p_reg0 * J_p_reg2 + J_p_reg3 * J_p_reg5);
	atomicAdd(&Pr_p[(3) *NP + p_idx], J_p_reg1 * J_p_reg1 + J_p_reg4 * J_p_reg4);
	atomicAdd(&Pr_p[(4) *NP + p_idx], J_p_reg1 * J_p_reg2 + J_p_reg4 * J_p_reg5);
	atomicAdd(&Pr_p[(5) *NP + p_idx], J_p_reg2 * J_p_reg2 + J_p_reg5 * J_p_reg5);

	// calculate + store gradient
	float err0 = err[idx];
	float err1 = err[idx+NM];
	atomicAdd(&grad[p_idx],        J_p_reg0 * err0 + J_p_reg3 * err1);
	atomicAdd(&grad[NP + p_idx],   J_p_reg1 * err0 + J_p_reg4 * err1);
	atomicAdd(&grad[2*NP + p_idx], J_p_reg2 * err0 + J_p_reg5 * err1);

	// store Jacobian in DRAM?
	if(store_jac){
		d_val_P[idx] =  J_p_reg0;
		d_val_P[NM + idx] =  J_p_reg1;
		d_val_P[2*NM + idx] =  J_p_reg2;
		d_val_P[3*NM + idx] =  J_p_reg3;
		d_val_P[4*NM + idx] =  J_p_reg4;
		d_val_P[5*NM + idx] =  J_p_reg5;
	}
}

__global__ void
	blockDiag_kernelCams12(float *J_c, float *Pr_c,  int *cidx, int N, int n_c)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) return;

		// fetch Jac into register file //
		float J_c_reg0 = J_c[idx];
		float J_c_reg1 = J_c[N + idx];
		float J_c_reg2 = J_c[2*N + idx];
		float J_c_reg3 = J_c[3*N + idx];
		float J_c_reg4 = J_c[4*N + idx];
		float J_c_reg5 = J_c[5*N + idx];
		float J_c_reg6 = J_c[6*N + idx];
		float J_c_reg7 = J_c[7*N + idx];
		float J_c_reg8 = J_c[8*N + idx];
		float J_c_reg9 = J_c[9*N + idx];
		float J_c_reg10 = J_c[10*N + idx];
		float J_c_reg11 = J_c[11*N + idx];

		// calculate H_ij = J_ij^T * J_ij
		int id = cidx[idx];

			atomicAdd(&Pr_c[(0) *n_c + id], J_c_reg0 * J_c_reg0 + J_c_reg6 * J_c_reg6);
			atomicAdd(&Pr_c[(1) *n_c + id], J_c_reg0 * J_c_reg1 + J_c_reg6 * J_c_reg7);
			atomicAdd(&Pr_c[(2) *n_c + id], J_c_reg0 * J_c_reg2 + J_c_reg6 * J_c_reg8);
			atomicAdd(&Pr_c[(3) *n_c + id], J_c_reg0 * J_c_reg3 + J_c_reg6 * J_c_reg9);
			atomicAdd(&Pr_c[(4) *n_c + id], J_c_reg0 * J_c_reg4 + J_c_reg6 * J_c_reg10);
			atomicAdd(&Pr_c[(5) *n_c + id], J_c_reg0 * J_c_reg5 + J_c_reg6 * J_c_reg11);

			atomicAdd(&Pr_c[(6) *n_c + id], J_c_reg1 * J_c_reg1 + J_c_reg7 * J_c_reg7);
			atomicAdd(&Pr_c[(7) *n_c + id], J_c_reg1 * J_c_reg2 + J_c_reg7 * J_c_reg8);
			atomicAdd(&Pr_c[(8) *n_c + id], J_c_reg1 * J_c_reg3 + J_c_reg7 * J_c_reg9);
			atomicAdd(&Pr_c[(9) *n_c + id], J_c_reg1 * J_c_reg4 + J_c_reg7 * J_c_reg10);
			atomicAdd(&Pr_c[(10) *n_c + id], J_c_reg1 * J_c_reg5 + J_c_reg7 * J_c_reg11);


			atomicAdd(&Pr_c[(11) *n_c + id], J_c_reg2 * J_c_reg2 + J_c_reg8 * J_c_reg8);
			atomicAdd(&Pr_c[(12) *n_c + id], J_c_reg2 * J_c_reg3 + J_c_reg8 * J_c_reg9);
			atomicAdd(&Pr_c[(13) *n_c + id], J_c_reg2 * J_c_reg4 + J_c_reg8 * J_c_reg10);
			atomicAdd(&Pr_c[(14) *n_c + id], J_c_reg2 * J_c_reg5 + J_c_reg8 * J_c_reg11);

			atomicAdd(&Pr_c[(15) *n_c + id], J_c_reg3 * J_c_reg3 + J_c_reg9 * J_c_reg9);
			atomicAdd(&Pr_c[(16) *n_c + id], J_c_reg3 * J_c_reg4 + J_c_reg9 * J_c_reg10);
			atomicAdd(&Pr_c[(17) *n_c + id], J_c_reg3 * J_c_reg5 + J_c_reg9 * J_c_reg11);


			atomicAdd(&Pr_c[(18) *n_c + id], J_c_reg4 * J_c_reg4 + J_c_reg10 * J_c_reg10);
			atomicAdd(&Pr_c[(19) *n_c + id], J_c_reg4 * J_c_reg5 + J_c_reg10 * J_c_reg11);


			atomicAdd(&Pr_c[(20) *n_c + id], J_c_reg5 * J_c_reg5 + J_c_reg11 * J_c_reg11);

	
}

__global__ void
	blockDiag_kernelCams18(float *J_c, float *Pr_c,  int *cidx, int N, int n_c)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) return;

		// fetch Jac into register file //
		float J_c_reg[18];
		for(int i = 0; i < 18 ; i++){
			J_c_reg[i] = J_c[i*N + idx];
		}

		int id = cidx[idx];

		// calculate H_ij = J_ij^T * J_ij
		int co = 0;
		for (int i = 0; i < 9; i++) {
			for (int j = i; j < 9; j++) {
				float temp = 0;
				for (int inner = 0; inner < 2; inner++) {
               
					temp += J_c_reg[(i+inner*9)] * J_c_reg[(j+inner*9)];				
				}

				atomicAdd(&Pr_c[(co) *n_c + id], temp);
				co++;
			}
		}
	
}

__global__ void
	blockDiag_kernelPoints(float *J_p, float *Pr_p, int *pidx, int N, int n_p)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) return;
	
		// fetch Jac into register file //
		float J_p_reg0 = J_p[idx];
		float J_p_reg1 = J_p[N + idx];
		float J_p_reg2 = J_p[2*N + idx];
		float J_p_reg3 = J_p[3*N + idx];
		float J_p_reg4 = J_p[4*N + idx];
		float J_p_reg5 = J_p[5*N + idx];


		int id = pidx[idx];
		
		
		// calculate H_ij = J_ij^T * J_ij		
		atomicAdd(&Pr_p[(0) *n_p + id], J_p_reg0 * J_p_reg0 + J_p_reg3 * J_p_reg3);
		atomicAdd(&Pr_p[(1) *n_p + id], J_p_reg0 * J_p_reg1 + J_p_reg3 * J_p_reg4);
		atomicAdd(&Pr_p[(2) *n_p + id], J_p_reg0 * J_p_reg2 + J_p_reg3 * J_p_reg5);

		atomicAdd(&Pr_p[(3) *n_p + id], J_p_reg1 * J_p_reg1 + J_p_reg4 * J_p_reg4);
		atomicAdd(&Pr_p[(4) *n_p + id], J_p_reg1 * J_p_reg2 + J_p_reg4 * J_p_reg5);

		atomicAdd(&Pr_p[(5) *n_p + id], J_p_reg2 * J_p_reg2 + J_p_reg5 * J_p_reg5);

}

__global__ void
	getDiagElements_kernel(float *d_prec_C, float *d_prec_P, float* diagElem, float l, int ncp, int NC, int NP)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= max(NP,NC)) return;

	// augment point part
	diagElem[NC*ncp/2 + NP*0+idx] = ((d_prec_P[idx + NP * 0]*l));
	diagElem[NC*ncp/2 + NP*1+idx] = ((d_prec_P[idx + NP * 3]*l));
	diagElem[NC*ncp/2 + NP*2+idx] = ((d_prec_P[idx + NP * 5]*l));

	if(idx >= NC) return;

	// augment camera part
	if(ncp==12){
		diagElem[NC*0+idx] = ((d_prec_C[idx + NC * 0]*l));
		diagElem[NC*1+idx] = ((d_prec_C[idx + NC * 6]*l));
		diagElem[NC*2+idx] = ((d_prec_C[idx + NC * 11]*l));
		diagElem[NC*3+idx] = ((d_prec_C[idx + NC * 15]*l));
		diagElem[NC*4+idx] = ((d_prec_C[idx + NC * 18]*l));
		diagElem[NC*5+idx] = ((d_prec_C[idx + NC * 20]*l));
	}
	if(ncp==18){
		diagElem[NC*0+idx] = ((d_prec_C[idx + NC * 0]*l));
		diagElem[NC*1+idx] = ((d_prec_C[idx + NC * 9]*l));
		diagElem[NC*2+idx] = ((d_prec_C[idx + NC * 17]*l));
		diagElem[NC*3+idx] = ((d_prec_C[idx + NC * 24]*l));
		diagElem[NC*4+idx] = ((d_prec_C[idx + NC * 30]*l));
		diagElem[NC*5+idx] = ((d_prec_C[idx + NC * 35]*l));
		diagElem[NC*6+idx] = ((d_prec_C[idx + NC * 39]*l));
		diagElem[NC*7+idx] = ((d_prec_C[idx + NC * 42]*l));
		diagElem[NC*8+idx] = ((d_prec_C[idx + NC * 44]*l));
	}
}

__global__ void
	__launch_bounds__(BLOCKSIZE,10)
	JacDiagMult_kernel12(float *z, float *s, float *d_prec_C, float *d_prec_P, int n_c, int n_p)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= n_c + n_p) return;


	// camera threads
	if(idx < n_c){ 

			// fetch vector //
			float v[6];
			v[0] = s[idx];
			v[1] = s[n_c + idx];
			v[2] = s[2*n_c + idx];
			v[3] = s[3*n_c + idx];
			v[4] = s[4*n_c + idx];
			v[5] = s[5*n_c + idx];

			// multiply M_c^-1 * v//
			float r = 0;

			r = d_prec_C[0 *n_c + idx] * v[0] + d_prec_C[1 *n_c + idx] * v[1] + d_prec_C[2 *n_c + idx] * v[2] + 
				d_prec_C[3 *n_c + idx] * v[3] + d_prec_C[4 *n_c + idx] * v[4] + d_prec_C[5 *n_c + idx] * v[5];
			z[idx]         = (isnan(r) || isinf(r))? 0 : r ;

			r = d_prec_C[1 *n_c + idx] * v[0] + d_prec_C[6 *n_c + idx] * v[1] + d_prec_C[7 *n_c + idx] * v[2] + 
				d_prec_C[8 *n_c + idx] * v[3] + d_prec_C[9 *n_c + idx] * v[4] + d_prec_C[10*n_c + idx] * v[5];
			z[n_c + idx]   = (isnan(r) || isinf(r))? 0 : r ;	

			r = d_prec_C[2 *n_c + idx] * v[0] + d_prec_C[7 *n_c + idx] * v[1] + d_prec_C[11*n_c + idx] * v[2] + 
				d_prec_C[12*n_c + idx] * v[3] + d_prec_C[13*n_c + idx] * v[4] + d_prec_C[14*n_c + idx] * v[5];
			z[2*n_c + idx] = (isnan(r) || isinf(r))? 0 : r ;	

			r = d_prec_C[3 *n_c + idx] * v[0] + d_prec_C[8 *n_c + idx] * v[1] + d_prec_C[12*n_c + idx] * v[2] + 
				d_prec_C[15*n_c + idx] * v[3] + d_prec_C[16*n_c + idx] * v[4] + d_prec_C[17*n_c + idx] * v[5];	
			z[3*n_c + idx] = (isnan(r) || isinf(r))? 0 : r ;

			r = d_prec_C[4 *n_c + idx] * v[0] + d_prec_C[9 *n_c + idx] * v[1] + d_prec_C[13*n_c + idx] * v[2] + 
				d_prec_C[16*n_c + idx] * v[3] + d_prec_C[18*n_c + idx] * v[4] + d_prec_C[19*n_c + idx] * v[5];;
			z[4*n_c + idx] = (isnan(r) || isinf(r))? 0 : r ;

			r = d_prec_C[5 *n_c + idx] * v[0] + d_prec_C[10*n_c + idx] * v[1] + d_prec_C[14*n_c + idx] * v[2] + 
				d_prec_C[17*n_c + idx] * v[3] + d_prec_C[19*n_c + idx] * v[4] + d_prec_C[20*n_c + idx] * v[5];
			z[5*n_c + idx] = (isnan(r) || isinf(r))? 0 : r ;


	} 
	// point threads
	if(idx >= n_c){
		
			// fetch vector //
			//float p[3];
			float p0 = s[6*n_c + (idx-n_c)]; 
			float p1 = s[6*n_c + n_p + (idx-n_c)]; 
			float p2 = s[6*n_c + 2*n_p + (idx-n_c)]; 

			// multiply M_p^-1 * v//
			float r = d_prec_P[idx-n_c] * p0 + d_prec_P[n_p + (idx-n_c)] * p1 + d_prec_P[2*n_p + (idx-n_c)] * p2;
			z[6*n_c + 0*n_p + (idx-n_c)] = (isnan(r) || isinf(r))? 0 : r ;			

			r = d_prec_P[n_p + (idx-n_c)] * p0 + d_prec_P[3*n_p + (idx-n_c)] * p1 + d_prec_P[4*n_p + (idx-n_c)] * p2;
			z[6*n_c + 1*n_p + (idx-n_c)] = (isnan(r) || isinf(r))? 0 : r ;			

			r = d_prec_P[2*n_p + (idx-n_c)] * p0 + d_prec_P[4*n_p + (idx-n_c)] * p1 + d_prec_P[5*n_p + (idx-n_c)] * p2;
			z[6*n_c + 2*n_p + (idx-n_c)] = (isnan(r) || isinf(r))? 0 : r ;			


	}
}

__global__ void
	__launch_bounds__(BLOCKSIZE,10)
	JacDiagMult_kernel18(float *z, float *s, float *d_prec_C, float *d_prec_P, int n_c, int n_p)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= n_c + n_p) return;


	// camera threads
	if(idx < n_c){ 

			// fetch vector //
			float v[9];
			for(int i = 0; i < 9 ; i++){
				v[i] = s[i*n_c + idx]; 
			}

			// multiply M_c^-1 * v//
			float resc = 0;	
			resc += d_prec_C[0*n_c + idx] * v[0];
			resc += d_prec_C[1*n_c + idx] * v[1];
			resc += d_prec_C[2*n_c + idx] * v[2];
			resc += d_prec_C[3*n_c + idx] * v[3];
			resc += d_prec_C[4*n_c + idx] * v[4];
			resc += d_prec_C[5*n_c + idx] * v[5];
			resc += d_prec_C[6*n_c + idx] * v[6];
			resc += d_prec_C[7*n_c + idx] * v[7];
			resc += d_prec_C[8*n_c + idx] * v[8];
			z[0*n_c + idx] = (isnan(resc) || isinf(resc))? 0 : resc;
			resc = 0;	
			resc += d_prec_C[1*n_c + idx] * v[0];
			resc += d_prec_C[9*n_c + idx] * v[1];
			resc += d_prec_C[10*n_c + idx] * v[2];
			resc += d_prec_C[11*n_c + idx] * v[3];
			resc += d_prec_C[12*n_c + idx] * v[4];
			resc += d_prec_C[13*n_c + idx] * v[5];
			resc += d_prec_C[14*n_c + idx] * v[6];
			resc += d_prec_C[15*n_c + idx] * v[7];
			resc += d_prec_C[16*n_c + idx] * v[8];
			z[1*n_c + idx] = (isnan(resc) || isinf(resc))? 0 : resc;
			resc = 0;	
			resc += d_prec_C[2*n_c + idx] * v[0];
			resc += d_prec_C[10*n_c + idx] * v[1];
			resc += d_prec_C[17*n_c + idx] * v[2];
			resc += d_prec_C[18*n_c + idx] * v[3];
			resc += d_prec_C[19*n_c + idx] * v[4];
			resc += d_prec_C[20*n_c + idx] * v[5];
			resc += d_prec_C[21*n_c + idx] * v[6];
			resc += d_prec_C[22*n_c + idx] * v[7];
			resc += d_prec_C[23*n_c + idx] * v[8];
			z[2*n_c + idx] = (isnan(resc) || isinf(resc))? 0 : resc;
			resc = 0;	
			resc += d_prec_C[3*n_c + idx] * v[0];
			resc += d_prec_C[11*n_c + idx] * v[1];
			resc += d_prec_C[18*n_c + idx] * v[2];
			resc += d_prec_C[24*n_c + idx] * v[3];
			resc += d_prec_C[25*n_c + idx] * v[4];
			resc += d_prec_C[26*n_c + idx] * v[5];
			resc += d_prec_C[27*n_c + idx] * v[6];
			resc += d_prec_C[28*n_c + idx] * v[7];
			resc += d_prec_C[29*n_c + idx] * v[8];
			z[3*n_c + idx] = (isnan(resc) || isinf(resc))? 0 : resc;
			resc = 0;	
			resc += d_prec_C[4*n_c + idx] * v[0];
			resc += d_prec_C[12*n_c + idx] * v[1];
			resc += d_prec_C[19*n_c + idx] * v[2];
			resc += d_prec_C[25*n_c + idx] * v[3];
			resc += d_prec_C[30*n_c + idx] * v[4];
			resc += d_prec_C[31*n_c + idx] * v[5];
			resc += d_prec_C[32*n_c + idx] * v[6];
			resc += d_prec_C[33*n_c + idx] * v[7];
			resc += d_prec_C[34*n_c + idx] * v[8];
			z[4*n_c + idx] = (isnan(resc) || isinf(resc))? 0 : resc;
			resc = 0;	
			resc += d_prec_C[5*n_c + idx] * v[0];
			resc += d_prec_C[13*n_c + idx] * v[1];
			resc += d_prec_C[20*n_c + idx] * v[2];
			resc += d_prec_C[26*n_c + idx] * v[3];
			resc += d_prec_C[31*n_c + idx] * v[4];
			resc += d_prec_C[35*n_c + idx] * v[5];
			resc += d_prec_C[36*n_c + idx] * v[6];
			resc += d_prec_C[37*n_c + idx] * v[7];
			resc += d_prec_C[38*n_c + idx] * v[8];
			z[5*n_c + idx] = (isnan(resc) || isinf(resc))? 0 : resc;
			resc = 0;	
			resc += d_prec_C[6*n_c + idx] * v[0];
			resc += d_prec_C[14*n_c + idx] * v[1];
			resc += d_prec_C[21*n_c + idx] * v[2];
			resc += d_prec_C[27*n_c + idx] * v[3];
			resc += d_prec_C[32*n_c + idx] * v[4];
			resc += d_prec_C[36*n_c + idx] * v[5];
			resc += d_prec_C[39*n_c + idx] * v[6];
			resc += d_prec_C[40*n_c + idx] * v[7];
			resc += d_prec_C[41*n_c + idx] * v[8];
			z[6*n_c + idx] = (isnan(resc) || isinf(resc))? 0 : resc;
			resc = 0;	
			resc += d_prec_C[7*n_c + idx] * v[0];
			resc += d_prec_C[15*n_c + idx] * v[1];
			resc += d_prec_C[22*n_c + idx] * v[2];
			resc += d_prec_C[28*n_c + idx] * v[3];
			resc += d_prec_C[33*n_c + idx] * v[4];
			resc += d_prec_C[37*n_c + idx] * v[5];
			resc += d_prec_C[40*n_c + idx] * v[6];
			resc += d_prec_C[42*n_c + idx] * v[7];
			resc += d_prec_C[43*n_c + idx] * v[8];
			z[7*n_c + idx] = (isnan(resc) || isinf(resc))? 0 : resc;
			resc = 0;	
			resc += d_prec_C[8*n_c + idx] * v[0];
			resc += d_prec_C[16*n_c + idx] * v[1];
			resc += d_prec_C[23*n_c + idx] * v[2];
			resc += d_prec_C[29*n_c + idx] * v[3];
			resc += d_prec_C[34*n_c + idx] * v[4];
			resc += d_prec_C[38*n_c + idx] * v[5];
			resc += d_prec_C[41*n_c + idx] * v[6];
			resc += d_prec_C[43*n_c + idx] * v[7];
			resc += d_prec_C[44*n_c + idx] * v[8];
			z[8*n_c + idx] = (isnan(resc) || isinf(resc))? 0 : resc;

	} 
	// point threads
	if(idx >= n_c){

		// fetch Preconditioner //

			// fetch vector //
			float p[3];
			p[0] = s[9*n_c + (idx-n_c)]; 
			p[1] = s[9*n_c + n_p + (idx-n_c)]; 
			p[2] = s[9*n_c + 2*n_p + (idx-n_c)]; 

			// multiply M_p^-1 * v//
			float r = d_prec_P[idx-n_c] * p[0] + d_prec_P[n_p + (idx-n_c)] * p[1] + d_prec_P[2*n_p + (idx-n_c)] * p[2];	
			z[9*n_c + 0*n_p + (idx-n_c)] = (isnan(r) || isinf(r))? 0 : r ;			

			r = d_prec_P[n_p + (idx-n_c)] * p[0] + d_prec_P[3*n_p + (idx-n_c)] * p[1] + d_prec_P[4*n_p + (idx-n_c)] * p[2];
			z[9*n_c + 1*n_p + (idx-n_c)] = (isnan(r) || isinf(r))? 0 : r ;			

			r = d_prec_P[2*n_p + (idx-n_c)] * p[0] + d_prec_P[4*n_p + (idx-n_c)] * p[1] + d_prec_P[5*n_p + (idx-n_c)] * p[2];
			z[9*n_c + 2*n_p + (idx-n_c)] = (isnan(r) || isinf(r))? 0 : r ;

	}
}



#pragma endregion

#pragma region interface

// calculates the dot product of two vectors
double calcVectorDot(float* v1, float* v2, float* parts, int len)
{
    const unsigned int  nblock = 32; 
    int  blen = ((len  + 32 - 1)/ 32 + BLOCKSIZE - 1) / BLOCKSIZE * BLOCKSIZE; 

    calcVectorDot_kernel<<<32, BLOCKSIZE>>>( v1, v2, len, blen, parts);

    float h_parts[32];  
	cudaMemcpy(h_parts, parts, sizeof(float)*32, cudaMemcpyDeviceToHost);

    double dot = 0; 
	
    for(unsigned int  i = 0; i < 32; ++i) 
		dot += h_parts[i];

    return dot;
}

// calculate error vector and the summed squared error
void CalcUtilsGPU::calcErrorVector_d(float &h_sse, float *c, float *p)
{

	// kernel launch
	int n_block_o = (int)( ( (float)n_obs/(float)BLOCKSIZE ) +1 );

	if(c == 0){  // for the first LMA iteration
		calcErrorVector_kernel<<<n_block_o,BLOCKSIZE>>>(d_err, d_cams, d_points,
			d_obs, d_cidx, d_pidx, n_obs, n_c, n_p);
	}else{		 // for temporal parameters, applied when newsse < oldsse in updateCheck() - function
		calcErrorVector_kernel<<<n_block_o,BLOCKSIZE>>>(d_err, c, p,
			d_obs, d_cidx, d_pidx, n_obs, n_c, n_p);
	}

	int n_block_ov;

	///////// compute dot(p*JtJp)

	h_sse = calcVectorDot(d_err, d_err, d_parts, ov_num);
}

// linearize at current position by calculating Jacobian and Preconditioner matrix + gradient
void CalcUtilsGPU::buildJacobian_d(int ri, float &lambdaC, float &lambdaP, Solver s, float &gradMag)
{

	//  kernel launch  //
	int n_block_o = (int)( ( (float)n_obs/(float)BLOCKSIZE ) +1 );
	int n_block_pv = (int)( ( (float)(pv_num)/(float)BLOCKSIZE ) +1 );

	cudaMemset(d_grad,0, sizeof(float)*(pv_num));
	int tcp = (n_cam_params==12)?21:45;
	if(ri==-1 || ri==1)cudaMemset(d_prec_c,0, sizeof(float)*tcp*n_c);
	if(ri==-1 || ri==0)cudaMemset(d_prec_p,0, sizeof(float)*n_p*6);

	bool store_jac = (s==LMA)?true:false;
	if(n_cam_params == 12){
		if(ri != 0)
			linearize_kernelCams12<<<n_block_o,BLOCKSIZE>>>(d_val_C, d_prec_c, d_grad, d_err, d_cams, d_points, d_pidx, d_cidx, n_obs, n_c, n_p, store_jac);
	    if(ri != 1)
			linearize_kernelPoints12<<<n_block_o,BLOCKSIZE>>>(d_val_P, d_prec_p, d_grad+(n_c*n_cam_params/2), d_err, d_cams, d_points, d_pidx, d_cidx, n_obs, n_c, n_p, store_jac);
	}else{
		if(ri != 0)
			linearize_kernelCams18<<<n_block_o,BLOCKSIZE>>>(d_val_C, d_prec_c, d_grad, d_err, d_cams, d_points, d_pidx, d_cidx, n_obs, n_c, n_p, store_jac);
	    if(ri != 1)
			linearize_kernelPoints18<<<n_block_o,BLOCKSIZE>>>(d_val_P, d_prec_p, d_grad+(n_c*n_cam_params/2), d_err, d_cams, d_points, d_pidx, d_cidx, n_obs, n_c, n_p, store_jac);
	}

	
	// extract first diagonal value for a fixed lambda for additive damping
	float Dc[1],Dp[1];
	if(lambdaC < 0 || lambdaP < 0){
			
		cudaMemcpy(Dc, d_prec_c, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(Dp, d_prec_p, sizeof(float), cudaMemcpyDeviceToHost);

		lambdaC *= -Dp[0];
		lambdaP *= -Dp[0];
		
	}
	// get diagonal elements for the multiplicative damping
	if(s==LMA && diagdamping){
		getDiagElements_kernel<<<n_block_pv,BLOCKSIZE>>>(d_prec_c, d_prec_p, d_diagE, lambdaP, n_cam_params, n_c, n_p);
	}

	// calculate gradient magnitude
	if(gradMag == 0){
		gradMag = calcVectorDot(d_grad, d_grad, d_parts, pv_num);
	}	


}

// solve normal equation with conjugate gradient algorithm 
int CalcUtilsGPU::solvePCG_Hessian_d(float lambdaC, float lambdaP, float gmag, int maxi, int mini, float fs){


	double b=0;
	float a=0;
	float t_lambdaC = sqrt(lambdaC); // (H augmented with lambda == J augmented with sqrt(lambda) for A = J^T * J)
	float t_lambdaP = sqrt(lambdaP);
	int k = 0;
	float e = fs;				   // stopping criteria for dot(s) < e*dot(s)
	double firstdotsz = 0;

	//reset state update vector
	cudaMemset(d_dP, 0, sizeof(float)*(pv_num));
	// s0 := g
	cudaMemcpy(s, d_grad, sizeof(float)*(pv_num), cudaMemcpyDeviceToDevice);

	// kernel launch parameter
	int n_block   = (int)( ( (float)(n_c+n_p)/(float)BLOCKSIZE ) +1 );    // for per param-block kernel
	int n_block_pv = (int)( ( (float)pv_num/(float)BLOCKSIZE ) +1 );    // for per param vector-entry kernel
	int n_block_opv = (int)( ( (float)(n_obs+pv_num)/(float)BLOCKSIZE ) +1 );    // for per measurement kernel and param vector-entry kernel
	int n_block_o = (int)( ( (float)(n_obs)/(float)BLOCKSIZE ) +1 );

	// z0 := M^-1 * s0
	if(n_cam_params == 12){
		JacDiagMult_kernel12<<<n_block,BLOCKSIZE>>>(z, s, d_prec_c, d_prec_p, n_c, n_p);
	}else{
		JacDiagMult_kernel18<<<n_block,BLOCKSIZE>>>(z, s, d_prec_c, d_prec_p, n_c, n_p);		
	}

	// p0 := z0
	cudaMemcpy(p, z, sizeof(float)*(pv_num), cudaMemcpyDeviceToDevice);

	//cout << "Start Conjugate Gradients Solver with lambda = " << lambda << endl;

	
	double dotsz = calcVectorDot(s, z, d_parts, pv_num); // s^T * s

	firstdotsz = dotsz;

	while(k < maxi){

		k++;

		// apply damping term
		if(diagdamping)
			vector_mult<<<n_block_pv,BLOCKSIZE>>>(d_diagE, p, JtJp, pv_num);
		else
			scaleAdd_kernel<<<n_block_pv,BLOCKSIZE>>>(JtJp, p, 0, lambdaP, pv_num);

		// solve matrix-vector product Ax
		if(n_cam_params == 12){
			calcHp12_kernel<-1,-1><<<n_block_o,BLOCKSIZE>>>(d_val_C, d_val_P, p, JtJp, d_cidx, d_pidx, n_c, n_p, n_obs);
		}else{
			calcHp18_kernel<-1,-1><<<n_block_o,BLOCKSIZE>>>(d_val_C, d_val_P, p, JtJp, d_cidx, d_pidx, n_c, n_p, n_obs);
		}

		// calculate J*p, s^T*z and alpha //
		a = calcAlpha_H(p, s, z, n_cam_params, dotsz, JtJp, t_lambdaC, t_lambdaP);

		// update x //
		scaleAdd_kernel<<<n_block_pv,BLOCKSIZE>>>(d_dP, p, d_dP, a, pv_num);	

		// update s //
		scaleAdd_kernel<<<n_block_pv,BLOCKSIZE>>>(s, JtJp, s, -a, pv_num);
		
		// update z //
		if(n_cam_params == 12){
				JacDiagMult_kernel12<<<n_block,BLOCKSIZE>>>(z, s, d_prec_c, d_prec_p, n_c, n_p);
			}else{
				JacDiagMult_kernel18<<<n_block,BLOCKSIZE>>>(z, s, d_prec_c, d_prec_p, n_c, n_p);
			}

		// calculate beta //
		if(k%20 != 0) // reset conjugate descent direction after 25 iterations due to accumalating round off errors
			b = calcBeta(s, z, n_cam_params, dotsz, 0);
		else
			b = 0;
		
		// check stopping criteria for dot(s)
		if((dotsz < e*e*firstdotsz && k>=mini) || dotsz > firstdotsz*10) 
			break;

		// update p //
		scaleAdd_kernel<<<n_block_pv,BLOCKSIZE>>>(p, p, z, b, pv_num);


	} 

	return k;



}

// solve reduced camera system with conjugate gradient algorithm
int CalcUtilsGPU::solvePCG_Schur_d(float lambdaC, float lambdaP, float gmag, int maxi, int mini, float fs)
{

	float a=0;
	double b=0;
	float t_lambdaC = sqrt(lambdaC);   // (H augmented with lambda == J augmented with sqrt(lambda) for A = J^T * J)
	float t_lambdaP = sqrt(lambdaP);
	int k = 0;
	float e = fs;				   // stopping criteria for dot(s) < e*dot(s)
	double firstdotsz = 0;

	//init variables
	cudaMemset(d_dP, 0, sizeof(float)*(pv_num));
		
	// kernel launch parameter
	int n_block_c = (int)( ( (float)n_c/(float)BLOCKSIZE ) +1 );		// for per param-block kernel
	int n_block_pc= (int)( ( (float)n_c*n_cam_params/2/(float)BLOCKSIZE ) +1 );		// for per param-block kernel

	// form RCS right side g_c' = g_c - CA^-1 * g_p = g_c - (J_c^T(J_p((J_p^T*J_p)^-1*g_p))))
	float *gc_rcs;
	cudaMalloc((void**)&gc_rcs, sizeof(float)*(n_cam_params/2*n_c));
	calcRSofRCS(gc_rcs); // store g_c' in d_grad_c

	// s0 := g_c'
	cudaMemcpy(s, gc_rcs, sizeof(float)*(n_cam_params/2*n_c), cudaMemcpyDeviceToDevice); // copy only camera part of gradient to s (point part g_p needed for backsubstitution)

	// z0 := M^-1 * s0
	if(n_cam_params == 12){
		JacDiagMult_kernel12<<<n_block_c,BLOCKSIZE>>>(z, s, d_prec_c, d_prec_p, n_c, 0); // precond for cams only
	}else{
		JacDiagMult_kernel18<<<n_block_c,BLOCKSIZE>>>(z, s, d_prec_c, d_prec_p, n_c, 0); // precond for cams only
	}

	// calculate first dotss
	firstdotsz = calcVectorDot(s, z, d_parts, (n_cam_params/2*n_c));

	// p0 := z0
	cudaMemcpy(p, z, sizeof(float)*((n_cam_params/2*n_c)), cudaMemcpyDeviceToDevice);

	double dotsz = firstdotsz; // s^T * s
	
	//cout << "Start Conjugate Gradients Solver with lambda = " << lambda << endl;	
	while(k < maxi){

		k++;		

		// calculate S*p, s^T*z and alpha //
		a = calcAlpha_S(p, s, z, n_cam_params, dotsz, Sp, lambdaC, lambdaP);

		// update x //
		scaleAdd_kernel<<<n_block_pc,BLOCKSIZE>>>(d_dP, p, d_dP, a, (n_cam_params/2)*n_c);

		// update s //
		scaleAdd_kernel<<<n_block_pc,BLOCKSIZE>>>(s, thrust::raw_pointer_cast(&Sp[0]), s, -a, (n_cam_params/2)*n_c);

		// check stopping criteria for dot(s)		
		if(dotsz < e*e*firstdotsz && k>mini || dotsz > firstdotsz*10) break;
		
		// update z //
		if(n_cam_params == 12){
			JacDiagMult_kernel12<<<n_block_c,BLOCKSIZE>>>(z, s, d_prec_c, d_prec_p, n_c, 0);
		}else{
			JacDiagMult_kernel18<<<n_block_c,BLOCKSIZE>>>(z, s, d_prec_c, d_prec_p, n_c, 0);
		}

		// calculate beta //
		if(k%25 != 0) // reset conjugate descent direction after 25 iterations due to accumalating round off errors
			b = calcBeta(s, z, n_cam_params, dotsz, 1);
		else
			b = 0;

		// update p //
		scaleAdd_kernel<<<n_block_pc,BLOCKSIZE>>>(p, p, z, (float)b, (n_cam_params/2*n_c));

	} 
	
	return k;
	//if(k >= th) cout << "Warning: No correct solution found!!" << endl;
	//cudaMemcpy(h_dP, d_dP, sizeof(float)*pv_num, cudaMemcpyDeviceToHost);	
	//for(int i=0; i< n_c*n_cam_params/2; i++) cout << std::setprecision(9) << i << " : " << h_dP[i] << endl;

}

// perform the backsubstitution after solving the reduced camera system
void CalcUtilsGPU::backSubstitution()
{
		int n_block_p = (int)( ( (float)(n_p)/(float)BLOCKSIZE ) +1 );
		int n_block_pp = (int)( ( (float)(n_p*3)/(float)BLOCKSIZE ) +1 );
		int n_block_m  = (int)( ( (float)(n_obs)/(float)BLOCKSIZE ) +1 ); 

		// Jp^T*(Jc*dp_c)
		cudaMemset(tp,0, sizeof(float)*(n_p*3));
		if(n_cam_params == 12){
			calcHp12_kernel<1,0><<<n_block_m,BLOCKSIZE>>>(d_val_C, d_val_P, d_dP, tp, 
				d_cidx, d_pidx, n_c, n_p, n_obs); // no lambda extension
		}else{
			calcHp18_kernel<1,0><<<n_block_m,BLOCKSIZE>>>(d_val_C, d_val_P, d_dP, tp, 
				d_cidx, d_pidx, n_c, n_p, n_obs); // no lambda extension
		}

		// g_p - Jp^T*(Jc*dp_c)
		scaleAdd_kernel<<<n_block_pp,BLOCKSIZE>>>(tp, tp, d_grad+(n_c*n_cam_params/2), -1, n_p*3);

		// solve JpJp*dp_p = g_p - Jp^T*(Jc*dp_c)
		JacDiagMult_kernel12<<<n_block_p,BLOCKSIZE>>>(d_dP+(n_c*n_cam_params/2), tp, d_prec_c, d_prec_p, 0, n_p);
	
}

// evaluate and possibly apply new state update
int CalcUtilsGPU::updateCheck(float oldSSE, float& newSSE, int ri, bool fwd){


	int n_block_c = (int)( ( (float)n_cam_params/2*n_c/(float)BLOCKSIZE ) +1 );    
	int n_block_p = (int)( ( (float)3*n_p/(float)BLOCKSIZE ) +1 );	

	// update to temporal new p //
	if(ri==1 || ri== -1)
		scaleAdd_kernel<<<n_block_c,BLOCKSIZE>>>(Pc_new, d_dP, d_cams, 1, (n_cam_params/2)*n_c);

	if(n_cam_params == 12 && (ri==1 || ri== -1))
		scaleAdd_kernel<<<n_block_c,BLOCKSIZE>>>(Pc_new+(6*n_c), d_dP, d_cams+(6*n_c), 0, 3*n_c); // retain fixed focal lengths and dist params

	if(ri==0 || ri== -1)
		scaleAdd_kernel<<<n_block_p,BLOCKSIZE>>>(Pp_new, d_dP+(n_cam_params/2*n_c), d_points, 1, 3*n_p);


	// calc new sse //
	if(ri ==-1 ) calcErrorVector_d(newSSE, Pc_new, Pp_new);
	if(ri == 1 ) calcErrorVector_d(newSSE, Pc_new, d_points);
	if(ri == 0 ) calcErrorVector_d(newSSE, d_cams, Pp_new);
	//cout << "old SSE: " << oldSSE << ", new SSE: " << newSSE << endl;

	if((newSSE < oldSSE) || fwd == true){

		if(ri==0 || ri== -1) cudaMemcpy(d_points, Pp_new, sizeof(float)*3*n_p, cudaMemcpyDeviceToDevice);
		if(ri==1 || ri== -1) cudaMemcpy(d_cams,   Pc_new, sizeof(float)*9*n_c, cudaMemcpyDeviceToDevice);
		return 1;

	}
	else{
		return 0;
	}
}

// calculate alpha for cg-algorithm (hessian)
float CalcUtilsGPU::calcAlpha_H(float *p, float *s, float *z, int ncp, double& dotsz, float *JtJp, float lc, float lp){


		///////// compute dot(p*JtJp)	

		double dotpJtJp = calcVectorDot(p, JtJp, d_parts, pv_num);
		///////// compute alpha
		float a = (float)dotsz/dotpJtJp;

		return a;
}

// calculate alpha for cg-algorithm (schur)
float CalcUtilsGPU::calcAlpha_S(float *p, float *s, float *z, int ncp, double& dotsz, float *Sp, float lc, float lp){


		int n_block_pc = (int)( ( (float)(n_c*ncp/2)/(float)BLOCKSIZE ) +1 );	
		int n_block_p = (int)( ( (float)(n_p)/(float)BLOCKSIZE ) +1 );
		int n_block_m  = (int)( ( (float)(n_obs)/(float)BLOCKSIZE ) +1 ); 

		
		// 1. Jc^T*(Jc*p_c) and Jp^T*((Jc)*pc)
		cudaMemset(pv, 0, sizeof(float)*pv_num);
		if(diagdamping)
			vector_mult<<<n_block_pc,BLOCKSIZE>>>(d_diagE, p, pv, n_c*ncp/2);
		else
			scaleAdd_kernel<<<n_block_pc,BLOCKSIZE>>>(pv, p, 0, lp, n_c*ncp/2);

		if(n_cam_params == 12){
			calcHp12_kernel<1,-1><<<n_block_m,BLOCKSIZE>>>(d_val_C, d_val_P, p, pv, // hier nen ganzen pv_num Vektor [Vp(cameras),t2(punkte)]
				d_cidx, d_pidx, n_c, n_p, n_obs);
		}else{
			calcHp18_kernel<1,-1><<<n_block_m,BLOCKSIZE>>>(d_val_C, d_val_P, p, pv, // hier nen ganzen pv_num Vektor [Vp(cameras),t2(punkte)]
				d_cidx, d_pidx, n_c, n_p, n_obs);
		}		

		// 2. (Jp^T*Jp)^-1(Jp^T*((Jc)*pc))		
		JacDiagMult_kernel12<<<n_block_p,BLOCKSIZE>>>(tp, pv + n_c*ncp/2, d_prec_c, d_prec_p, 0, n_p);


		// 3. Jc^T(Jp*((Jp^T*Jp)^-1(Jp^T*((Jc)*pc)))
		cudaMemset(tc,0, sizeof(float)*(n_c*ncp/2));
		if(n_cam_params == 12){
			calcHp12_kernel<0,1><<<n_block_m,BLOCKSIZE>>>(d_val_C, d_val_P, tp, tc, 
				d_cidx, d_pidx, n_c, n_p, n_obs);
		}else{
			calcHp18_kernel<0,1><<<n_block_m,BLOCKSIZE>>>(d_val_C, d_val_P, tp, tc, 
				d_cidx, d_pidx, n_c, n_p, n_obs);
		}

		// 4. Sp = Upc - Vpc
		scaleAdd_kernel<<<n_block_pc,BLOCKSIZE>>>(Sp, tc, pv, -1, n_c*ncp/2);

		///////// compute pc^T*(S*pc)	
		double dotpSp = calcVectorDot(p, Sp, d_parts, (n_cam_params/2*n_c));

		///////// return alpha					
		return (float)(dotsz/dotpSp);

}

// calculate beta for (n)cg-algorithm according to Fletcher&Reeves
double CalcUtilsGPU::calcBeta(float *s, float *z, int ncp, double &dots, int schur){


		double dotsn = 0;

		// compute beta a = dot(s_new)/dot(s) //
		if(schur == 0)
			dotsn = calcVectorDot(s, z, d_parts, pv_num);		
		else
			dotsn = calcVectorDot(s, z, d_parts, n_c*n_cam_params/2);

			
		// compute beta					
		double beta = dotsn/dots;
		dots = dotsn;
		return beta;

}

// calculate the right side of the reduced camera system
void CalcUtilsGPU::calcRSofRCS(float* d_rcsRS)
{

	int n_block_p = (int)( ( (float)(n_p)/(float)BLOCKSIZE ) +1 );
	int n_block_m  = (int)( ( (float)(n_obs)/(float)BLOCKSIZE ) +1 ); 

	// 1. (J_p^T*J_p)^-1*g_p
	JacDiagMult_kernel12<<<n_block_p,BLOCKSIZE>>>(tp, d_grad+(n_c*n_cam_params/2), d_prec_c, d_prec_p, 0, n_p); // for points only

	// 2. (J_c^T(J_p((J_p^T*J_p)^-1*g_p)))
	cudaMemset(Sp,0, sizeof(float)*(n_c*n_cam_params/2));
	if(n_cam_params == 12){
		calcHp12_kernel<0,1><<<n_block_m,BLOCKSIZE>>>(d_val_C, d_val_P, tp, Sp, 
			d_cidx, d_pidx, n_c, n_p, n_obs); // no lambda extension for the right side of the eq
	}else{
		calcHp18_kernel<0,1><<<n_block_m,BLOCKSIZE>>>(d_val_C, d_val_P, tp, Sp, 
			d_cidx, d_pidx, n_c, n_p, n_obs); // no lambda extension for the right side of the eq
	}

	// 3. g_c - (J_c^T(J_p((J_p^T*J_p)^-1*g_p)))
	int n_block_c = (int)( ( (float)(n_c*n_cam_params/2)/(float)BLOCKSIZE ) +1 );
	scaleAdd_kernel<<<n_block_c,BLOCKSIZE>>>(d_rcsRS, Sp, d_grad, -1, n_c*n_cam_params/2);

}

// calculate preconditioner with recalculating the jacobian
void CalcUtilsGPU::formBdiagHessian(float lambdaC, float lambdaP, int ri){


	int NC = n_c;
	int NP = n_p;
	if(ri==0) NC = 0;
	if(ri==1) NP = 0;	
	int n_block = (int)( ( (float)n_obs/(float)BLOCKSIZE ) +1 );

	// init space in DRAM for Block Jacobi Preconditioner points: 3x3 per block and cams: 9x9/6x6 per block (symmetric)
	int tcp = (n_cam_params==12)?21:45;
	cudaMemset(d_prec_c, 0, sizeof(float)*n_c*tcp);
	cudaMemset(d_prec_p, 0, sizeof(float)*n_p*6);

	// build block jacobi preconditioner
	if(n_cam_params == 12){
		blockDiag_kernelCams12<<<n_block,BLOCKSIZE>>>(d_val_C, d_prec_c, d_cidx, n_obs, NC);
	    blockDiag_kernelPoints<<<n_block,BLOCKSIZE>>>(d_val_P, d_prec_p, d_pidx, n_obs, NP);
	}else{
		blockDiag_kernelCams18<<<n_block,BLOCKSIZE>>>(d_val_C, d_prec_c, d_cidx, n_obs, NC);
	    blockDiag_kernelPoints<<<n_block,BLOCKSIZE>>>(d_val_P, d_prec_p, d_pidx, n_obs, NP);
	}


}	

// calculate next Gradient descent step and apply it
void CalcUtilsGPU::calcGDstep_d(float &sse, int ncp, int k, float l, int b_reset, float t, float c1, float c2, float start_a){


	int n_block_pv   = (int)( ( (float)(pv_num)/(float)BLOCKSIZE ) +1 );
	int n_block_cp   = (int)( ( (float)(n_c+n_p)/(float)BLOCKSIZE ) +1 );
	double b = 0;
	float a = start_a;

	if(k==1){
	////////init and copy input parameter data for NLCG/////////
		cudaMemcpy(d_dP, d_cams, sizeof(float)*(n_c*ncp/2), cudaMemcpyDeviceToDevice); 
		cudaMemcpy(d_dP + (n_c*ncp/2), d_points, sizeof(float)*(n_p*3), cudaMemcpyDeviceToDevice);		
		// set z
		if(n_cam_params == 18)
			solveChol_kernel9<typ><<< (int)( ( (float)(n_c)/(float)BLOCKSIZE ) +1 ),BLOCKSIZE>>>(d_prec_c, z, d_grad, n_c, l, true);
		else
			solveChol_kernel6<typ><<< (int)( ( (float)(n_c)/(float)BLOCKSIZE ) +1 ),BLOCKSIZE>>>(d_prec_c, z, d_grad, n_c, l, true);			

		solveChol_kernel3<typ><<< (int)( ( (float)(n_p)/(float)BLOCKSIZE ) +1 ),BLOCKSIZE>>>(d_prec_p, z+(n_c*n_cam_params/2), d_grad+(n_c*n_cam_params/2), n_p, l, true);

		// set initial search direction p
		cudaMemcpy(d_p, z, sizeof(float)*(pv_num), cudaMemcpyDeviceToDevice);

		// calculate first dot(z,grad)
		prevdot = calcVectorDot(d_grad, z, d_parts, pv_num);

	}else{
	////////calculate new gradient and beta/////////

		// update z
		if(n_cam_params == 18)
			solveChol_kernel9<typ><<< (int)( ( (float)(n_c)/(float)BLOCKSIZE ) +1 ),BLOCKSIZE>>>(d_prec_c, z, d_grad, n_c, l, true);
		else
			solveChol_kernel6<typ><<< (int)( ( (float)(n_c)/(float)BLOCKSIZE ) +1 ),BLOCKSIZE>>>(d_prec_c, z, d_grad, n_c, l, true);			

		solveChol_kernel3<typ><<< (int)( ( (float)(n_p)/(float)BLOCKSIZE ) +1 ),BLOCKSIZE>>>(d_prec_p, z+(n_c*n_cam_params/2), d_grad+(n_c*n_cam_params/2), n_p, l, true);

		
		// calc beta according to Fletcher&Reeves
		b = calcBeta(d_grad, z, ncp, prevdot, 0);

	}

	/////////update search direction///////////
	scaleAdd_kernel<<<n_block_pv,BLOCKSIZE>>>(d_p, d_p, z, b, pv_num); 

	// grad*dp
	double gdp = calcVectorDot(d_grad, d_p, d_parts, pv_num);	

	// reset conjugacy	
	if(k % b_reset == 0 || gdp <= 0){
		scaleAdd_kernel<<<n_block_pv,BLOCKSIZE>>>(d_p, d_p, z, 0, pv_num);
		gdp = calcVectorDot(d_grad, d_p, d_parts, pv_num);
	}

	// calculate dynamic starting value for a
	if(start_a == 0){	
		// prec*dp		
		if(n_cam_params == 12){
			JacDiagMult_kernel12<<<n_block_cp,BLOCKSIZE>>>(grad_new, d_p, d_prec_c, d_prec_p, n_c, n_p);
		}else{
			JacDiagMult_kernel18<<<n_block_cp,BLOCKSIZE>>>(grad_new, d_p, d_prec_c, d_prec_p, n_c, n_p);
		}
		
		// dp*prec*dp
		double dpPrdp = calcVectorDot(grad_new, d_p, d_parts, pv_num);

		// a= (grad*dp)/(dp*prec*dp)
		a = (gdp/dpPrdp);
		if(_isnan(a)) a = 2;
	}

	// start a inexact line search on the descent direction
	linesearch(a, sse, t, c1, c2);

	/////////scale search direction with alpha and apply to x/////////
	scaleAdd_kernel<<<n_block_pv,BLOCKSIZE>>>(d_dP, d_p, d_dP, a, pv_num);

	/////////apply changes/////////
	if(a!=0){
		cudaMemcpy(d_cams, d_dP, sizeof(float)*(n_c*ncp/2), cudaMemcpyDeviceToDevice); 
		cudaMemcpy(d_points, d_dP + (n_c*ncp/2), sizeof(float)*(n_p*3), cudaMemcpyDeviceToDevice);
	}
		
}

// perform an inexact linesearch on a given descent direction
void CalcUtilsGPU::linesearch(float &a, float &sse, float t, float c1, float c2 )
{
	int n_block_pv   = (int)( ( (float)(pv_num)/(float)BLOCKSIZE ) +1 );

	// calc m //		
	float m = calcVectorDot(d_grad, d_p, d_parts, pv_num);

	// init diff //
	float diff = 0;
	float tempSSE = 0;
	float te = c1*abs(m);
	float ls = 0;
	float rs = c2*abs(m);

	// while Armijo & Curvature conditions are not met (strong wolfe conditions) //
	while(diff <= a*te || ls > rs) 
	{

		// calculate temporal sse //
		scaleAdd_kernel<<<n_block_pv,BLOCKSIZE>>>(Pc_new, d_p, d_cams, a, n_c*n_cam_params/2);
		if(n_cam_params==12)			
			scaleAdd_kernel<<<n_block_pv,BLOCKSIZE>>>(Pc_new + (n_c*6), d_p, d_cams + (n_c*6), 0, (n_c*3));				
		scaleAdd_kernel<<<n_block_pv,BLOCKSIZE>>>(Pp_new, d_p + (n_c*n_cam_params/2), d_points, a, (n_p*3));

		calcErrorVector_d(tempSSE, Pc_new, Pp_new);

		// calculate diff f(x) - f(x - a*g) //
		diff = sse - tempSSE;

		// if no decrease in this search direction found return a = 0
		if(a<=1e-5f){			
			break;
		}
		if(isinf(sse)){
			a=0;
			break;
		}
		
		///// calculate ls for curvature condition ??
		if(rs>0){
			
			// calc temp gradient
			int n_block_o = (int)( ( (float)n_obs/(float)BLOCKSIZE ) +1 );
			cudaMemset(d_grad,0, sizeof(float)*pv_num);
			linearize_kernelCams12<<<n_block_o,BLOCKSIZE>>>(d_val_C, d_prec_c, d_grad, d_err, d_cams, d_points, d_pidx, d_cidx, n_obs, n_c, n_p, false);
			linearize_kernelPoints12<<<n_block_o,BLOCKSIZE>>>(d_val_P, d_prec_p, d_grad+(n_c*n_cam_params/2), d_err, d_cams, d_points, d_pidx, d_cidx, n_obs, n_c, n_p, false);

			// calc dot product dp*grad_temp
			ls = abs(calcVectorDot(d_grad, d_p, d_parts, pv_num));

		}

		if(isnan(diff))
			diff=0;
				
		// decrease alpha //
		a = a*t;

	}
	// apply last sse
	if(a!=0)
		sse = tempSSE;
	

}

// solve normal equation for each single camera
void CalcUtilsGPU::resection(float lambda){

	int n_block   = (int)( ( (float)(n_c)/(float)BLOCKSIZE ) +1 );

	cudaMemset(d_dP, 0, sizeof(float)*(pv_num));

	// solve for camera parameters only
	if(n_cam_params == 18)
		solveChol_kernel9<typ><<<n_block,BLOCKSIZE>>>(d_prec_c, d_dP, d_grad, n_c, lambda, diagdamping);
	else
		solveChol_kernel6<typ><<<n_block,BLOCKSIZE>>>(d_prec_c, d_dP, d_grad, n_c, lambda, diagdamping);	
}

// solve normal equation for each single point
void CalcUtilsGPU::intersection(float* t_grad, float lambda){
	
	int n_block   = (int)( ( (float)(n_p)/(float)BLOCKSIZE ) +1 );

	// solve for point parameters only
	cudaMemset(d_dP, 0, sizeof(float)*pv_num);
	if(t_grad == 0)
		solveChol_kernel3<typ><<<n_block,BLOCKSIZE>>>(d_prec_p, d_dP+(n_c*n_cam_params/2), d_grad+(n_c*n_cam_params/2), n_p, lambda, diagdamping);
	else
		solveChol_kernel3<typ><<<n_block,BLOCKSIZE>>>(d_prec_p, d_dP+(n_c*n_cam_params/2), t_grad+(n_c*n_cam_params/2), n_p, lambda, diagdamping);


}

// calculate gain ration for advanced damping term treatment
float CalcUtilsGPU::calcro(float lambdaC, float lambdaP)
{

	int n_block_pv   = (int)( ( (float)(pv_num)/(float)BLOCKSIZE ) +1 );

	// (lambda*dP)
	if(diagdamping){
		vector_mult<<<n_block_pv,BLOCKSIZE>>>(d_diagE, d_dP, as, pv_num);
		scaleAdd_kernel<<<n_block_pv,BLOCKSIZE>>>(as, as, d_grad, 1, pv_num); 
	}else{
		scaleAdd_kernel<<<n_block_pv,BLOCKSIZE>>>(as, d_dP, d_grad, lambdaC, pv_num);
	}

	// dot(dP,(lambda*dP))
	float ro = calcVectorDot(d_dP, as, d_parts, pv_num);	

	return ro;
}

// invert preconditioner
void CalcUtilsGPU::invHessDiagBlocks(float l){

	int n_block_c = (int)( ( (float)n_c/(float)BLOCKSIZE ) +1 );
	int n_block_p = (int)( ( (float)n_p/(float)BLOCKSIZE ) +1 );

	// invert matrix for preconditioning purposes
	matinv_3x3<typ><<<n_block_p,BLOCKSIZE>>>(d_prec_p, d_prec_p, n_p, l, diagdamping);
	
	if(n_cam_params == 12)
		matinv_6x6<typ><<<n_block_c,BLOCKSIZE>>>(d_prec_c, d_prec_c, n_c, l, diagdamping);
	else
		matinv_9x9<typ><<<n_block_c,BLOCKSIZE>>>(d_prec_c, d_prec_c, n_c, l, diagdamping);

}

// perform embedded point iterations and apply stateupdate if an error decrease is accomplished
int CalcUtilsGPU::epi(bool backsub, float lambda, int iter, float sse, float &newSSE, float &iSSE){
	

	int n_block = (int)( ( (float)n_obs/(float)BLOCKSIZE ) +1 );
	int n_block_cc = (int)( ( (float)n_cam_params/2*n_c/(float)BLOCKSIZE ) +1 ); 
	int n_block_pp = (int)( ( (float)(n_p*3)/(float)BLOCKSIZE ) +1 );
	
	float tsse = sse;
	int iterations = iter;
	if(!backsub) iterations++; // for EPI only allow one more iteration

	// copy params and/or apply dp
	scaleAdd_kernel<<<n_block_cc,BLOCKSIZE>>>(Pc_new, d_dP, d_cams, 1, (n_cam_params/2)*n_c);
	if(n_cam_params == 12)
		scaleAdd_kernel<<<n_block_cc,BLOCKSIZE>>>(Pc_new+(6*n_c), d_dP, d_cams+(6*n_c), 0, 3*n_c); // retain fixed focal lengths and dist params
	if(backsub)
		scaleAdd_kernel<<<n_block_pp,BLOCKSIZE>>>(Pp_new, d_dP+((n_cam_params/2)*n_c), d_points, 1, 3*n_p); 
	else
		cudaMemcpy(Pp_new, d_points, sizeof(float)*3*n_p, cudaMemcpyDeviceToDevice);

	// calculate new error
	calcErrorVector_d(newSSE, Pc_new, Pp_new);
	iSSE = newSSE;
	
	// apply changes at this point
	if(newSSE < sse){
		cudaMemcpy(d_points, Pp_new, sizeof(float)*3*n_p, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_cams,   Pc_new, sizeof(float)*9*n_c, cudaMemcpyDeviceToDevice);			
		tsse = newSSE;
	}
	
	// linearize with new params
	cudaMemset(grad_new,0, sizeof(float)*(pv_num));
	cudaMemset(d_prec_p,0, sizeof(float)*n_p*6);

	if(n_cam_params == 12)
		linearize_kernelPoints12<<<n_block,BLOCKSIZE>>>(d_val_P, d_prec_p, grad_new+(n_c*n_cam_params/2), d_err, Pc_new, Pp_new, d_pidx, d_cidx, n_obs, n_c, n_p, false);
	else
		linearize_kernelPoints18<<<n_block,BLOCKSIZE>>>(d_val_P, d_prec_p, grad_new+(n_c*n_cam_params/2), d_err, Pc_new, Pp_new, d_pidx, d_cidx, n_obs, n_c, n_p, false);

	// perform the embedded point iterations
	float psse = tsse; // previous sse
	for(int i = 0 ; i < iter; i++){

		// solve
		intersection(grad_new, lambda);

		// apply new points 
		scaleAdd_kernel<<<n_block_pp,BLOCKSIZE>>>(Pp_new, d_dP+(n_cam_params/2*n_c), Pp_new, 1, 3*n_p);

		// calculate new error
		calcErrorVector_d(newSSE, Pc_new, Pp_new);
		
		// check for error increase during the epi and brake up in such case
		if(backsub)
			if(newSSE > psse){			
				if(i>0){
					scaleAdd_kernel<<<n_block_pp,BLOCKSIZE>>>(Pp_new, d_dP+(n_cam_params/2*n_c), Pp_new, -1, 3*n_p);
					calcErrorVector_d(newSSE, Pc_new, Pp_new);
					//newSSE = psse;
				}
				break;  
			}else{
				psse = newSSE;
			}
		
		// linearize for next point iteration
		if(i+1<iter){
			cudaMemset(grad_new,0, sizeof(float)*(pv_num));
			cudaMemset(d_prec_p,0, sizeof(float)*n_p*6);
			if(n_cam_params == 12)
				linearize_kernelPoints12<<<n_block,BLOCKSIZE>>>(d_val_P, d_prec_p, grad_new+(n_c*n_cam_params/2), d_err, Pc_new, Pp_new, d_pidx, d_cidx, n_obs, n_c, n_p, false);
			else
				linearize_kernelPoints18<<<n_block,BLOCKSIZE>>>(d_val_P, d_prec_p, grad_new+(n_c*n_cam_params/2), d_err, Pc_new, Pp_new, d_pidx, d_cidx, n_obs, n_c, n_p, false);
		}
	}
	
	
	if(newSSE < sse && newSSE > 0){  // sse(Newton-step + epi) < sse(newton-step) < previous Error

			cudaMemcpy(d_points, Pp_new, sizeof(float)*3*n_p, cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_cams,   Pc_new, sizeof(float)*9*n_c, cudaMemcpyDeviceToDevice);

			return 1;
	}else{			  // reject new parameters by not applying them
					
			if(tsse < sse){
				calcErrorVector_d(newSSE, 0, 0);
				return 1;  // only Newton step = decrease
			}
			else{
				return 0;  // no decrease
			}
	}
	

}

// write back solution to host
void CalcUtilsGPU::wb_result(float *h_cams, float *h_points)
{

	cudaMemcpy(h_points, d_points, sizeof(float)*3*n_p, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_cams,   d_cams, sizeof(float)*9*n_c, cudaMemcpyDeviceToHost);

}


#pragma endregion

