#pragma once

/*
* Implementations of dense Cholesky Solvers. 
* 1) decompose
* 2) forward substitution
* 3) backward substitution
* For dimension 3, 6 and 9
*/


#include <iomanip>


template<typename T>
__device__ __forceinline__ T dot_prod(const T *x, const T *y, int n)
{
    T res = 0.0;
    for (int i = 0 ; i < n; i++)
    {
        res += x[i] * y[i];
    }
    return res;
}

template<typename T> 
__global__ void solveChol_kernel3(float* Hp, float* dp, float* gp, int batch, float la, float mult)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= batch) return;

	// fetch Input
	T P[9];
//	int dim = 3;

	if(mult)
	{
		P[0] = Hp[batch*0+idx]*(1+la);
		P[1] = Hp[batch*1+idx];
		P[2] = Hp[batch*2+idx];
		P[3] = Hp[batch*1+idx];
		P[4] = Hp[batch*3+idx]*(1+la);
		P[5] = Hp[batch*4+idx];
		P[6] = Hp[batch*2+idx];
		P[7] = Hp[batch*4+idx];
		P[8] = Hp[batch*5+idx]*(1+la);
	}else{
		P[0] = Hp[batch*0+idx]+(la);
		P[1] = Hp[batch*1+idx];
		P[2] = Hp[batch*2+idx];
		P[3] = Hp[batch*1+idx];
		P[4] = Hp[batch*3+idx]+(la);
		P[5] = Hp[batch*4+idx];
		P[6] = Hp[batch*2+idx];
		P[7] = Hp[batch*4+idx];
		P[8] = Hp[batch*5+idx]+(la);
	}
	
	T  d;
	T* Ucoli = P;	
	
	// decompose P = LLT	
	d = Ucoli[0] - dot_prod<T>(Ucoli, Ucoli, 0);
    Ucoli[0] = __fsqrt_rn(d);

    d = __frcp_rn(Ucoli[0]);  
    P[3] = d*(P[3] - dot_prod<T>(Ucoli, P+3, 0));	
	P[6] = d*(P[6] - dot_prod<T>(Ucoli, P+6, 0));
        
	Ucoli+=3;
	d = Ucoli[1] - dot_prod<T>(Ucoli, Ucoli, 1);
    Ucoli[1] = __fsqrt_rn(d);
    d = __frcp_rn(Ucoli[1]);   
    P[7] = d*(P[7] - dot_prod<T>(Ucoli, P+6, 1));	
        
	Ucoli+=3;
	d = Ucoli[2] - dot_prod<T>(Ucoli, Ucoli, 2);
    Ucoli[2] = __fsqrt_rn(d);

	// intermediate solution y
	T y[3];

	//forwardsub		
	y[0]= __fdividef((gp[idx]),P[0]);

	T s = y[0]*P[3];
	y[1]= __fdividef((gp[batch+idx]-s),P[4]);

	s = y[0]*P[6] + y[1]*P[7];
	y[2]= __fdividef((gp[batch*2+idx]-s),P[8]);
    
	//backwardsub
	dp[batch*2+idx] = __fdividef((y[2]),P[8]);
	if(isnan(dp[batch*2+idx]) || isinf(dp[batch*2+idx])) dp[batch*2+idx] = 0;

	s = dp[batch*2+idx]*P[7];
	dp[batch+idx] = __fdividef((y[1]-s),P[4]);
	if(isnan(dp[batch+idx]) || isinf(dp[batch+idx])) dp[batch+idx] = 0;

	s = dp[batch*2+idx]*P[6] + dp[batch+idx]*P[3];
	dp[idx] = __fdividef((y[0]-s),P[0]);
	if(isnan(dp[idx]) || isinf(dp[idx])) dp[idx] = 0;
	


}

template<typename T> 
__global__ void solveChol_kernel6(float* Hp, float* dp, float* gp, int batch, float l, bool mult)
{


	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= batch) return;
	int dim = 6;

	// fetch Input
	T P[36];


	P[0] = Hp[batch*0+idx];
	P[1] = Hp[batch*1+idx];
	P[2] = Hp[batch*2+idx];
	P[3] = Hp[batch*3+idx];
	P[4] = Hp[batch*4+idx];
	P[5] = Hp[batch*5+idx];

	P[6] = Hp[batch*1+idx];
	P[7] = Hp[batch*6+idx];
	P[8] = Hp[batch*7+idx];
	P[9] = Hp[batch*8+idx];
	P[10] = Hp[batch*9+idx];
	P[11] = Hp[batch*10+idx];

	P[12] = Hp[batch*2+idx];
	P[13] = Hp[batch*7+idx];
	P[14] = Hp[batch*11+idx];
	P[15] = Hp[batch*12+idx];
	P[16] = Hp[batch*13+idx];
	P[17] = Hp[batch*14+idx];

	P[18] = Hp[batch*3+idx];
	P[19] = Hp[batch*8+idx];
	P[20] = Hp[batch*12+idx];
	P[21] = Hp[batch*15+idx];
	P[22] = Hp[batch*16+idx];
	P[23] = Hp[batch*17+idx];

	P[24] = Hp[batch*4+idx];
	P[25] = Hp[batch*9+idx];
	P[26] = Hp[batch*13+idx];
	P[27] = Hp[batch*16+idx];
	P[28] = Hp[batch*18+idx];
	P[29] = Hp[batch*19+idx];

	P[30] = Hp[batch*5+idx];
	P[31] = Hp[batch*10+idx];
	P[32] = Hp[batch*14+idx];
	P[33] = Hp[batch*17+idx];
	P[34] = Hp[batch*19+idx];
	P[35] = Hp[batch*20+idx];

	if(mult) 
	{
		P[0] = (P[0] + (P[0]*l));
		P[7] = (P[7] + (P[7]*l));
		P[14] = (P[14] + (P[14]*l));
		P[21] = (P[21] + (P[21]*l));
		P[28] = (P[28] + (P[28]*l));
		P[35] = (P[35] + (P[35]*l));
	}else{
		P[0] += l;
		P[7] += l;
		P[14] += l;
		P[21] += l;
		P[28] += l;
		P[35] += l;
	}

	int i, j;
	T  d;
	T* Ucoli;	

	// decompose P = LLT
    for( Ucoli=P, i=0; i<dim; ++i, Ucoli+=dim)
    {  
        d = Ucoli[i] - dot_prod<T>(Ucoli, Ucoli, i);
        Ucoli[i] = __fsqrt_rn(d);
        d = __frcp_rn(Ucoli[i]);
        for( j=i+1; j<dim; ++j)
        {   
            P[i+j*dim] = d*(P[i+j*dim] - dot_prod<T>(Ucoli, P+j*dim, i));			
        }
    }

	// intermediate solution y
	T y[6];

	//forwardsub
	
	y[0]= __fdividef((gp[idx]),P[0]);

	T s = y[0]*P[6];
	y[1]= __fdividef((gp[batch+idx]-s),P[7]);

	s = y[0]*P[12] + y[1]*P[13];
	y[2]= __fdividef((gp[batch*2+idx]-s),P[14]);

	s = y[0]*P[18] + y[1]*P[19] + y[2]*P[20];
	y[3]= __fdividef((gp[batch*3+idx]-s),P[21]);

	s = y[0]*P[24] + y[1]*P[25] + y[2]*P[26] + y[3]*P[27];
	y[4]= __fdividef((gp[batch*4+idx]-s),P[28]);

	s = y[0]*P[30] + y[1]*P[31] + y[2]*P[32] + y[3]*P[33] + y[4]*P[34];
	y[5]= __fdividef((gp[batch*5+idx]-s),P[35]);


	//backsub
	dp[batch*5+idx] = __fdividef((y[5]),P[35]);
	if(isnan(dp[batch*5+idx]) || isinf(dp[batch*5+idx])) dp[batch*5+idx] = 0;

	s = dp[batch*5+idx]*P[34];
	dp[batch*4+idx] = __fdividef((y[4]-s),P[28]);
	if(isnan(dp[batch*4+idx]) || isinf(dp[batch*4+idx])) dp[batch*4+idx] = 0;

	s = dp[batch*5+idx]*P[33] + dp[batch*4+idx]*P[27];
	dp[batch*3+idx] = __fdividef((y[3]-s),P[21]);
	if(isnan(dp[batch*3+idx]) || isinf(dp[batch*3+idx])) dp[batch*3+idx] = 0;
	
	s = dp[batch*5+idx]*P[32] + dp[batch*4+idx]*P[26] + dp[batch*3+idx]*P[20];
	dp[batch*2+idx] = __fdividef((y[2]-s),P[14]);
	if(isnan(dp[batch*2+idx]) || isinf(dp[batch*2+idx])) dp[batch*2+idx] = 0;

	s = dp[batch*5+idx]*P[31] + dp[batch*4+idx]*P[25] + dp[batch*3+idx]*P[19] + dp[batch*2+idx]*P[13];
	dp[batch*1+idx] = __fdividef((y[1]-s),P[7]);
	if(isnan(dp[batch*1+idx]) || isinf(dp[batch*1+idx])) dp[batch*1+idx] = 0;

	s = dp[batch*5+idx]*P[30] + dp[batch*4+idx]*P[24] + dp[batch*3+idx]*P[18] + dp[batch*2+idx]*P[12] + dp[batch+idx]*P[6];
	dp[idx]			= __fdividef((y[0]-s),P[0]);
	if(isnan(dp[idx]) || isinf(dp[idx])) dp[idx] = 0;



}

template<typename T> 
__global__ void solveChol_kernel9(float* Hp, float* dp, float* gp, int batch, float l, bool mult)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= batch) return;
	int dim = 9;

	// fetch Input
	T P[81];
	T x[9];
	T b[9];

	
	P[0] = Hp[batch*0+idx];
	P[1] = Hp[batch*1+idx];
	P[2] = Hp[batch*2+idx];
	P[3] = Hp[batch*3+idx];
	P[4] = Hp[batch*4+idx];
	P[5] = Hp[batch*5+idx];
	P[6] = Hp[batch*6+idx];
	P[7] = Hp[batch*7+idx];
	P[8] = Hp[batch*8+idx];

	P[9] = Hp[batch*1+idx];
	P[10] = Hp[batch*9+idx];
	P[11] = Hp[batch*10+idx];
	P[12] = Hp[batch*11+idx];
	P[13] = Hp[batch*12+idx];
	P[14] = Hp[batch*13+idx];
	P[15] = Hp[batch*14+idx];
	P[16] = Hp[batch*15+idx];
	P[17] = Hp[batch*16+idx];

	P[18] = Hp[batch*2+idx];
	P[19] = Hp[batch*10+idx];
	P[20] = Hp[batch*17+idx];
	P[21] = Hp[batch*18+idx];
	P[22] = Hp[batch*19+idx];
	P[23] = Hp[batch*20+idx];
	P[24] = Hp[batch*21+idx];
	P[25] = Hp[batch*22+idx];
	P[26] = Hp[batch*23+idx];

	P[27] = Hp[batch*3+idx];
	P[28] = Hp[batch*11+idx];
	P[29] = Hp[batch*18+idx];
	P[30] = Hp[batch*24+idx];
	P[31] = Hp[batch*25+idx];
	P[32] = Hp[batch*26+idx];
	P[33] = Hp[batch*27+idx];
	P[34] = Hp[batch*28+idx];
	P[35] = Hp[batch*29+idx];

	P[36] = Hp[batch*4+idx];
	P[37] = Hp[batch*12+idx];
	P[38] = Hp[batch*19+idx];
	P[39] = Hp[batch*25+idx];
	P[40] = Hp[batch*30+idx];
	P[41] = Hp[batch*31+idx];
	P[42] = Hp[batch*32+idx];
	P[43] = Hp[batch*33+idx];
	P[44] = Hp[batch*34+idx];

	P[45] = Hp[batch*5+idx];
	P[46] = Hp[batch*13+idx];
	P[47] = Hp[batch*20+idx];
	P[48] = Hp[batch*26+idx];
	P[49] = Hp[batch*31+idx];
	P[50] = Hp[batch*35+idx];
	P[51] = Hp[batch*36+idx];
	P[52] = Hp[batch*37+idx];
	P[53] = Hp[batch*38+idx];

	P[54] = Hp[batch*6+idx];
	P[55] = Hp[batch*14+idx];
	P[56] = Hp[batch*21+idx];
	P[57] = Hp[batch*27+idx];
	P[58] = Hp[batch*32+idx];
	P[59] = Hp[batch*36+idx];
	P[60] = Hp[batch*39+idx];
	P[61] = Hp[batch*40+idx];
	P[62] = Hp[batch*41+idx];

	P[63] = Hp[batch*7+idx];
	P[64] = Hp[batch*15+idx];
	P[65] = Hp[batch*22+idx];
	P[66] = Hp[batch*28+idx];
	P[67] = Hp[batch*33+idx];
	P[68] = Hp[batch*37+idx];
	P[69] = Hp[batch*40+idx];
	P[70] = Hp[batch*42+idx];
	P[71] = Hp[batch*43+idx];

	P[72] = Hp[batch*8+idx];
	P[73] = Hp[batch*16+idx];
	P[74] = Hp[batch*23+idx];
	P[75] = Hp[batch*29+idx];
	P[76] = Hp[batch*34+idx];
	P[77] = Hp[batch*38+idx];
	P[78] = Hp[batch*41+idx];
	P[79] = Hp[batch*43+idx];
	P[80] = Hp[batch*44+idx];

	if(mult) 
	{
		P[0] = (P[0] + (P[0]*l));
		P[10] = (P[10] + (P[10]*l));
		P[20] = (P[20] + (P[20]*l));
		P[30] = (P[30] + (P[30]*l));
		P[40] = (P[40] + (P[40]*l));
		P[50] = (P[50] + (P[50]*l));
		P[60] = (P[60] + (P[60]*l));
		P[70] = (P[70] + (P[70]*l));
		P[80] = (P[80] + (P[80]*l));
	}else{

		P[0] += l;
		P[10] += l;
		P[20] += l;
		P[30] += l;
		P[40] += l;
		P[50] += l;
		P[60] += l;
		P[70] += l;
		P[80] += l;
	}

	for(int i=0; i< dim; i++) b[i] = (T) gp[batch*i+idx];
	for(int i=0; i< dim; i++) x[i] = 0;

	int i, j;
	T  d;
	T* Ucoli;	

	// decompose P = LLT
    for( Ucoli=P, i=0; i<dim; ++i, Ucoli+=dim)
    {  
        d = Ucoli[i] - dot_prod<T>(Ucoli, Ucoli, i);
        Ucoli[i] = __fsqrt_rn(d);
        d = __frcp_rn(Ucoli[i]);
        for( j=i+1; j<dim; ++j)
        {   
            P[i+j*dim] = d*(P[i+j*dim] - dot_prod<T>(Ucoli, P+j*dim, i));			
        }
    }

	// intermediate solution y
	T y[9];

	//forwardsub
    for( int i=0; i<dim; i++)
    {  
		T s = 0;
		for(int j=0; j < i; j++)
			s += y[j]*P[i*dim+j];
		y[i]= (b[i]-s)/P[i*dim+i];
    }

	//backwardsub
    for( int j=dim-1; j>=0; j--)
    {  
		T s = 0;
		for(int i=dim-1; i > j; i--)
			s += x[i]*P[i*dim+j];
		x[j]= (y[j]-s)/P[j*dim+j];
    }

	// store result x
	for(int i=0; i< dim; i++) dp[batch*i+idx] = (isnan(x[i])||isinf(x[i]))?0:(float)x[i];
}