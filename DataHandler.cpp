#include "DataHandler.h"


using namespace std;

// sort array ascending
void quickSort1(int arr1[], int arr2[], float arr3[], float arr4[], int left, int right) {

      int i = left, j = right;
      int tmp,tmp2;
	  float tmp3,tmp4;
       
	  int pivot = arr1[(left + right) / 2];

      /* partition */
      while (i <= j) {
            while (arr1[i] < pivot)
                  i++;
            while (arr1[j] > pivot)
                  j--;
            if (i <= j) {
				  //cidx,pidx
                  tmp = arr1[i];
				  tmp2 = arr2[i];
                  arr1[i] = arr1[j];
				  arr2[i] = arr2[j];
                  arr1[j] = tmp;
				  arr2[j] = tmp2;
				  //measurement
				  if(arr3!=0){
					  tmp3 = arr3[i];
					  tmp4 = arr4[i];
					  arr3[i] = arr3[j];
					  arr4[i] = arr4[j];
					  arr3[j] = tmp3;
					  arr4[j] = tmp4;
				  }
                  i++;
                  j--;
            }
      };

      /* recursion */
      if (left < j)
            quickSort1(arr1, arr2, arr3, arr4, left, j);
      if (i < right)
            quickSort1(arr1, arr2, arr3, arr4, i, right);

}

// sort array ascending ( not used for final tests )
void quickSort2(int arr0[], int arr1[], int arr2[], float arr3[], float arr4[], int left, int right) {

      int i = left, j = right;
	  int tmp0;
      int tmp,tmp2;
	  float tmp3,tmp4;
       
	  int pivot = arr0[(left + right) / 2];

      /* partition */
      while (i <= j) {
            while (arr0[i] < pivot)
                  i++;
            while (arr0[j] > pivot)
                  j--;
            if (i <= j) {
				  //cidx,pidx
				  tmp0 = arr0[i];
                  tmp = arr1[i];
				  tmp2 = arr2[i];
				  arr0[i] = arr0[j];
                  arr1[i] = arr1[j];
				  arr2[i] = arr2[j];
				  arr0[j] = tmp0;
                  arr1[j] = tmp;
				  arr2[j] = tmp2;
				  //measurement
				  if(arr3!=0){
					  tmp3 = arr3[i];
					  tmp4 = arr4[i];
					  arr3[i] = arr3[j];
					  arr4[i] = arr4[j];
					  arr3[j] = tmp3;
					  arr4[j] = tmp4;
				  }
                  i++;
                  j--;
            }
      };

      /* recursion */
      if (left < j)
            quickSort2(arr0, arr1, arr2, arr3, arr4, left, j);
      if (i < right)
            quickSort2(arr0, arr1, arr2, arr3, arr4, i, right);

}

// transform array to another array necessary for the final repeated sort ( not used for final tests )
void mytransform(int in[], int out[], int Nc, int Nm){

	int prev = in[0];
	out[0] = in[0];
	int v = 1;
	int maxElem = Nc;
	if(Nc==0)
		maxElem = in[Nm-1]+1;

	for(int i = 1; i< Nm ;i++){

		if(in[i] == prev){
			out[i] = in[i] + maxElem*v;
			v += 2;
		}
		else{
			out[i] = in[i];
			v = 1;
		}
		prev = in[i];
		
	}
}

inline void SetRodriguesRotation(const float r[3], float R[3][3]){

        double a = sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
        double ct = a==0.0?0.5:(1.0-cos(a))/(a*a);
        double st = a==0.0?1:sin(a)/a;

        R[0][0]=float(1.0 - (r[1]*r[1] + r[2]*r[2])*ct);
        R[0][1]=float(r[0]*r[1]*ct - r[2]*st);
        R[0][2]=float(r[2]*r[0]*ct + r[1]*st);
        R[1][0]=float(r[0]*r[1]*ct + r[2]*st);
        R[1][1]=float(1.0 - (r[2]*r[2] + r[0]*r[0])*ct);
        R[1][2]=float(r[1]*r[2]*ct - r[0]*st);
        R[2][0]=float(r[2]*r[0]*ct - r[1]*st);
        R[2][1]=float(r[1]*r[2]*ct + r[0]*st);
        R[2][2]=float(1.0 - (r[0]*r[0] + r[1]*r[1])*ct);

		for(int i = 3; i < 9; ++i) 
			R[0][i] = - R[0][i];

}

bool parseGeneralConfig(char *init, BAmode &im, Solver &sm, char* &path, bool &fix, bool &del, bool &reorder, bool &prprog, bool &trace, bool &smsse){

	if(init == NULL) return false;
    ifstream in(init);
	if(!in.is_open()) return false;

	char buffer[256];
	path = (char*) malloc(sizeof(char)*50);
	
	while(!in.eof()){

		in.getline(buffer, sizeof(buffer));		
		if(buffer[0] != '-') continue;
		string str = (string) buffer;		


		// BAmode
		if(str.find("-BAmode:metric") != string::npos) im = metric;		
		if(str.find("-BAmode:focal_radial") != string::npos) im = focal_radial;
		
		// Solver
		if(str.find("-Solver:LMA") != string::npos) sm = LMA;		
		if(str.find("-Solver:RI") != string::npos) sm = RI;		
		if(str.find("-Solver:NCG") != string::npos) sm = NCG;
		
		// path
		int pos = str.find("-path:");
		if(pos != string::npos) 
			strncpy(path, &buffer[pos+6], 50);
		
		// preprocessing
		if(str.find("-degeneracy_fix:true") != string::npos) fix = true;
		if(str.find("-degeneracy_fix:false") != string::npos) fix = false;

		if(str.find("-del_proj:true") != string::npos) del = true;
		if(str.find("-del_proj:false") != string::npos) del = false;

		if(str.find("-reorder:true") != string::npos) reorder = true;
		if(str.find("-reorder:false") != string::npos) reorder = false;

		// printing
		if(str.find("-print_progress:true") != string::npos) prprog = true;
		if(str.find("-print_progress:false") != string::npos) prprog = false;

		if(str.find("-trace:true") != string::npos) trace = true;
		if(str.find("-trace:false") != string::npos) trace = false;

		if(str.find("-show_min_sse:true") != string::npos) smsse = true;
		if(str.find("-show_min_sse:false") != string::npos) smsse = false;
	}

	in.close();
	return true;

}

bool parseSolverConfig(char *init, int &maxI, int &cg_minI, int &cg_maxI, int &epi, float &gradM_stop, float &mse_stop, float &diff_stop, int &il, float &maxL, float &minL,
					  float &tau, float &n_fs, bool &schur, bool &backsub, bool &diagdamping, float &t, float &c1, float &c2, float &start_a, int &b_reset, float &hyb_sw){

	if(init == NULL) return false;
    ifstream in(init);
	if(!in.is_open()) return false;

	char buffer[256];
	
	while(!in.eof()){

		in.getline(buffer, sizeof(buffer));		
		if(buffer[0] != '-') continue;
		string str = (string) buffer;

		int pos = -1;

		// stopping tresholds
		pos = str.find("-gradM_stop:");
		if(pos != string::npos) gradM_stop = atof(&buffer[12+pos]);
		pos = str.find("-mse_stop:");
		if(pos != string::npos) mse_stop = atof(&buffer[10+pos]);
		pos = str.find("-diff_stop:");
		if(pos != string::npos) diff_stop = atof(&buffer[11+pos]);

		// iteration numbers
		pos = str.find("-maxI:");
		if(pos != string::npos) maxI = atof(&buffer[6+pos]);

		// initial damping factor
		pos = str.find("-tau:");
		if(pos != string::npos) tau = atof(&buffer[5+pos]);

		// min/max lambda relative to the initial damping factor (tau)
		pos = str.find("-maxL:");
		if(pos != string::npos) maxL = atof(&buffer[6+pos]);
		pos = str.find("-minL:");
		if(pos != string::npos) minL = atof(&buffer[6+pos]);

		// LMA specific
		pos = str.find("-cg_maxI:");
		if(pos != string::npos) cg_maxI = atof(&buffer[9+pos]);
		pos = str.find("-cg_minI:");
		if(pos != string::npos) cg_minI = atof(&buffer[9+pos]);
		pos = str.find("-epi:");
		if(pos != string::npos) epi = atof(&buffer[5+pos]);
		pos = str.find("-n_fs:");
		if(pos != string::npos) n_fs = atof(&buffer[6+pos]);
		pos = str.find("-hyb_switch:");
		if(pos != string::npos) hyb_sw = atof(&buffer[12+pos]);

		if(str.find("-diagdamping:true") != string::npos) diagdamping = true;
		if(str.find("-diagdamping:false") != string::npos) diagdamping = false;

		if(str.find("-backsub:true") != string::npos) backsub = true;
		if(str.find("-backsub:false") != string::npos) backsub = false;

		if(str.find("-schur:true") != string::npos) schur = true;
		if(str.find("-schur:false") != string::npos) schur = false;

		// NCG specific
		pos = str.find("-b_reset:");
		if(pos != string::npos) b_reset = atof(&buffer[9+pos]);
		pos = str.find("-start_a:");
		if(pos != string::npos) start_a = atof(&buffer[9+pos]);
		pos = str.find("-t:");
		if(pos != string::npos) t = atof(&buffer[3+pos]);
		pos = str.find("-c1:");
		if(pos != string::npos) c1 = atof(&buffer[4+pos]);
		pos = str.find("-c2:");
		if(pos != string::npos) c2 = atof(&buffer[4+pos]);

		// RI specific
		pos = str.find("-inner_loops:");
		if(pos != string::npos) il = atof(&buffer[13+pos]);
	}

	in.close();
	return true;

}

bool LoadBundlerModelBAL(char* fname, float*& camera_data, float*& point_data,
              float*& measurements, int*& ptidx, int*& camidx, int& n_measurements, int& n_cameras, int& n_points)
{

	if(fname == NULL)return false;
    ifstream in(fname); 

    std::cout << "Loading cameras/points: " << fname <<"\n" ;
    if(!in.is_open()) return false;

    // read bundle data from a file

    size_t ncam = 0, npt = 0, nproj = 0;
    if(!(in >> ncam >> npt >> nproj)) return false;
    ///////////////////////////////////////////////////////////////////////////////
    std::cout << ncam << " cameras; " << npt << " 3D points; " << nproj << " projections\n";
	n_measurements = nproj;
	n_cameras = ncam;
	n_points = npt;

	// scale parameter
	float c = 0.5;
	vector<float> focals(ncam);
	vector<float> zs(npt);
	float medF = 0;
	float medz = 0;

	// allocate input data space
	camera_data = (float*) malloc(9 * ncam * sizeof(float));
	point_data = (float*) malloc(3 * npt * sizeof(float));
	measurements = (float*) malloc(2 * nproj * sizeof(float));
    camidx = (int*) malloc(nproj * sizeof(int));
    ptidx = (int*) malloc(nproj * sizeof(int));


    for(size_t i = 0; i < nproj; ++i)    
    {
        float x, y;    int cidx, pidx;
        in >> cidx >> pidx >> x >> y;
        if(((size_t) pidx) == npt && ncam > i) 
        {
			free(camidx);
            camidx = (int*) malloc(i * sizeof(float));
			free(ptidx);
            ptidx = (int*) malloc(i * sizeof(float));
			free(measurements);
            measurements = (float*) malloc(2 * i * sizeof(float));
		

            std::cout << "Truncate measurements to " << i << '\n';
        }else if(((size_t) pidx) >= npt)
        {
            continue;
        }else
        {
            camidx[i] = cidx;    
			ptidx[i] = pidx;
            measurements[i]         = x;
			measurements[nproj + i] = -y;
        }
    }

    for(size_t i = 0; i < ncam; ++i)
    {
        float p[9];
        for(int j = 0; j < 9; ++j) in >> p[j];
		for(int j = 0; j < 3; j++) camera_data[i + ncam*j] = p[j];  // r
		camera_data[i + ncam*3] = p[3];  //tx
		camera_data[i + ncam*4] = -p[4]; //ty
		camera_data[i + ncam*5] = -p[5]; //tz
        camera_data[i + ncam*6] = p[6];  //f      
        camera_data[i + ncam*7] = p[7];  //k1
		camera_data[i + ncam*8] = p[8];  //k2
		focals[i] = p[6];
    }


    for(size_t i = 0; i < npt; ++i)
    {
        float pt[3];
        in >> pt[0] >> pt[1] >> pt[2];
        for(int j = 0; j < 3; j++) point_data[i + npt*j] = pt[j];
		zs[i] = pt[2];
    }

	in.close();
	cout << "...done" << endl;
    return true;

};

void PreProcessing(float*& camera_data, float*& point_data,
              float*& measurements, int*& ptidx, int*& camidx, int& n_measurements, int& n_cameras, int& n_points, bool fix, bool deleteProj, bool reorder)
{
		
		
		const float     dist_bound = 1.0f;
        vector<float>   oz(n_measurements);
        vector<float>   cpdist1(n_cameras,  dist_bound);
        vector<float>   cpdist2(n_cameras, -dist_bound); 
        vector<int>     camnpj(n_cameras, 0), cambpj(n_cameras, 0);
		vector<int>		ill_cam(n_cameras, 0);

        int bad_point_count = 0; 

		int __num_point_behind = 0;
		int __num_camera_modified = 0;
		int __deletedProj = 0;
		bool __deleteProj = deleteProj;

		// fix ill-conditioned cameras 
		// extracted and modified from PBA
		if(fix){

			for(int i = 0; i < n_measurements; ++i)
			{
				int cmidx = camidx[i];
				//CameraT * cam = camera_data + cmidx;
				float cam_rot[3], cam_tr[3], cam_f;
				cam_rot[0] = camera_data[0*n_cameras + cmidx];
				cam_rot[1] = camera_data[1*n_cameras + cmidx];
				cam_rot[2] = camera_data[2*n_cameras + cmidx];

				cam_tr[0] = camera_data[3*n_cameras + cmidx];
				cam_tr[1] = camera_data[4*n_cameras + cmidx];
				cam_tr[2] = camera_data[5*n_cameras + cmidx];
				cam_f     = camera_data[6*n_cameras + cmidx];

				// get last row of rot matrix
				double a = sqrt(cam_rot[0]*cam_rot[0]+cam_rot[1]*cam_rot[1]+cam_rot[2]*cam_rot[2]);
				double ct = a==0.0?0.5:(1.0-cos(a))/(a*a);
				double st = a==0.0?1:sin(a)/a;
				float rz[3][3];
				SetRodriguesRotation(cam_rot, rz);

				//float *rz = cam->m[2];
				float x[3];
				x[0] = point_data[ptidx[i]];
				x[1] = point_data[1*n_points + ptidx[i]];
				x[2] = point_data[2*n_points + ptidx[i]];

				oz[i] = (rz[2][0]*x[0]+rz[2][1]*x[1]+rz[2][2]*x[2]+ cam_tr[2]);

				/////////////////////////////////////////////////
				//points behind camera may cause big problems 
				float ozr = oz[i] / cam_tr[2];
				if(fabs(ozr) < 0.01f) 
				{
					bad_point_count++; 
					float px = cam_f * (rz[0][0]*x[0]+rz[0][1]*x[1]+rz[0][2]*x[2]+ cam_tr[0]);
					float py = cam_f * (rz[1][0]*x[0]+rz[1][1]*x[1]+rz[1][2]*x[2]+ cam_tr[1]);
					float mx = measurements[i], my = measurements[ n_measurements + i];
					bool checkx = fabs(mx) > fabs(my);
					if( ( checkx && px * oz[i] * mx < 0 && fabs(mx) > 64) || (!checkx && py * oz[i] * my < 0 && fabs(my) > 64)) 
					{ 
						if(oz[i] > 0)     cpdist2[cmidx] = 0;
						else              cpdist1[cmidx] = 0;
					}
					if(oz[i] >= 0) cpdist1[cmidx] = std::min(cpdist1[cmidx], oz[i]);
					else           cpdist2[cmidx] = std::max(cpdist2[cmidx], oz[i]); 
				}
				if(oz[i] < 0) { __num_point_behind++;   cambpj[cmidx]++;}
				camnpj[cmidx]++;
			}

			std::nth_element(oz.begin(), oz.begin() + n_measurements / 2, oz.end());
			float oz_median = oz[n_measurements / 2];
			float shift_min = std::min(oz_median * 0.001f, 1.0f);
			float dist_threshold = shift_min * 0.1f;

			for(int i = 0; i < n_cameras; ++i)
			{
				//move the camera a little bit?
				if((cpdist1[i] < dist_threshold || cpdist2[i] > -dist_threshold) )
				{
                
					bool  boths = cpdist1[i] < dist_threshold && cpdist2[i] > -dist_threshold;
					if(!__deleteProj){
						camera_data[5*n_cameras + i] += shift_min;
						 __num_camera_modified++;
					}
					else
						ill_cam[i] = 1;                 
				 }
			}
			// delete ill-conditioned projections
			if(__deleteProj){

				
				for(int i = 0; i < n_measurements; ++i)
					if(ill_cam[camidx[i]]) __deletedProj++;

				float *temp1;
				int *temp2,*temp3;
				temp1 = (float*) malloc(2*sizeof(float)*(n_measurements-__deletedProj));
				temp2 = (int*) malloc(sizeof(float)*(n_measurements-__deletedProj));
				temp3 = (int*) malloc(sizeof(float)*(n_measurements-__deletedProj));

				int idx = 0;
				for(int i = 0; i < n_measurements; ++i)
				{
					int cmidx = camidx[i];
					if(!ill_cam[cmidx]){
						temp3[idx] = camidx[i];
						temp2[idx] = ptidx[i];
						temp1[idx] = measurements[i];
						temp1[idx+(n_measurements-__deletedProj)] = measurements[i + (n_measurements)];
						idx++;
					}
				}
						
				n_measurements -= __deletedProj;
				measurements = temp1;
				ptidx = temp2;
				camidx = temp3;
			}
        
		
			if(__num_point_behind > 0)    std::cout << "WARNING: " << __num_point_behind << " points are behind cameras.\n";
			if(__num_camera_modified > 0) std::cout << "WARNING: " << __num_camera_modified << " cameras moved to avoid degeneracy.\n";
			if(__deletedProj > 0) std::cout << "WARNING: " << __deletedProj << " projections deleted to avoid degeneracy.\n";

	}


	//////////////////////////////// Reorderings
	if(reorder){
		int mode = 4; // 0=KP; 1=PK; 2=P-R; 3=K-R; 4=PK-B; 5=KP-B

		if(mode==0) // Sort: Cameras full -> Points per Camera
		{
			quickSort1(camidx, ptidx, measurements, &measurements[n_measurements], 0, n_measurements-1);
			int t = 0;
			for(int i=1; i< n_measurements; i++) {
				if(camidx[i] != camidx[i-1]){
					quickSort1(ptidx, camidx, measurements, &measurements[n_measurements], t, i-1);
					t = i;
				}
			}
		}
		if(mode==1) // Sort: Points full -> Cameras per Point (default for BAL datasets)
		{}
		if(mode==2) // Sort: Points repeated
		{
			//quickSort1(ptidx, camidx, measurements, &measurements[n_measurements], 0, n_measurements-1);
			int *ptemp = (int*) malloc(sizeof(int) * n_measurements);
			mytransform(ptidx, ptemp, n_points-1, n_measurements);
			quickSort2(ptemp, ptidx, camidx, measurements, &measurements[n_measurements], 0, n_measurements-1);
		}
		if(mode==3) // Sort: Cameras repeated
		{
			quickSort1(camidx, ptidx, measurements, &measurements[n_measurements], 0, n_measurements-1);
			int *ctemp = (int*) malloc(sizeof(int) * n_measurements);
			mytransform(camidx, ctemp, n_cameras-1, n_measurements);
			quickSort2(ctemp, camidx, ptidx, measurements, &measurements[n_measurements], 0, n_measurements-1);
		}
		if(mode==4) // Sort: Points first -> Cameras within Points -> Cameras blocked -> Points within Camera
		{
			//int bs = n_measurements/10000;
			int bs = (int) (((float)n_cameras/(float)n_points)*(float)n_measurements)/20;
			if(bs<=32) return;
			int n_block_ov = (int)( ( (float)(n_measurements)/(float)bs ) +1 );

			for(int i=0; i< (n_block_ov-1); i++) quickSort1(camidx, ptidx, measurements, &measurements[n_measurements], i*bs, (i*bs)+bs);
			int t = 0;
			for(int i=1; i< n_measurements; i++) {
				if(camidx[i] != camidx[i-1]){
					quickSort1(ptidx, camidx, measurements, &measurements[n_measurements], t, i-1);
					t = i;
				}
			}
		}
		if(mode==5) // Sort: Cameras first -> Points within Cameras -> Points blocked -> Cameras within Point
		{
			int bs = n_measurements/10000;
			int n_block_ov = (int)( ( (float)(n_measurements)/(float)bs ) +1 );

			quickSort1(camidx, ptidx, measurements, &measurements[n_measurements], 0, n_measurements-1);
			for(int i=0; i< (n_block_ov-1); i++) quickSort1(ptidx, camidx, measurements, &measurements[n_measurements], i*bs, (i*bs)+bs);

			int t = 0;
			for(int i=1; i< n_measurements; i++) {
				if(ptidx[i] != ptidx[i-1]){
					quickSort1(camidx, ptidx, measurements, &measurements[n_measurements], t, i-1);
					t = i;
				}
			}
		}
	}

}