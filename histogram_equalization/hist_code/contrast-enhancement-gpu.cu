#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <cuda_runtime.h>
#include <assert.h> 


PGM_IMG contrast_enhancement_g_gpu(PGM_IMG img_in)
{	
  // Timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
	
  PGM_IMG result;
  int hist[256];
  int cdf[256];
  int imgsize;
  int nbr_bin; 
  int min, d, i;
  int lines;

  // device variables
  int *d_hist;
  unsigned char *d_img;
  int *d_imgsize;
  int *d_nbr_bin;
  unsigned char *d_result;
  int *d_min;
  int *d_d;
  int *d_cdf;
  int *d_lines;

  result.w = img_in.w;
  result.h = img_in.h;
  result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
  imgsize = img_in.w * img_in.h;
  nbr_bin = 256;
  lines = (img_in.w*img_in.h)/256;
  
  // device memory allocation and copying data over
  cudaMalloc(&d_hist, sizeof(int) * 256);
  cudaMalloc(&d_img, img_in.w * img_in.h * sizeof(unsigned char));
  cudaMalloc(&d_imgsize, sizeof(int));
  cudaMalloc(&d_nbr_bin, sizeof(int));
  cudaMalloc(&d_lines, sizeof(int));
  cudaMemcpy(d_img, img_in.img, img_in.w * img_in.h * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_imgsize, &imgsize, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nbr_bin, &nbr_bin, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lines, &lines, sizeof(int), cudaMemcpyHostToDevice);
  // gpu function
  
  cudaEventRecord(start);
  histogram_gpu<<<1, 256>>>( d_hist, d_img, d_imgsize, d_nbr_bin, d_lines );	
  cudaEventRecord(stop);
  cudaMemcpy(hist, d_hist, sizeof(int)*256, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  //printf("GPU MILISEC: %f\n", milliseconds * 0.001);
  // cdf needs to be constructed sequentially here
  cdf[0] = hist[0];
  for(i = 1; i < 256; ++i){ 
	  cdf[i] = hist[i] + cdf[i-1];
  }
  i = 0;
  min = 0;
  while(min == 0){
        min = hist[i++];
    }
  d = imgsize - min;
  
  // device memory allocation and copying data over
  cudaMalloc(&d_min, sizeof(int));
  cudaMalloc(&d_d  , sizeof(int));
  cudaMalloc(&d_result, sizeof(unsigned char) * img_in.w * img_in.h);
  cudaMalloc(&d_cdf, sizeof(int) * 256);
  cudaMemcpy(d_min, &min, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_d  , &d  , sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cdf, cdf , sizeof(int)*256, cudaMemcpyHostToDevice);
  
  // gpu function
  histogram_equalization_gpu<<<1,256>>>(d_result, d_img, d_hist, d_imgsize, d_min, d_d, d_cdf);
  cudaMemcpy(result.img, d_result, sizeof(unsigned char) * img_in.w * img_in.h, cudaMemcpyDeviceToHost);
 

  // free device memory
  cudaFree( d_cdf );
  cudaFree( d_result );
  cudaFree( d_min);
  cudaFree( d_d );
  cudaFree( d_hist );
  cudaFree( d_img  );
  cudaFree( d_imgsize );
  cudaFree( d_nbr_bin  );
	
	
  //histogram(hist, img_in.img, img_in.h * img_in.w, 256);
	
  //histogram_equalization(result.img, img_in.img, hist, result.w*result.h, 256);
								
  return result;
}


PPM_IMG contrast_enhancement_c_yuv_gpu(PPM_IMG img_in){
	// Timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    PPM_IMG result;
	result.w = img_in.w;
	result.h = img_in.h;
	
	int imgsize = (img_in.h * img_in.w);
	PGM_IMG temp;
		
	int lines = ceil( imgsize / THREADS ) ;
	
	// Device Variables 
	unsigned char *d_img_in_r;
	unsigned char *d_img_in_g;
	unsigned char *d_img_in_b;
	unsigned char *d_img_out_y;
	unsigned char *d_img_out_u;
	unsigned char *d_img_out_v;
	int *d_lines;
	int *d_imgsize;
	float milliseconds = 0;
	
	cudaMalloc( &d_img_in_r , sizeof(unsigned char) * imgsize );
	cudaMemcpy(d_img_in_r, img_in.img_r , sizeof(unsigned char) * imgsize , cudaMemcpyHostToDevice);
	cudaMalloc( &d_img_in_g , sizeof(unsigned char) * imgsize  );
	cudaMemcpy(d_img_in_g, img_in.img_g , sizeof(unsigned char) * imgsize , cudaMemcpyHostToDevice);
	cudaMalloc( &d_img_in_b , sizeof(unsigned char) * imgsize );
	cudaMemcpy(d_img_in_b, img_in.img_b , sizeof(unsigned char) * imgsize , cudaMemcpyHostToDevice);
	
	cudaMalloc( &d_img_out_y , sizeof(unsigned char) * img_in.h * img_in.w );
	cudaMalloc( &d_img_out_u , sizeof(unsigned char) * img_in.h * img_in.w );
	cudaMalloc( &d_img_out_v , sizeof(unsigned char) * img_in.h * img_in.w );
	cudaMalloc( &d_lines  , sizeof(int) );
	cudaMalloc( &d_imgsize, sizeof(int) );
	
	cudaMemcpy( d_lines, &lines, sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_imgsize, &imgsize, sizeof(int), cudaMemcpyHostToDevice );
	
	cudaEventRecord(start);
	rbg2yuv_gpu<<<1, THREADS >>>(d_img_in_r, d_img_in_g, d_img_in_b, d_img_out_y, d_img_out_u, d_img_out_v, d_lines, d_imgsize);
	
	
	temp.w = img_in.w;
	temp.h = img_in.h;	
	temp.img = (unsigned char *)malloc( sizeof(unsigned char) * temp.w * temp.h);
	
	cudaMemcpy(temp.img, d_img_out_y, sizeof(unsigned char) * imgsize, cudaMemcpyDeviceToHost);
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	// do histogram stuff here -- no diff than gray scale so just use the same function.
	temp = contrast_enhancement_g_gpu(temp);
	cudaMemcpy(d_img_out_y, temp.img, sizeof(unsigned char) * imgsize, cudaMemcpyHostToDevice);
	yuv2rbg_gpu<<<1, THREADS >>>(d_img_in_r, d_img_in_g, d_img_in_b, d_img_out_y, d_img_out_u, d_img_out_v, d_lines, d_imgsize);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("contrast_enhancement_c_yuv_gpu_naive: %f\n", milliseconds * 0.001);
	
	// init result arrays while gpu is doing work
	result.img_r = (unsigned char *)malloc( sizeof(unsigned char) * imgsize );
	result.img_g = (unsigned char *)malloc( sizeof(unsigned char) * imgsize );
	result.img_b = (unsigned char *)malloc( sizeof(unsigned char) * imgsize );
	
	// cpy over gpu work to the return array
	cudaMemcpy( result.img_r , d_img_in_r, sizeof(unsigned char) * imgsize, cudaMemcpyDeviceToHost);
	cudaMemcpy( result.img_g , d_img_in_g, sizeof(unsigned char) * imgsize, cudaMemcpyDeviceToHost);
	cudaMemcpy( result.img_b , d_img_in_b, sizeof(unsigned char) * imgsize, cudaMemcpyDeviceToHost);

	// free cuda variables
	cudaFree( d_img_in_r );
	cudaFree( d_img_in_g );
	cudaFree( d_img_in_b );
	cudaFree( d_img_out_y );
	cudaFree( d_img_out_u );
	cudaFree( d_img_out_v );
	cudaFree( d_lines );
	cudaFree( d_imgsize );
	free(temp.img);
	
    return result;
}



__global__ void rbg2yuv_gpu(unsigned char *img_in_r, unsigned char *img_in_g, unsigned char *img_in_b, unsigned char *img_out_y, unsigned char *img_out_u, unsigned char *img_out_v, int *lines, int *imgsize){
	
	unsigned char r,g,b;
    int start = (threadIdx.x + blockIdx.x * blockDim.x) * (*lines);
	
	int i;
	//printf("Thread[%d] working from %d to %d\n", threadIdx.x, start, start + *lines);
	
    for ( i = start; i < start + *lines && (i < *imgsize); ++i){
		r = img_in_r[i];
		g = img_in_g[i];
	    b = img_in_b[i];
		img_out_y[i] = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
        img_out_u[i] = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
        img_out_v[i] = (unsigned char)( 0.499*r - 0.418*g -  0.0813*b + 128);
    }
}

//Convert YUV to RGB, all components in [0, 255]
__global__ void yuv2rbg_gpu(unsigned char *img_out_r, unsigned char *img_out_g, unsigned char *img_out_b, unsigned char *img_in_y, unsigned char *img_in_u, unsigned char *img_in_v, int *lines, int *imgsize){
	int y,cb,cr, rt, gt, bt;
    int start = (threadIdx.x + blockIdx.x * blockDim.x) * (*lines);
	int i;
	
    for ( i = start; i < start + *lines && (i < *imgsize); ++i){
        y  = (int)img_in_y[i];
        cb = (int)img_in_u[i] - 128;
        cr = (int)img_in_v[i] - 128; 
		rt  = (int)( y + 1.402*cr);
        gt  = (int)( y - 0.344*cb - 0.714*cr);
        bt  = (int)( y + 1.772*cb);
		
		if(rt > 255){
			rt = 255;
		}else if(rt < 0){
			rt = 0;
		}
		
		if(gt > 255){
			gt = 255;
		}else if(gt < 0){
			gt = 0;
		}
		
		if(bt > 255){
			bt = 255;
		}else if(bt < 0){
			bt = 0;
		}
		img_out_r[i] = (unsigned char) rt;
		img_out_b[i] = (unsigned char) bt;
		img_out_g[i] = (unsigned char) gt;
    }
}


PPM_IMG contrast_enhancement_c_hsl_gpu(PPM_IMG img_in){
	//timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	
	PPM_IMG result;
	PGM_IMG temp;
	temp.w = img_in.w;
	temp.h = img_in.h;
	result.w = img_in.w;
	result.h = img_in.h;
	int imgsize = img_in.w * img_in.h;	
	int lines = ceil( imgsize / THREADS ) ;
	
	// Device Variables
	unsigned char *d_img_in_r;
	unsigned char *d_img_in_g;
	unsigned char *d_img_in_b;
	float *d_img_out_h;
	float *d_img_out_s;
	unsigned char *d_img_out_l;
	int *d_lines;
	int *d_imgsize;
	
	cudaMalloc( &d_img_in_r , sizeof(unsigned char) * imgsize );
	cudaMemcpy( d_img_in_r, img_in.img_r , sizeof(unsigned char) * imgsize , cudaMemcpyHostToDevice);
	cudaMalloc( &d_img_in_g , sizeof(unsigned char) * imgsize  );
	cudaMemcpy( d_img_in_g, img_in.img_g , sizeof(unsigned char) * imgsize , cudaMemcpyHostToDevice);
	cudaMalloc( &d_img_in_b , sizeof(unsigned char) * imgsize );
	cudaMemcpy( d_img_in_b, img_in.img_b , sizeof(unsigned char) * imgsize , cudaMemcpyHostToDevice);
	
	cudaMalloc( &d_img_out_h , sizeof(float) * imgsize );
	cudaMalloc( &d_img_out_s , sizeof(float) * imgsize );
	cudaMalloc( &d_img_out_l , sizeof(unsigned char) * imgsize );
	cudaMalloc( &d_lines, sizeof(int) );
	cudaMalloc( &d_imgsize, sizeof(int) );
	
	cudaMemcpy( d_lines, &lines, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( d_imgsize, &imgsize, sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	rgb2hsl_gpu<<<1, THREADS>>>( d_img_in_r, d_img_in_g, d_img_in_b, d_img_out_h, d_img_out_s, d_img_out_l, d_lines, d_imgsize);
	if(!(temp.img = (unsigned char*)malloc(sizeof(unsigned char) * imgsize))){
		printf("Malloc failed\n");
		assert(0);
	}
	cudaMemcpy( temp.img, d_img_out_l, sizeof(unsigned char) * imgsize, cudaMemcpyDeviceToHost);
	
	temp = contrast_enhancement_g_gpu(temp);
	cudaMemcpy( d_img_out_l, temp.img, sizeof(unsigned char) * imgsize, cudaMemcpyHostToDevice);


	hsl2rgb_gpu<<<1, THREADS>>>( d_img_in_r, d_img_in_g, d_img_in_b, d_img_out_h, d_img_out_s, d_img_out_l, d_lines, d_imgsize);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	result.img_r = (unsigned char *)malloc( sizeof(unsigned char) * imgsize );
	result.img_g = (unsigned char *)malloc( sizeof(unsigned char) * imgsize );
	result.img_b = (unsigned char *)malloc( sizeof(unsigned char) * imgsize );
	
	printf("contrast_enhancement_c_hsl_gpu_naive: %f \n", milliseconds*0.001);
	// cpy over gpu work to the return array
	cudaMemcpy( result.img_r , d_img_in_r, sizeof(unsigned char) * imgsize, cudaMemcpyDeviceToHost);
	cudaMemcpy( result.img_g , d_img_in_g, sizeof(unsigned char) * imgsize, cudaMemcpyDeviceToHost);
	cudaMemcpy( result.img_b , d_img_in_b, sizeof(unsigned char) * imgsize, cudaMemcpyDeviceToHost);
		// free cuda variables
	cudaFree( d_img_in_r );
	cudaFree( d_img_in_g );
	cudaFree( d_img_in_b );
	cudaFree( d_img_out_h );
	cudaFree( d_img_out_s );
	cudaFree( d_img_out_l );
	cudaFree( d_lines );
	cudaFree( d_imgsize );
	free(temp.img);
	
	return result;
}

__global__ void rgb2hsl_gpu(unsigned char *img_in_r, unsigned char *img_in_g, unsigned char *img_in_b, float *img_out_h, float *img_out_s, unsigned char *img_out_l, int* lines, int *imgsize){
	
    int start = (threadIdx.x + blockIdx.x * blockDim.x) * (*lines);
	int i;
    float H, S, L;
	
	for ( i = start; i < start + *lines && (i < *imgsize); ++i){
		 
        float var_r = ( (float)img_in_r[i]/255 );//Convert RGB to [0,1]
        float var_g = ( (float)img_in_g[i]/255 );
        float var_b = ( (float)img_in_b[i]/255 );
        float var_min = (var_r < var_g) ? var_r : var_g;
        var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
        float var_max = (var_r > var_g) ? var_r : var_g;
        var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
        float del_max = var_max - var_min;               //Delta RGB value
        
        L = ( var_max + var_min ) / 2;
        if ( del_max == 0 )//This is a gray, no chroma...
        {
            H = 0;         
            S = 0;    
        }
        else                                    //Chromatic data...
        {
            if ( L < 0.5 )
                S = del_max/(var_max+var_min);
            else
                S = del_max/(2-var_max-var_min );

            float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
            float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
            float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
            if( var_r == var_max ){
                H = del_b - del_g;
            }
            else{       
                if( var_g == var_max ){
                    H = (1.0/3.0) + del_r - del_b;
                }
                else{
                        H = (2.0/3.0) + del_g - del_r;
                }   
            }
            
        }
        
        if ( H < 0 )
            H += 1;
        if ( H > 1 )
            H -= 1;

        img_out_h[i] = H;
        img_out_s[i] = S;
        img_out_l[i] = (unsigned char)(L*255);
	}
}

__global__ void hsl2rgb_gpu(unsigned char *img_out_r, unsigned char *img_out_g, unsigned char *img_out_b, float *img_in_h, float *img_in_s, unsigned char *img_in_l, int* lines, int *imgsize){

    int start = (threadIdx.x + blockIdx.x * blockDim.x) * (*lines);
	int i;
	for ( i = start; i < start + *lines && (i < *imgsize); ++i){
        float H = img_in_h[i];
        float S = img_in_s[i];
        float L = img_in_l[i]/255.0f;
        float var_1, var_2;
        
        unsigned char r,g,b;
        
        if ( S == 0 )
        {
            r = L * 255;
            g = L * 255;
            b = L * 255;
        }
        else
        {
            if ( L < 0.5 )
                var_2 = L * ( 1 + S );
            else
                var_2 = ( L + S ) - ( S * L );

            var_1 = 2 * L - var_2;
            r = 255 * Hue_2_RGB_gpu( var_1, var_2, H + (1.0f/3.0f) );
            g = 255 * Hue_2_RGB_gpu( var_1, var_2, H );
            b = 255 * Hue_2_RGB_gpu( var_1, var_2, H - (1.0f/3.0f) );
        }
        img_out_r[i] = r;
        img_out_g[i] = g;
        img_out_b[i] = b;
    }
}

__device__ float Hue_2_RGB_gpu( float v1, float v2, float vH ){
    if ( vH < 0 ) vH += 1;
    if ( vH > 1 ) vH -= 1;
    if ( ( 6 * vH ) < 1 ) return ( v1 + ( v2 - v1 ) * 6 * vH );
    if ( ( 2 * vH ) < 1 ) return ( v2 );
    if ( ( 3 * vH ) < 2 ) return ( v1 + ( v2 - v1 ) * ( ( 2.0f/3.0f ) - vH ) * 6 );
    return ( v1 );
}

PPM_IMG contrast_enhancement_c_rgb_gpu(PPM_IMG img_in)
{
	// Timing
	cudaEvent_t start, stop;
	float milliseconds = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
    PPM_IMG result;
    result.w = img_in.w;
    result.h = img_in.h;
	
	PGM_IMG temp;
	temp.w = img_in.w;
	temp.h = img_in.h;

	cudaEventRecord(start);
	
	temp.img = img_in.img_r;
	temp = contrast_enhancement_g_gpu( temp );
	result.img_r = temp.img;
	
	temp.img = img_in.img_g;
	temp = contrast_enhancement_g_gpu( temp );
	result.img_g = temp.img;
	
	temp.img = img_in.img_b;
	temp = contrast_enhancement_g_gpu( temp );	
	result.img_b = temp.img;
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("colour_enhancement_c_rgb_gpu_naive: %f\n", milliseconds * 0.001);
    return result;
}


