#include <stdio.h>
#include <string.h>
#include <stdlib.h>
//#include <cutil_inline.h>
#include <cuda_runtime.h>
#include "hist-equ.h"
#include <time.h>
#include <sys/time.h>

void run_cpu_color_test(PPM_IMG img_in);
void run_gpu_color_test(PPM_IMG img_in);
void run_cpu_gray_test(PGM_IMG img_in);
void run_gpu_gray_test(PGM_IMG img_in);
void run_color_test(PPM_IMG img_in);


int main(){
    PGM_IMG img_ibuf_g;
    PPM_IMG img_ibuf_c;
    
    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm("in.pgm");
    run_cpu_gray_test(img_ibuf_g);
    run_gpu_gray_test(img_ibuf_g);
    free_pgm(img_ibuf_g);
	
	printf("Running colour enhancement for colour images.\n");   
	img_ibuf_c = read_ppm("in.ppm");
	run_color_test(img_ibuf_c);
    free_ppm(img_ibuf_c);
    
    printf("Running contrast enhancement for color images.\n");
    img_ibuf_c = read_ppm("in.ppm");
    run_cpu_color_test(img_ibuf_c);
    run_gpu_color_test(img_ibuf_c);
    free_ppm(img_ibuf_c);
	
	printf("\n");
	printf("Running contrast enhancement for color 1920 images.\n");
    img_ibuf_c = read_ppm("in-1920.ppm");
    run_cpu_color_test(img_ibuf_c);
    run_gpu_color_test(img_ibuf_c);
    free_ppm(img_ibuf_c);
    
    return 0;
}

void run_color_test(PPM_IMG img_in){
	PPM_IMG cpu, gpu;
	struct timeval start, end;
    printf("Starting CPU processing...\n");
	gettimeofday(&start, NULL);
	cpu = contrast_enhancement_c_rgb( img_in );
	gettimeofday(&end, NULL);
	printf("colour_enhancement_c_rgb() %f\n", ((end.tv_sec + end.tv_usec * 0.000001)  - (start.tv_sec + start.tv_usec * 0.000001)));
    printf("Starting GPU processing...\n");
	gpu = contrast_enhancement_c_rgb_gpu( img_in );    
	gpu = contrast_enhancement_c_rgb_gpu_tiled( img_in );
	int equal = 1;
	int i;
	for ( i = 0; i < img_in.w*img_in.h; ++i){
		
		if( cpu.img_r[i] != gpu.img_r[i] )
			equal = 0;
		if( cpu.img_g[i] != gpu.img_g[i] )
			equal = 0;
		if( cpu.img_b[i] != gpu.img_b[i] )
			equal = 0;
	}
	if(!equal){
		printf("\nRGB NOT EQUAL!\n\n");
	}
	
}

void run_gpu_color_test(PPM_IMG img_in)
{
    int i;
	struct timeval start, end;
	PPM_IMG img_obuf_hsl, img_obuf_yuv;
	PPM_IMG img_obuf_hsl2, img_obuf_yuv2;
   
	
    printf("Starting GPU processing...\n");
	gettimeofday(&start, NULL);
    img_obuf_hsl = contrast_enhancement_c_hsl_gpu(img_in);
	gettimeofday(&end, NULL);
	//printf("contrast_enhancement_c_hsl_gpu() %f\n", ((end.tv_sec + end.tv_usec * 0.000001)  - (start.tv_sec + start.tv_usec * 0.000001)));

    write_ppm(img_obuf_hsl, "out_hsl_gpu.ppm");
    gettimeofday(&start, NULL);
	img_obuf_yuv = contrast_enhancement_c_yuv_gpu(img_in);
	gettimeofday(&end, NULL);
	//printf("contrast_enhancement_c_yuv_gpu() %f\n", ((end.tv_sec + end.tv_usec * 0.000001)  - (start.tv_sec + start.tv_usec * 0.000001)));
	
    write_ppm(img_obuf_yuv, "out_yuv_gpu.ppm"); 
 
	
	printf("Starting GPU-Tiled processing...\n");
	gettimeofday(&start, NULL);
    img_obuf_hsl2 = contrast_enhancement_c_hsl_gpu_tiled(img_in);
	gettimeofday(&end, NULL);
	//printf("contrast_enhancement_c_hsl_gpu_tiled() %f\n", ((end.tv_sec + end.tv_usec * 0.000001)  - (start.tv_sec + start.tv_usec * 0.000001)));

    write_ppm(img_obuf_hsl, "out_hsl_gpu.ppm");
    gettimeofday(&start, NULL);
	img_obuf_yuv2 = contrast_enhancement_c_yuv_gpu_tiled(img_in);
	gettimeofday(&end, NULL);
	//printf("contrast_enhancement_c_yuv_gpu_tiled() %f\n", ((end.tv_sec + end.tv_usec * 0.000001)  - (start.tv_sec + start.tv_usec * 0.000001)));
	
    write_ppm(img_obuf_yuv, "out_yuv_gpu.ppm"); 
    
	bool equal = 1;
	for( i = 0; i < img_in.w * img_in.h; ++i){
		if( img_obuf_hsl.img_r[i] != img_obuf_hsl2.img_r[i] ||  
			img_obuf_hsl.img_g[i] != img_obuf_hsl2.img_g[i] || 
			img_obuf_hsl.img_b[i] != img_obuf_hsl2.img_b[i]){
		   equal = 0;
		   }	
		if(img_obuf_yuv.img_r[i] != img_obuf_yuv2.img_r[i] || 
		   img_obuf_yuv.img_g[i] != img_obuf_yuv2.img_g[i] || 
		   img_obuf_yuv.img_b[i] != img_obuf_yuv2.img_b[i]){
		   equal = 0;
		   }	
		     
	}
	if(!equal){
		printf("\n not equal\n\n");
	}
	
	free_ppm(img_obuf_hsl);
    free_ppm(img_obuf_yuv);    
	free_ppm(img_obuf_hsl2);
    free_ppm(img_obuf_yuv2);   
}

void run_gpu_gray_test(PGM_IMG img_in){
	
	// Timing
	cudaEvent_t start, stop;
	float milliseconds = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	PGM_IMG a, b;
    printf("Starting GPU processing...\n");
	cudaEventRecord(start);
	a = contrast_enhancement_g_gpu(img_in);	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("gpu-naive gray test %f\n", milliseconds * 0.001);
	
	printf("Starting GPU-Tiled processing...\n");
	cudaEventRecord(start);
	b = contrast_enhancement_g_gpu_tiled(img_in);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("gpu-tiled gray test %f\n", milliseconds * 0.001);
	
	int i;
	int equal = 1;
	for( i = 0; i < img_in.w * img_in.h; ++i){
		if(a.img[i] != b.img[i]){
			equal = 0;
		}
	}
	if(!equal)
		printf("\nNOT EQUAL!\n\n");
}

void run_cpu_color_test(PPM_IMG img_in)
{
	struct timeval start, end;
    PPM_IMG img_obuf_hsl, img_obuf_yuv;
    
    printf("Starting CPU processing...\n");
	gettimeofday(&start, NULL);
    img_obuf_hsl = contrast_enhancement_c_hsl(img_in);
	gettimeofday(&end, NULL);
	printf("contrast_enhancement_c_hsl() %f\n", ((end.tv_sec + end.tv_usec * 0.000001)  - (start.tv_sec + start.tv_usec * 0.000001)));

    write_ppm(img_obuf_hsl, "out_hsl.ppm");
    gettimeofday(&start, NULL);
	img_obuf_yuv = contrast_enhancement_c_yuv(img_in);
	gettimeofday(&end, NULL);
	printf("contrast_enhancement_c_yuv() %f\n", ((end.tv_sec + end.tv_usec * 0.000001)  - (start.tv_sec + start.tv_usec * 0.000001)));

    write_ppm(img_obuf_yuv, "out_yuv.ppm"); 
    
	free_ppm(img_obuf_hsl);
    free_ppm(img_obuf_yuv);
}




void run_cpu_gray_test(PGM_IMG img_in)
{	
	// Timing
	cudaEvent_t start, stop;
	float milliseconds = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
    printf("Starting CPU processing...\n");
	cudaEventRecord(start);
    contrast_enhancement_g(img_in);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("cpu gray test %f\n", milliseconds * 0.001);
	
    //cutilCheckError(cutStopTimer(timer));
    //printf("Processing time: %f (ms)\n", cutGetTimerValue(timer));
    //cutilCheckError(cutDeleteTimer(timer));
    
    //write_pgm(img_obuf, "out.pgm");
    //free_pgm(img_obuf);
}



PPM_IMG read_ppm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    char *ibuf;
    PPM_IMG result;
    int v_max, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    /*Skip the magic number*/
    fscanf(in_file, "%s", sbuf);


    //result = malloc(sizeof(PPM_IMG));
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    ibuf         = (char *)malloc(3 * result.w * result.h * sizeof(char));

    
    fread(ibuf,sizeof(unsigned char), 3 * result.w*result.h, in_file);

    for(i = 0; i < result.w*result.h; i ++){
        result.img_r[i] = ibuf[3*i + 0];
        result.img_g[i] = ibuf[3*i + 1];
        result.img_b[i] = ibuf[3*i + 2];
    }
    
    fclose(in_file);
    free(ibuf);
    
    return result;
}

void write_ppm(PPM_IMG img, const char * path){
    FILE * out_file;
    int i;
    
    char * obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));

    for(i = 0; i < img.w*img.h; i ++){
        obuf[3*i + 0] = img.img_r[i];
        obuf[3*i + 1] = img.img_g[i];
        obuf[3*i + 2] = img.img_b[i];
    }
    out_file = fopen(path, "wb");
    fprintf(out_file, "P6\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(obuf,sizeof(unsigned char), 3*img.w*img.h, out_file);
    fclose(out_file);
    free(obuf);
}

void free_ppm(PPM_IMG img)
{
    free(img.img_r);
    free(img.img_g);
    free(img.img_b);
}

PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

        
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}

