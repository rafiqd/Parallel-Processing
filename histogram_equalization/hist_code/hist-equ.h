#include <cuda_runtime.h>
#ifndef HIST_EQU_COLOR_H
#define HIST_EQU_COLOR_H
#define BLOCKS 4
#define THREADS 1024

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    

typedef struct{
    int w;
    int h;
    unsigned char * img_r;
    unsigned char * img_g;
    unsigned char * img_b;
} PPM_IMG;

typedef struct{
    int w;
    int h;
    unsigned char * img_y;
    unsigned char * img_u;
    unsigned char * img_v;
} YUV_IMG;


typedef struct
{
    int width;
    int height;
    float * h;
    float * s;
    unsigned char * l;
} HSL_IMG;

    

PPM_IMG read_ppm(const char * path);
void write_ppm(PPM_IMG img, const char * path);
void free_ppm(PPM_IMG img);

PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);

HSL_IMG rgb2hsl(PPM_IMG img_in);
PPM_IMG hsl2rgb(HSL_IMG img_in);

YUV_IMG rgb2yuv(PPM_IMG img_in);
PPM_IMG yuv2rgb(YUV_IMG img_in);    

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin);

//Contrast enhancement for gray-scale images
PGM_IMG contrast_enhancement_g(PGM_IMG img_in);

//Contrast enhancement for color images
PPM_IMG contrast_enhancement_c_rgb(PPM_IMG img_in);
PPM_IMG contrast_enhancement_c_yuv(PPM_IMG img_in);
PPM_IMG contrast_enhancement_c_hsl(PPM_IMG img_in);


//Contrast enhancement for gray-scale images -- GPU
PGM_IMG contrast_enhancement_g_gpu(PGM_IMG img_in);
__global__ void histogram_gpu(int *hist_out, unsigned char *img_in, int *img_size, int *nbr_bin, int *lines);
__global__ void histogram_equalization_gpu(unsigned char *img_out, unsigned char *img_in, int *hist_in, int *img_size, int *min, int *d, int *cdf);

PPM_IMG contrast_enhancement_c_yuv_gpu(PPM_IMG img_in);
__global__ void rbg2yuv_gpu(unsigned char *img_in_r, unsigned char *img_in_g, unsigned char *img_in_b, unsigned char *img_out_y, unsigned char *img_out_u, unsigned char *img_out_v, int *lines, int *imgsize);
__global__ void yuv2rbg_gpu(unsigned char *img_out_r, unsigned char *img_out_g, unsigned char *img_out_b, unsigned char *img_in_y, unsigned char *img_in_u, unsigned char *img_in_v, int *lines, int *imgsize);

PPM_IMG contrast_enhancement_c_hsl_gpu(PPM_IMG img_in);
__global__ void rgb2hsl_gpu(unsigned char *img_in_r, unsigned char *img_in_g, unsigned char *img_in_b, float *img_out_h, float *img_out_s, unsigned char *img_out_l, int* lines, int *imgsize);
__global__ void hsl2rgb_gpu(unsigned char *img_out_r, unsigned char *img_out_g, unsigned char *img_out_b, float *img_in_h, float *img_in_s, unsigned char *img_in_l, int* lines, int *imgsize);
__device__ float Hue_2_RGB_gpu( float v1, float v2, float vH );

// Tiled versions
PGM_IMG contrast_enhancement_g_gpu_tiled(PGM_IMG img_in);
__global__ void histogram_gpu_tiled(int *hist_out, unsigned char *img_in, int *img_size, int *nbr_bin, int *lines);
__global__ void histogram_equalization_gpu_tiled( int *min, int *d, int *cdf, int* lut);
__global__ void histogram_equalization_gpu_tiled_p2(unsigned char *img_out, unsigned char *img_in, int *img_size, int* lut, int* lines);
PPM_IMG contrast_enhancement_c_yuv_gpu_tiled(PPM_IMG img_in);
PPM_IMG contrast_enhancement_c_hsl_gpu_tiled(PPM_IMG img_in);

PPM_IMG contrast_enhancement_c_rgb_gpu(PPM_IMG img_in);
PPM_IMG contrast_enhancement_c_rgb_gpu_tiled(PPM_IMG img_in);
#endif
