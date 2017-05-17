#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#define VALUE_MAX 10000000
#define MAX_THREAD 1024

struct match{
	int bestRow;
	int bestCol;
	int bestSAD;
}position;

__global__ void cudaTempMatch(unsigned char *sourcePixel, unsigned char *patternPixel, int *result,
							  int result_w, int src_w, int pat_w)
{
	//int blockCol = blockIdx.x;
	//int blockRow = blockIdx.y;
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int tidSource = 512*(threadIdx.y + blockIdx.y) + threadIdx.x + blockIdx.x;
	int tidPattern = 48*threadIdx.y + threadIdx.x;
	int SAD = 0;
	//unsigned char subSrcElement;
	//unsigned char patElement;
	//unsigned char *subSource = sourcePixel + src_w*blockRow + blockCol;
	__shared__ unsigned char patternCache[48*48];
	__shared__ unsigned char subSrcElement[48*48];

	for (int i = 0; i < 2; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			//subSrcElement = *(sourcePixel + i*512*24 + j*24 + tidSource);
			subSrcElement[24*48*i + 24*j + 48*tidy + tidx] = *(sourcePixel + i*512*24 + j*24 + tidSource);
			patternCache[24*48*i + 24*j + 48*tidy + tidx] = *(patternPixel + i*48*24 + j*24 + tidPattern);
			SAD += abs(subSrcElement[24*48*i + 24*j + 48*tidy + tidx] - 
					   patternCache[24*48*i + 24*j + 48*tidy + tidx]);			
		}
	}
	//__syncthreads();

	*(result + blockIdx.y*result_w + blockIdx.x) = SAD;
}

int main( int argc, char** argv )
{
	IplImage* sourceImg; 
	IplImage* patternImg; 
	CvPoint pt1, pt2;
	int minSAD = VALUE_MAX;
	int x, y;
	struct timespec t_start, t_end;
	double elapsedTime;
	int result_height;
	int result_width;
	unsigned char *sourcePixelData, *patternPixelData;
	unsigned char *source_d, *pattern_d;
	int *result_d;
	int *host_result;

	if( argc != 3 )
	{
		printf("Using command: %s source_image search_image\n",argv[0]);
		exit(1);
	}

	// Load source image
	if((sourceImg = cvLoadImage( argv[1], 0)) == NULL)
	{
		printf("%s cannot be openned\n",argv[1]);
		exit(1);
	}

	const int sizeSource = sourceImg->imageSize * sizeof(unsigned char);
	printf("height of sourceImg:%d\n",sourceImg->height);
	printf("width of sourceImg:%d\n",sourceImg->width);
	printf("size of sourceImg:%d\n",sourceImg->imageSize); 

	// Load pattern image
	if((patternImg = cvLoadImage( argv[2], 0)) == NULL)
	{
		printf("%s cannot be openned\n",argv[2]);
		exit(1);
	}

	const int sizePattern = patternImg->imageSize * sizeof(unsigned char);
	printf("height of sourceImg:%d\n",patternImg->height);
	printf("width of sourceImg:%d\n",patternImg->width);
	printf("size of sourceImg:%d\n",patternImg->imageSize);

	// Define result size
	result_height = sourceImg->height - patternImg->height + 1;
	result_width = sourceImg->width - patternImg->width + 1;
	const int sizeResult = result_height * result_width * sizeof(int);
	host_result=(int*)malloc(sizeResult);

	// Record Image pixel datas in CPU
	sourcePixelData = (unsigned char*)malloc(sizeSource);
	patternPixelData = (unsigned char*)malloc(sizePattern);

	for (int i = 0; i < sourceImg->height; ++i)
	{
		for (int j = 0; j < sourceImg->width; ++j)
		{
			sourcePixelData[i*sourceImg->widthStep + j] = sourceImg->imageData[i*sourceImg->widthStep + j];
		}
	}

	for (int i = 0; i < patternImg->height; ++i)
	{
		for (int j = 0; j < patternImg->width; ++j)
		{
			patternPixelData[i*patternImg->widthStep + j] = patternImg->imageData[i*patternImg->widthStep + j];
		}
	}

	// start time
	clock_gettime( CLOCK_REALTIME, &t_start);

	// allocate device memory
	cudaMalloc((void**)&source_d, sizeSource);
	cudaMalloc((void**)&pattern_d, sizePattern);
	cudaMalloc((void**)&result_d, sizeResult);

	// copy datas to device from host
	cudaMemcpy(source_d, sourcePixelData, sizeSource,
							cudaMemcpyHostToDevice);
	cudaMemcpy(pattern_d, patternPixelData, sizePattern,
							cudaMemcpyHostToDevice);

	// kernel function
	dim3 blocksPerGrid(result_width, result_width);
	dim3 threadsPerBlock(24, 24);
	cudaTempMatch<<<blocksPerGrid, threadsPerBlock>>>
	(source_d, pattern_d, result_d, result_width, sourceImg->width, patternImg->width);

	// copy results to host from device
	cudaMemcpy(host_result, result_d, sizeResult,
							cudaMemcpyDeviceToHost); 

	// stop time
	clock_gettime( CLOCK_REALTIME, &t_end);
	// compute and print the elapsed time in millisec
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("%lf ms\n", elapsedTime);

	/*
	int count = 0;
	for (int i = 0; i < sizeResult/sizeof(int); ++i)
	{
		printf("%d ", host_result[i]);
		count++;
		if(count%10 == 0)
		{
			printf("\n");
		}
	}
	*/

	// find min SAD's sub source matrix place
	for( y=0; y<result_height; y++ ) 
	{
		for( x=0; x<result_width; x++ ) 
		{
			if ( minSAD > host_result[y * result_width + x] )
			{
				minSAD = host_result[y * result_width + x];

				// give me VALUE_MAX
				position.bestRow = y;
				position.bestCol = x;
				position.bestSAD = host_result[y * result_width + x];
			}
		}
	}

	printf("minSAD is %d\n", minSAD);

	//setup the two points for the best match
	pt1.x = position.bestCol;
	pt1.y = position.bestRow;
	pt2.x = pt1.x + patternImg->width;
	pt2.y = pt1.y + patternImg->height;

	printf("the point is (%d, %d) and (%d, %d)\n", pt1.x, pt1.y, pt2.x, pt2.y);

	// Draw the rectangle in the source image
	cvRectangle( sourceImg, pt1, pt2, CV_RGB(0,255,0), 3, 8, 0 );
	cvNamedWindow( "sourceImage", 1 );
	cvShowImage( "sourceImage", sourceImg );
	cvNamedWindow( "patternImage", 1 );
	cvShowImage( "patternImage", patternImg );
	cvWaitKey(0); 

	cvDestroyWindow( "sourceImage" );
	cvReleaseImage( &sourceImg );
	cvDestroyWindow( "patternImage" );
	cvReleaseImage( &patternImg );

	return 0;
}

