#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>

#include "PluginUtils.h"

using namespace std;


// simple implementation of adder filter
template <typename Ftype>
__global__ void adderFilter(int in_c, int in_h, int in_w, int k, int stride, int padding,
                            int out_h, int out_w, const Ftype* input, Ftype* output, const Ftype* weights)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid = tid_y*out_w + tid_x;

    int filterIdx = blockIdx.x;

    int out_idx = out_h * out_w * filterIdx + tid;
    output[out_idx] = 0;

    for(int a=0; a<in_c; a++)
    {
        for(int i=0; i<k; i++)
        {
            for(int j=0; j<k; j++)
            {
                int val;
                int input_pos_y = tid_y*stride + i - k/2;
                int input_pos_x = tid_x*stride + j - k/2;
                int input_idx = a*(in_h*in_w) + input_pos_y*in_w + input_pos_x;

                if(input_pos_y<0 || input_pos_y>in_h-1 || input_pos_x<0 || input_pos_x>in_w-1)
                {
                    val = 0.0;
                }
                else
                {
                    val = input[input_idx];
                }

                int weight_idx = filterIdx*in_c*k*k + a*k*k + i*k+ j;
                output[out_idx] += fabs(val - weights[weight_idx]);
            }
        }
    }
}

template <typename Dtype>
void forwardGpu(int n_filters,int in_c, int in_h, int in_w, int k, int stride, int pad,
                int out_h, int out_w, const Dtype* input, Dtype* output, const Dtype* weights)
{
    dim3 blkDim(out_w,out_h, 1);
    dim3 gridDim(n_filters,1,1);

    adderFilter<<<n_filters, blkDim>>>(in_c, in_h, in_w, k, stride, pad, out_h, out_w, input, output, weights);

    CUDA_CHECK(cudaDeviceSynchronize());
}


void printMatrix(double * in, int z, int y, int x)
{
    cout << "[";
    for(int b=0; b < z; b++)
    {
        cout << "[";
        for(int a=0; a<y; a++)
        {
            cout << "[";
            for(int c=0; c<x; c++)
            {
                cout << in[c + a*x + b*x*y] << ",";
            }
            cout << "]," << endl;
        }
        cout << "]," << endl;
    }
    cout << "]" << endl;
}

int main()
{
    int in_h = 5;
    int in_w = 5;
    int in_c = 5;

    printf("in_c, in_h, in_w: %d, %d, %d\n", in_c, in_h, in_w);

    int n_fil = 5;
    int stride = 1;
    int pad = 1;
    int k=3;

    int out_h = (in_h + 2*pad - k) / stride + 1;
    int out_w = (in_w + 2*pad - k) / stride + 1;
    int out_c = n_fil;

    printf("out_c, out_h, out_w: %d, %d, %d\n", out_c, out_h, out_w);

    double * in = nullptr;
    double * out = nullptr;
    double * fil = nullptr;

    CUDA_CHECK(cudaMallocManaged(&in, in_c*in_h*in_w*sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&out, n_fil*out_h*out_w*sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&fil, n_fil*in_c*k*k*sizeof(double)));

    for(int i=0; i<in_c*in_h*in_w; i++)
    {
        in[i] = 1;
    }

    for(int i=0; i<n_fil*in_c*k*k; i++)
    {
        fil[i] = 1;
    }

    cout << "######Input Feature Map, Size=(5,5,5)######" << endl;
    printMatrix(in, in_c, in_h, in_w);
    cout << endl;


    cout << "######Weights of One Adder Filter, Size=(5,3,3)######" << endl;
    printMatrix(fil, in_c, k, k);
    cout << endl;

    forwardGpu(n_fil, in_c, in_h, in_w, k, stride, pad, out_h, out_w, in, out, fil);

    cout << "######Padding=1, Stride=1#####" << endl;
    cout << "######Output Feature Map######" << endl;
    printMatrix(out, n_fil, out_h, out_w);
    cout << endl;

    cout << "Successfully tested the AdderFilter cuda kernel" << endl;

    return 0;

}
