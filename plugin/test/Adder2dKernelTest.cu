#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>

#include "PluginUtils.h"

using namespace std;


template <typename Ftype>
__global__ void adderFilter(int filterIdx, int in_c, int in_h, int in_w, int k, int stride, int padding,
                            int out_h, int out_w, const Ftype* input, Ftype* output, const Ftype* weights)
{
    int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
    int tid_y = threadIdx.y + blockDim.y*blockIdx.y;
    int tid = tid_y*k + tid_x;

    int out_idx = out_h * out_w * filterIdx + tid;
    output[out_idx] = 0;

    int tot_outputs = in_c * in_h * in_w;
    for(int a=0; a<in_c; a++)
    {
        for(int i=0; i<k; i++)
        {
            for(int j=0; j<k; j++)
            {
                int val = 0;
                int input_pos_y = tid_y*stride + i - k/2;
                int input_pos_x = tid_x*stride + j - k/2;
                int input_idx = a*(in_h*in_w) + input_pos_y*in_w + input_pos_x;

                if(input_pos_y>-1 && input_pos_y<in_h && input_pos_x>-1 && input_pos_x<in_w)
                    val = input[input_idx];

//                if(input_idx > -1 && input_idx < tot_outputs)
//                {
//                    val = input[input_idx];
//                }
//                printf("tid_x:%d, tid_y:%d, tid:%d, out_idx:%d, input_pos_y:%d, input_pos_x:%d, input_idx:%d, val:%d\n",
//                        tid_x, tid_y, tid, out_idx, input_pos_y, input_pos_x, input_idx, val);
                int weight_idx = filterIdx*k*k + i*k+j;
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

    for(int i=0; i<n_filters; i++)
    {
        adderFilter<<<gridDim, blkDim>>>(i, in_c, in_h, in_w, k, stride, pad, out_h, out_w, input, output, weights);
    }

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
    CUDA_CHECK(cudaMallocManaged(&fil, n_fil*k*k*sizeof(double)));

    for(int i=0; i<in_c*in_h*in_w; i++)
    {
        in[i] = 1;
    }

    for(int i=0; i<in_c*k*k; i++)
    {
        fil[i] = 2;
    }

    printMatrix(in, in_c, in_h, in_w);
    printMatrix(fil, n_fil, k, k);

    forwardGpu(n_fil, in_c, in_h, in_w, k, stride, pad, out_h, out_w, in, out, fil);

    printMatrix(out, n_fil, out_h, out_w);

    return 0;

}
