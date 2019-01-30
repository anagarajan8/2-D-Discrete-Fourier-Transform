#include <iostream>
#include "complex.cuh"
#include "input_image.h"
#include <string>
#include <cmath>
#include <thread>

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/*
This function is used for asserting the success of any kernel function in cuda
*/
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const float PI = 3.14159265358979f;

/*
This function is used to print the results of the DFT or Inverse DFT on the screen.
Complex* sr - input
*/
void printResults(Complex* sr, int w, int h){
    cout<<"Results = "<<endl;
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            cout<<sr[i*w+j].real<<","<<sr[i*w+j].imag<<" ; "; 
        }
        cout<<endl;
    }
    cout<<"------------------------------------------------"<<endl;
}

/*
This kernel function is used by each thread to compute the row-transform for the element in the matrix it responsible for. 
Complex* data - input
COmplex* out - output
*/
__global__ void computeRow(Complex* data, Complex* out, int w, int h){ 
    uint32_t index = threadIdx.x + blockIdx.x*blockDim.x;
    uint32_t start_ind = w*(index/w);
    uint32_t j = index%w;
    out[index].real = 0;
    out[index].imag = 0;
    for(int k=0; k<w; k++){
        float angle = -2*PI*j*k/w;
        Complex c(cos(angle),sin(angle));
        out[index] = out[index] + c*data[start_ind+k];              
    }
}

/*
This kernel function is used by each thread to compute the column-transform for the element in the matrix it responsible for. 
Complex* data - input
COmplex* out - output
*/	
__global__  void computeColumn(Complex* data, Complex* out, int w, int h){
    uint32_t index = threadIdx.x + blockIdx.x*blockDim.x;
    uint32_t start = index%w;
    uint32_t j  = index/w;
    out[index].real = 0;
    out[index].imag = 0;
    for(int k=0; k<h; k++){
        float angle = -2*PI*j*k/h;
        Complex c(cos(angle),sin(angle));
        out[index] = out[index] + c*data[k*w+start];                
    }
}

/*
This kernel function is used by each thread to compute the row-transform(Inverse DFT) for the element in the matrix it responsible for. 
Complex* data - input
COmplex* out - output
*/
__global__ void computeRowInv(Complex* data, Complex* out, int w, int h){
    uint32_t index = threadIdx.x + blockIdx.x*blockDim.x;
    uint32_t start_ind = w*(index / w);
    uint32_t j = index % w;
	out[index].real = 0;
	out[index].imag = 0;
    for(int k=0; k<w; k++){
        float angle = 2*PI*j*k/w;
        Complex c(cos(angle),sin(angle));
        out[index] = out[index] + c*data[start_ind+k];              
    }
	out[index].real /= w;
	out[index].imag /= w;
}

/*
This kernel function is used by each thread to compute the column-transform(Inverse DFT) for the element in the matrix it responsible for. 
Complex* data - input
COmplex* out - output
*/	
__global__  void computeColumnInv(Complex* data, Complex* out, int w, int h){
    uint32_t index = threadIdx.x+ blockIdx.x * blockDim.x;
    uint32_t start = index % w;
    uint32_t j = index / w; 
   	out[index].real = 0;
    out[index].imag = 0;
    for(int k=0; k<h; k++){
        float angle = 2*PI*j*k/h;
        Complex c(cos(angle),sin(angle));
        out[index] = out[index] + c*data[k*w+start];                
    }
    out[index].real /= w;
    out[index].imag /= w;
}
int main(int argc, char* argv[]){
    if(argc<4){
        cout<<"Insufficient number of arguments\n"; 
        return 0;
    }
    string type = argv[1];
    string input_file = argv[2];
    string output_file = argv[3];
    InputImage img(input_file.c_str());
    int w = img.get_width();
    int h = img.get_height();
    Complex* data = img.get_image_data();
    Complex* inp;
    Complex* out;
    Complex* result = new Complex[h*w];

    int numThreads = w > 1024? 1024:w;
    int numBlocks = w * h /(numThreads);
    gpuErrchk(cudaMalloc((void **)&inp, w*h*sizeof(Complex)));
    gpuErrchk(cudaMalloc((void **)&out, w*h*sizeof(Complex)));

    cudaMemcpy(inp, data, w*h*sizeof(Complex), cudaMemcpyHostToDevice);

    gpuErrchk(cudaDeviceSynchronize());	
    if(type.compare("forward") == 0){
        computeRow<<<numBlocks,numThreads>>>(inp, out,w,h);
        gpuErrchk(cudaDeviceSynchronize());
        computeColumn<<<numBlocks,numThreads>>>(out, inp, w,h);
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaMemcpy(result, inp, w*h*sizeof(Complex), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        img.save_image_data(output_file.c_str(), result, w, h);     
    }
    else{
        computeRowInv<<<numBlocks,numThreads>>>(inp, out,w,h);
        gpuErrchk(cudaDeviceSynchronize());
        computeColumnInv<<<numBlocks,numThreads>>>(out, inp, w,h);
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaMemcpy(result, inp, w*h*sizeof(Complex), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        img.save_image_data(output_file.c_str(), result, w, h);     
    }       

    return 0;
}
