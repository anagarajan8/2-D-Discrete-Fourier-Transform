#include <iostream>
#include <complex.h>
#include <input_image.h>
#include <string>
#include <cmath>
#include "mpi.h"

using namespace std;

const float PI = 3.14159265358979f;

/*
This function is used to compute the DFT for the matrix.(Used for verifying output of MPI-FFT)
Complex* data - input
COmplex* sr - output
*/
void computeDFT(Complex* data, Complex* fr, Complex* sr, int w, int h){
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            for(int k=0; k<w; k++){
                float angle = -2*PI*j*k/w;
                Complex c(cos(angle),sin(angle));
                fr[i*w+j] = fr[i*w+j] + c*data[i*w+k];              
            }
        }
    }

    for(int i=0; i<w; i++){
        for(int j=0; j<h; j++){
            for(int k=0; k<h; k++){
                float angle = -2*PI*j*k/h;
                Complex c(cos(angle),sin(angle));
                sr[j*w+i] = sr[j*w+i] + c*fr[k*w+i];                
            }
        }
    }
}

/*
This function is used to compute the Inverse DFT for the matrix.(Used for verifying output of MPI-IDFT)
Complex* data - input
COmplex* sr - output
*/
void computeIDFT(Complex* data, Complex* fr, Complex* sr, int w, int h){
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            fr[i*w+j].real = 0;
            fr[i*w+j].imag = 0;
            for(int k=0; k<w; k++){
                float angle = 2*PI*j*k/w;
                Complex c(cos(angle),sin(angle));
                fr[i*w+j] = fr[i*w+j] + c*sr[i*w+k];              
            }
            fr[i*w+j].real /= w;
            fr[i*w+j].imag /= w;
        }
    }

    for(int i=0; i<w; i++){
        for(int j=0; j<h; j++){
            data[j*h+i].real = 0;
            data[j*h+i].imag = 0;
            for(int k=0; k<h; k++){
                float angle = 2*PI*j*k/h;
                Complex c(cos(angle),sin(angle));
                data[j*h+i] = data[j*h+i] + c*fr[k*h+i];                
            }
            data[j*h+i].real /= h;
            data[j*h+i].imag /= h;
        }
    }
}

/*
This function is used to compute the FFT for the matrix.
Complex* fr - float array which stores the block of real values followed by the imaginary values for a row.
int shard_size - number of rows each process in MPI is repsonsible for computing the transform.
Uses the Cooley Tukey Algorithm for computing the FFT.
*/
void computeFFTForMPI(float* fr, int w, int h, int shard_size){    
    if(w>=2){
        if(w>=3){
            float temp[2*w];
            int i = 0;
            int j = 0;
            while(i<w){
                temp[j] = fr[i];
                temp[j+w] = fr[i+shard_size*h];
                i+=2;
                j++;
            }
            i=1;
            while(i<w && j<w){
                temp[j] = fr[i];
                temp[j+w] = fr[i+shard_size*h];
                i+=2;
                j++;
            }
            for(int index=0; index<w; index++){
                fr[index] = temp[index];
                fr[index+shard_size*h] = temp[index+w];
            }
        }
        computeFFTForMPI(fr, w/2, h, shard_size);
        computeFFTForMPI(fr+w/2, w/2, h, shard_size);
        int i = 0;
        while(i<w/2){
            float angle = -2*PI*i/w;
            float c_real = cos(angle); 
            float c_imag = sin(angle);
            float fr_real = fr[i];
            float fr_imag = fr[i+shard_size*h];
            float fr_sym_real = fr[i+w/2];
            float fr_sym_imag = fr[i+w/2+shard_size*h];
            fr[i]                  = fr_real + c_real*fr_sym_real - c_imag*fr_sym_imag; //new_fr_real
            fr[i+shard_size*h]     = fr_imag + c_real*fr_sym_imag + c_imag*fr_sym_real; //new_fr_imag                         
            fr[i+w/2]              = fr_real - c_real*fr_sym_real + c_imag*fr_sym_imag; //new_fr_sym_real
            fr[i+w/2+shard_size*h] = fr_imag - c_real*fr_sym_imag - c_imag*fr_sym_real; //new_fr_sym_imag
            i++;
        }
    }
}

/*
This function is used to compute the Inverse DFT for the rows the process in MPI is responsible fors.
Complex* input - input float array which stores the block of real values for all rows in the shard followed by the imaginary values.
Complex* output - output float array which stores the Inverse DFT row transform. The block of real values for all rows in the shard are stored 
followed by the imaginary values.
int shard_size - number of rows each process in MPI is repsonsible for computing the transform.
*/
void computeIFTForMPI(float* input, float* output, int w, int shard_size){
    for(int j=0; j<w; j++){
        output[j] = 0;
        output[j+w*shard_size] = 0;
        for(int k=0; k<w; k++){
            float angle = 2*PI*j*k/w;
            float c_real = cos(angle); 
            float c_imag = sin(angle);
            output[j] = output[j] + c_real*input[k] - c_imag*input[k+shard_size*w];              
            output[j+shard_size*w] = output[j+shard_size*w] + c_real*input[k+shard_size*w] + c_imag*input[k];              
        }
        output[j] /= w;
        output[j+shard_size*w] /= w;
    }
}

/*
This function is used to transpose the float matrix storing complex numbers.
Complex* m - float array which stores the block of real values followed by the imaginary values.
*/
void transposeFloatMatrix(float* m, int w, int h){
    for(int i=0; i<h; i++){
        for(int j=0; j<i; j++){
            swap(m[i*w+j],m[j*w+i]);
            swap(m[i*w+j+w*h],m[j*w+i+w*h]);
        }
    }
}

/*
This function is used to print the results of the FFT or Inverse DFT on the screen.
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
This function is used to print the float matrix which stores the FFT or Inverse DFT on the screen.
float* ft - input - float array which stores the block of real values followed by the imaginary values.
*/
void printFT(float* ft, int w, int h){
    cout<<"------------------------------------------------\n";
    cout<<"ft = "<<endl;
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            cout<<ft[i*w+j]<<","<<ft[w*h+i*w+j]<<" ; ";
        }
        cout<<endl;
    }
    cout<<"------------------------------------------------\n";
}

int main(int argc, char* argv[]){
    if(argc<4){
        cout<<"Insufficient number of arguments\n"; 
        return 0;
    }
    int rank, num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    string type = argv[1];
    string input_file = argv[2];
    string output_file = argv[3];
    int w, h, num_processes_limit;
    int shard_size, extra, total_shard_size;
    Complex *data, *fr, *sr;

    if(type.compare("forward") == 0){
        if(rank == 0){
            InputImage img(input_file.c_str());
            w = img.get_width();
            h = img.get_height();
            data = img.get_image_data();
            sr = new Complex[h*w];
            num_processes_limit = min(w, num_processes);
            int init_info[1];
            init_info[0] = w;
            for(int i=1; i<num_processes; i++){
                MPI_Send(init_info, 1, MPI_INT, i, 0, MPI_COMM_WORLD); 
            }
            shard_size = w/num_processes_limit;
            extra = w%num_processes_limit;
            total_shard_size = shard_size;
            if(extra && rank<extra){
                total_shard_size++;
            }
            float output[total_shard_size*w*2];
            float ft[w*h*2];
            int offset[num_processes_limit];
            int process_shard_size[num_processes_limit];
            offset[0] = 0;
            process_shard_size[0] = total_shard_size;
            for(int i=1; i<num_processes_limit; i++){
                offset[i] = offset[i-1] + process_shard_size[i-1]*w;                            
                process_shard_size[i] = shard_size;
                if(extra && i<extra){
                    process_shard_size[i] += 1;
                }
            }
            for(int i=0; i<h; i++){
                for(int j=0; j<w; j++){
                    if(i<total_shard_size){
                        output[i*w+j] = data[i*w+j].real;
                        output[i*w+j+total_shard_size*w] = data[i*w+j].imag;
                    }
                    ft[i*w+j] = data[i*w+j].real;
                    ft[w*h+i*w+j] = data[i*w+j].imag;
                }
            }
            for(int i=1; i<num_processes_limit; i++){
                MPI_Send(ft+offset[i], process_shard_size[i]*w, MPI_FLOAT, i, 0, MPI_COMM_WORLD); 
                MPI_Send(ft+offset[i]+w*h, process_shard_size[i]*w, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
            for(int i=0; i<total_shard_size; i++){
                computeFFTForMPI(output+i*w, w, h, total_shard_size);
            }
            for(int i=1; i<num_processes_limit; i++){
                MPI_Recv(ft+offset[i], process_shard_size[i]*w, MPI_FLOAT, i, 0, MPI_COMM_WORLD,&status);             
                MPI_Recv(ft+offset[i]+w*h, process_shard_size[i]*w, MPI_FLOAT, i, 0, MPI_COMM_WORLD,&status);  
            }
            for(int i=0; i<total_shard_size; i++){
                for(int j=0; j<w; j++){
                    ft[i*w+j] = output[i*w+j];
                    ft[i*w+j+w*h] = output[i*w+j+total_shard_size*w];
                }
            }
            transposeFloatMatrix(ft,w,h);
            for(int i=1; i<num_processes_limit; i++){
                MPI_Send(ft+offset[i], process_shard_size[i]*w, MPI_FLOAT, i, 0, MPI_COMM_WORLD); 
                MPI_Send(ft+offset[i]+w*h, process_shard_size[i]*w, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
            for(int i=0; i<total_shard_size; i++){
                for(int j=0; j<w; j++){
                    output[i*w+j] = ft[i*w+j];
                    output[i*w+j+total_shard_size*w] = ft[i*w+j+w*h];
                }
            }
            for(int i=0; i<total_shard_size; i++){
                computeFFTForMPI(output+i*w, w, h, total_shard_size);
            }
            for(int i=1; i<num_processes_limit; i++){
                MPI_Recv(ft+offset[i], process_shard_size[i]*w, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);             
                MPI_Recv(ft+offset[i]+w*h, process_shard_size[i]*w, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);  
            }
            for(int i=0; i<total_shard_size; i++){
                for(int j=0; j<w; j++){
                    ft[i*w+j] = output[i*w+j];
                    ft[i*w+j+w*h] = output[i*w+j+total_shard_size*w];
                }
            }
            transposeFloatMatrix(ft, w, h);
            for(int i=0; i<h; i++){
                for(int j=0; j<w; j++){
                    sr[i*w+j].real = ft[i*w+j];
                    sr[i*w+j].imag = ft[w*h+i*w+j];
                }
            }
            img.save_image_data(output_file.c_str(), sr, w, h);
        }
        else{
            int init_info[1];
            MPI_Recv(init_info, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            w = init_info[0];
            h = w;
            num_processes_limit = min(w, num_processes);
            if(rank<num_processes_limit)
            {
                int shard_size = w/num_processes_limit;
                int extra = w%num_processes_limit;
                int total_shard_size = shard_size;
                if(extra && rank<extra){
                    total_shard_size++;
                }
                float output[total_shard_size*w*2];
                MPI_Recv(output, total_shard_size*w, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);             
                MPI_Recv(output+total_shard_size*w, total_shard_size*w, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status); 
                for(int i=0; i<total_shard_size; i++){
                    computeFFTForMPI(output+i*w, w, h, total_shard_size);
                }
                MPI_Send(output, total_shard_size*w, MPI_FLOAT, 0, 0, MPI_COMM_WORLD); 
                MPI_Send(output+total_shard_size*w, total_shard_size*w, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);                            
                MPI_Recv(output, total_shard_size*w, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);             
                MPI_Recv(output+total_shard_size*w, total_shard_size*w, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status); 
                for(int i=0; i<total_shard_size; i++){
                    computeFFTForMPI(output+i*w, w, h, total_shard_size);
                }
                MPI_Send(output, total_shard_size*w, MPI_FLOAT, 0, 0, MPI_COMM_WORLD); 
                MPI_Send(output+total_shard_size*w, total_shard_size*w, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);                
            }
        }
    }
    else{
        if(rank == 0){
            InputImage img(input_file.c_str());
            w = img.get_width();
            h = img.get_height();
            data = img.get_image_data();
            sr = new Complex[h*w];
            num_processes_limit = min(w, num_processes);
            int init_info[1];
            init_info[0] = w;
            for(int i=1; i<num_processes; i++){
                MPI_Send(init_info, 1, MPI_INT, i, 0, MPI_COMM_WORLD); 
            }
            shard_size = w/num_processes_limit;
            extra = w%num_processes_limit;
            total_shard_size = shard_size;
            if(extra && rank<extra){
                total_shard_size++;
            }
            float input[total_shard_size*w*2];
            float output[total_shard_size*w*2];
            float ft[w*h*2];
            int offset[num_processes_limit];
            int process_shard_size[num_processes_limit];
            offset[0] = 0;
            process_shard_size[0] = total_shard_size;
            for(int i=1; i<num_processes_limit; i++){
                offset[i] = offset[i-1] + process_shard_size[i-1]*w;                            
                process_shard_size[i] = shard_size;
                if(extra && i<extra){
                    process_shard_size[i] += 1;
                }
            }
            for(int i=0; i<h; i++){
                for(int j=0; j<w; j++){
                    if(i<total_shard_size){
                        input[i*w+j] = data[i*w+j].real;
                        input[i*w+j+total_shard_size*w] = data[i*w+j].imag;
                    }
                    ft[i*w+j] = data[i*w+j].real;
                    ft[w*h+i*w+j] = data[i*w+j].imag;
                }
            }
            for(int i=1; i<num_processes_limit; i++){
                MPI_Send(ft+offset[i], process_shard_size[i]*w, MPI_FLOAT, i, 0, MPI_COMM_WORLD); 
                MPI_Send(ft+offset[i]+w*h, process_shard_size[i]*w, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
            for(int i=0; i<total_shard_size; i++){
                computeIFTForMPI(input+i*w, output+i*w, w, total_shard_size);
            }
            for(int i=1; i<num_processes_limit; i++){
                MPI_Recv(ft+offset[i], process_shard_size[i]*w, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);             
                MPI_Recv(ft+offset[i]+w*h, process_shard_size[i]*w, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);  
            }
            for(int i=0; i<total_shard_size; i++){
                for(int j=0; j<w; j++){
                    ft[i*w+j] = output[i*w+j];
                    ft[i*w+j+w*h] = output[i*w+j+total_shard_size*w];
                }
            }
            transposeFloatMatrix(ft, w, h);
            for(int i=1; i<num_processes_limit; i++){
                MPI_Send(ft+offset[i], process_shard_size[i]*w, MPI_FLOAT, i, 0, MPI_COMM_WORLD); 
                MPI_Send(ft+offset[i]+w*h, process_shard_size[i]*w, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
            for(int i=0; i<total_shard_size; i++){
                for(int j=0; j<w; j++){
                    input[i*w+j] = ft[i*w+j];
                    input[i*w+j+total_shard_size*w] = ft[i*w+j+w*h];
                }
            }
            for(int i=0; i<total_shard_size; i++){
                computeIFTForMPI(input+i*w, output+i*w, w, total_shard_size);
            }
            for(int i=1; i<num_processes_limit; i++){
                MPI_Recv(ft+offset[i], process_shard_size[i]*w, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);             
                MPI_Recv(ft+offset[i]+w*h, process_shard_size[i]*w, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);  
            }
            for(int i=0; i<total_shard_size; i++){
                for(int j=0; j<w; j++){
                    ft[i*w+j] = output[i*w+j];
                    ft[i*w+j+w*h] = output[i*w+j+total_shard_size*w];
                }
            }
            transposeFloatMatrix(ft, w, h);
            for(int i=0; i<h; i++){
                for(int j=0; j<w; j++){
                    sr[i*w+j].real = ft[i*w+j];
                    sr[i*w+j].imag = ft[w*h+i*w+j];
                }
            }
            img.save_image_data(output_file.c_str(), sr, w, h);
        }
        else{
            int init_info[1];
            MPI_Recv(init_info, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            w = init_info[0];
            h = w;
            num_processes_limit = min(w, num_processes);
            if(rank<num_processes_limit)
            {
                int shard_size = w/num_processes_limit;
                int extra = w%num_processes_limit;
                int total_shard_size = shard_size;
                if(extra && rank<extra){
                    total_shard_size++;
                }
                float input[total_shard_size*w*2];
                float output[total_shard_size*w*2];
                MPI_Recv(input, total_shard_size*w, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,&status);             
                MPI_Recv(input+total_shard_size*w, total_shard_size*w, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status); 
                for(int i=0; i<total_shard_size; i++){
                    computeIFTForMPI(input+i*w, output+i*w, w, total_shard_size);
                }
                MPI_Send(output, total_shard_size*w, MPI_FLOAT, 0, 0, MPI_COMM_WORLD); 
                MPI_Send(output+total_shard_size*w, total_shard_size*w, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);                            
                MPI_Recv(input, total_shard_size*w, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,&status);             
                MPI_Recv(input+total_shard_size*w, total_shard_size*w, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,&status);             
                for(int i=0; i<total_shard_size; i++){
                    computeIFTForMPI(input+i*w, output+i*w, w, total_shard_size);
                }
                MPI_Send(output, total_shard_size*w, MPI_FLOAT, 0, 0, MPI_COMM_WORLD); 
                MPI_Send(output+total_shard_size*w, total_shard_size*w, MPI_FLOAT, 0, 0, MPI_COMM_WORLD); 
            }
        }
    }

    MPI_Finalize();
    return 0;
}
