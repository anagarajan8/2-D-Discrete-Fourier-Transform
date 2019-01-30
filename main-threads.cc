#include <iostream>
#include <complex.h>
#include <input_image.h>
#include <string>
#include <cmath>
#include <thread>

using namespace std;

const float PI = 3.14159265358979f;

/*
This function is used by each thread to compute the row transform for the rows it is responsible for. 
Complex* data - input
COmplex* fr - output
*/
void computeRow(Complex* data, Complex* fr, int w, int h, int index, int num_threads){
    for(int i=index; i<index+h/num_threads; i++){
        for(int j=0; j<w; j++){
            for(int k=0; k<w; k++){
                float angle = -2*PI*j*k/w;
                Complex c(cos(angle),sin(angle));
                fr[i*w+j] = fr[i*w+j] + c*data[i*w+k];              
            }
        }        
    }    
}

/*
This function is used by each thread to compute the column transform for the columns it is responsible for. 
Complex* fr - input
COmplex* sr - output
*/
void computeColumn(Complex* fr, Complex* sr, int w, int h, int index, int num_threads){
    for(int i=index; i<index+w/num_threads; i++){
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
This function is used by each thread to compute the row transform in the Inverse DFT for the rows it is responsible for. 
Complex* fr - input
COmplex* sr - output
*/
void computeRowIFT(Complex* fr, Complex* sr, int w, int h, int index, int num_threads){
    for(int i=index; i<index+h/num_threads; i++){
        for(int j=0; j<w; j++){
            sr[i*w+j].real = 0;
            sr[i*w+j].imag = 0;
            for(int k=0; k<w; k++){
                float angle = 2*PI*j*k/w;
                Complex c(cos(angle),sin(angle));
                sr[i*w+j] = sr[i*w+j] + c*fr[i*w+k];              
            }
            sr[i*w+j].real /= w;
            sr[i*w+j].imag /= w;
        }
    }
}

/*
This function is used by each thread to compute the column transform in the Inverse DFT for the columns it is responsible for. 
Complex* fr - input
COmplex* sr - output
*/
void computeColumnIFT(Complex* fr, Complex* sr, int w, int h, int index, int num_threads){
    for(int i=index; i<index+h/num_threads; i++){
        for(int j=0; j<h; j++){
            sr[j*h+i].real = 0;
            sr[j*h+i].imag = 0;
            for(int k=0; k<h; k++){
                float angle = 2*PI*j*k/h;
                Complex c(cos(angle),sin(angle));
                sr[j*h+i] = sr[j*h+i] + c*fr[k*h+i];                
            }
            sr[j*h+i].real /= h;
            sr[j*h+i].imag /= h;
        }
    }
}

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
    Complex* fr = new Complex[h*w];
    Complex* sr = new Complex[h*w];
    
    if(type.compare("forward") == 0){
        int num_threads = 8;
        if(w<num_threads){
            num_threads = w; 
        }
        thread thread_list[num_threads];
        for(int i=0; i<num_threads; i++){
            thread_list[i] = thread(computeRow, data, fr, w, h, i*w/num_threads, num_threads);
        }
        for(int i=0; i<num_threads; i++){
            thread_list[i].join();
        } 
        for(int i=0; i<num_threads; i++){
            thread_list[i] = thread(computeColumn, fr, sr, w, h, i*w/num_threads, num_threads);
        }
        for(int i=0; i<num_threads; i++){
            thread_list[i].join();
        } 
        img.save_image_data(output_file.c_str(), sr, w, h);        
    }
    else{
        int num_threads = 8;
        if(w<num_threads){
            num_threads = w; 
        }
        thread thread_list[num_threads];
        for(int i=0; i<num_threads; i++){
            thread_list[i] = thread(computeRowIFT, data, fr, w, h, i*w/num_threads, num_threads);
        }
        for(int i=0; i<num_threads; i++){
            thread_list[i].join();
        } 
        for(int i=0; i<num_threads; i++){
            thread_list[i] = thread(computeColumnIFT, fr, sr, w, h, i*w/num_threads, num_threads);
        }
        for(int i=0; i<num_threads; i++){
            thread_list[i].join();
        }
        img.save_image_data(output_file.c_str(), sr, w, h);
    }
    
    return 0;
}
