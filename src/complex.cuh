//
// Created by brian on 11/20/18.
//

#pragma once

#include <iostream>

class Complex {
public:
    __host__ __device__ Complex();
    __host__ __device__ Complex(float r, float i);
    __host__ __device__ Complex(float r);
    __host__ __device__ Complex operator+(const Complex& b) const;
    __host__ __device__ Complex operator-(const Complex& b) const;
    __host__ __device__ Complex operator*(const Complex& b) const;

    Complex mag() const;
    Complex angle() const;
    Complex conj() const;

    float real;
    float imag;
};

std::ostream& operator<<(std::ostream& os, const Complex& rhs);

