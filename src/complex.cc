//
// Created by brian on 11/20/18.
//

#include "complex.h"

#include <cmath>

const float PI = 3.14159265358979f;

Complex::Complex() : real(0.0f), imag(0.0f) {}

Complex::Complex(float r) : real(r), imag(0.0f) {}

Complex::Complex(float r, float i) : real(r), imag(i) {}

Complex Complex::operator+(const Complex &b) const {
    Complex c;
    c.real = this->real+b.real;
    c.imag = this->imag+b.imag;
    return c;
}

Complex Complex::operator-(const Complex &b) const {
    Complex c;
    c.real = this->real-b.real;
    c.imag = this->imag-b.imag;
    return c;
}

Complex Complex::operator*(const Complex &b) const {
    Complex c;
    c.real = this->real*b.real-this->imag*b.imag;
    c.imag = this->real*b.imag+this->imag*b.real;
    return c;
}

Complex Complex::mag() const {
    float x = this->real;
    float y = this->imag;
    float m = sqrt(x*x+y*y);
    return m;
}

Complex Complex::angle() const {
    float a = atanl(this->imag/this->real);
    return a;
}

Complex Complex::conj() const {
    Complex c;
    c.real = this->real;
    c.imag = -this->imag;
    return c;
}

std::ostream& operator<< (std::ostream& os, const Complex& rhs) {
    Complex c(rhs);
    if(fabsf(rhs.imag) < 1e-10) c.imag = 0.0f;
    if(fabsf(rhs.real) < 1e-10) c.real = 0.0f;

    if(c.imag == 0) {
        os << c.real;
    }
    else {
        os << "(" << c.real << "," << c.imag << ")";
    }
    return os;
}