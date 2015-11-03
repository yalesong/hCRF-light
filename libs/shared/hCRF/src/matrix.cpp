/*
hCRF-light Library 2.0 (full version http://hcrf.sf.net)
Copyright (C) 2012 Yale Song (yalesong@mit.edu)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "matrix.h"
#include <iostream>
#include <math.h>
// Needed by GCC 4.3
#include <climits>

#ifdef _WIN32
#include "msinttypes/stdint.h"
#else
#include <stdint.h>
#endif


template <class elType>
void Matrix<elType>::add(const Matrix &m)
{
#ifdef _DEBUG
	if(!sameDim(m)) {
		//error
		throw std::invalid_argument("Matrix dimension must agree for addition");
	}
#endif

	elType* ptrSource = m.pData;
	int sizeMatrix = height * width;
	for(elType* ptrData = pData; ptrData < pData+sizeMatrix; ptrData++, ptrSource++) 
	{
		*(ptrData) += *(ptrSource);
	}
}


template <class elType>
void Matrix<elType>::subtract(const Matrix &m)
{
#ifdef _DEBUG
	if(!sameDim(m)) {
		//error
		throw std::invalid_argument("Matrix dimension must agree for addition");
	}
#endif

	elType* ptrSource = m.pData;
	int sizeMatrix = height * width;
	for(elType* ptrData = pData; ptrData < pData+sizeMatrix; ptrData++, ptrSource++) 
	{
		*(ptrData) -= *(ptrSource);
	}
}

template <class elType>
void Matrix<elType>::create(int w, int h, elType value)
{
	if (w==0 || h==0)
	{
		width = w;
		height = h;
		freeMemory(); //pData = 0;
						// Modified by Congkai, only pData may cause memory leak;
		return;
	}
	if((w != width || h != height || pData == NULL))
	{
		freeMemory();
		pData = new elType [w*h];
		width = w;
		height = h;
	}
	set(value);
}

template <class elType>
void Matrix<elType>::freeMemory()
{
	if (pData) {
		delete[] pData;
	}
	pData = 0;
}

// this = this * value;
template <class elType>
void Matrix<elType>::multiply(elType value)
{
	int sizeMatrix = height * width;
	for(elType* ptrData = pData; ptrData < pData+sizeMatrix; ptrData++)
		*(ptrData) *= value;
}

// this = this .* this;
template <class elType>
void Matrix<elType>::eltMpy(const Matrix<elType> &m)
{
	elType* ptrData = pData;
	elType* ptrSource = m.pData;
	int sizeMatrix = height * width;

	for(int i = 0; i < sizeMatrix; i++ )
	{
		*(ptrData) *= *(ptrSource);
		ptrData++;
		ptrSource++;
	}
}


template <class elType>
void Matrix<elType>::resize(int w, int h, elType value)
{
	if (w==width && h==height)
		//Nothing to do
		return;
	if(w<=0 || h<=0){
		throw std::invalid_argument("Size must be positive");
	}

	// determine overlap
    int min_h,min_w;
	min_h = h;
	min_w = w;
	if(min_h>height) {
		min_h = height;
	}
	if(min_w>width) {
		min_w = width;
	}

	// create matrix of size (w,h) and copy overlap region
	// create matrix of size (w,h) and copy overlap region
	Matrix<elType> tmpMatrix(w,h,value);
	for(int col=0;col<min_w;col++) 
		memcpy(tmpMatrix.get() + col*h, pData + col*height, min_h*sizeof(elType));

	// replace old matrix with new one
	set(tmpMatrix);
}

template <class elType>
void Matrix<elType>::set(elType value)
{
#ifdef _DEBUG
	if(height==0 || width==0)
		throw std::invalid_argument("Impossible to set value for ghost matrix");
#endif

	elType* tmpData = pData;
	int sizeMatrix = width * height;

	for(int i = 0; i < sizeMatrix; i++) 
		*(tmpData++) = value;

}




template void Matrix<unsigned char>::add(const Matrix<unsigned char> &m);
template void Matrix<uint64_t>::add(const Matrix<uint64_t> &m);
template void Matrix<unsigned int>::add(const Matrix<unsigned int> &m);
template void Matrix<int>::add(const Matrix<int> &m);
template void Matrix<float>::add(const Matrix<float> &m);

template void Matrix<unsigned char>::subtract(const Matrix<unsigned char> &m);
template void Matrix<uint64_t>::subtract(const Matrix<uint64_t> &m);
template void Matrix<unsigned int>::subtract(const Matrix<unsigned int> &m);
template void Matrix<int>::subtract(const Matrix<int> &m);
template void Matrix<float>::subtract(const Matrix<float> &m);

template void Matrix<unsigned char>::create(int w, int h, unsigned char value);
template void Matrix<uint64_t>::create(int w, int h, uint64_t value);
template void Matrix<unsigned int>::create(int w, int h, unsigned int value);
template void Matrix<int>::create(int w, int h, int value);
template void Matrix<float>::create(int w, int h, float value);

template void Matrix<unsigned char>::freeMemory();
template void Matrix<uint64_t>::freeMemory();
template void Matrix<unsigned int>::freeMemory();
template void Matrix<int>::freeMemory();
template void Matrix<float>::freeMemory();

template void Matrix<unsigned char>::eltMpy(const Matrix<unsigned char> &m);
template void Matrix<uint64_t>::eltMpy(const Matrix<uint64_t> &m);
template void Matrix<unsigned int>::eltMpy(const Matrix<unsigned int> &m);
template void Matrix<int>::eltMpy(const Matrix<int> &m);
template void Matrix<float>::eltMpy(const Matrix<float> &m);

template void Matrix<unsigned char>::multiply(unsigned char value);
template void Matrix<uint64_t>::multiply(uint64_t value);
template void Matrix<unsigned int>::multiply(unsigned int value);
template void Matrix<int>::multiply(int value);
template void Matrix<float>::multiply(float value); 

template void Matrix<unsigned char>::resize(int w, int h, unsigned char value);
template void Matrix<uint64_t>::resize(int w, int h, uint64_t value);
template void Matrix<unsigned int>::resize(int w, int h, unsigned int value);
template void Matrix<int>::resize(int w, int h, int value);
template void Matrix<float>::resize(int w, int h, float value);

template void Matrix<unsigned char>::set(unsigned char value);
template void Matrix<uint64_t>::set(uint64_t value);
template void Matrix<unsigned int>::set(unsigned int value);
template void Matrix<int>::set(int value);
template void Matrix<float>::set(float value);

#ifndef __SSE2__
template void Matrix<double>::set(double value);
template void Matrix<double>::resize(int w, int h, double value);
template void Matrix<double>::multiply(double value);
template void Matrix<double>::eltMpy(const Matrix<double> &m);
template void Matrix<double>::freeMemory();
template void Matrix<double>::create(int w, int h, double value);
template void Matrix<double>::add(const Matrix<double> &m);
template void Matrix<double>::subtract(const Matrix<double> &m);
#endif
