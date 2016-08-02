/*
 hCRF-light Library 3.0 (full version http://hcrf.sf.net)
 Copyright (C) Yale Song (yalesong@mit.edu)
 
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


#ifdef __SSE2__

#include <iostream>
#include <math.h>

// Needed by GCC 4.3
#include <climits>
#include <stdlib.h>
#include <memory.h>

#ifdef __APPLE__
#include <malloc/malloc.h>
#else
#include <malloc.h>
#endif

#if     1400 <= _MSC_VER
#include <intrin.h>
#endif/*1400 <= _MSC_VER*/

#ifdef __GNUC__
#include <xmmintrin.h>
#endif

#include "hcrf/matrix.h"

namespace hCRF
{
  
  static int roundOut(int n)
  {
    n += 7;
    n /= 8;
    n *= 8;
    return n;
  }
  
  inline static void vecFree(void *memblock)
  {
#ifdef	_MSC_VER
    _aligned_free(memblock);
#else
    free(memblock);
#endif
  }
  
  template <>
  void Matrix<double>::freeMemory()
  {
    if (pData)
    {
#ifdef _MSC_VER
      _aligned_free(pData);
#else
      free(pData);
#endif
    }
    pData = 0;
  }
  
  template <> void Matrix<double>::set(double value)
  {
#ifdef _DEBUG
    if(height==0 || width==0)
      throw std::invalid_argument("Impossible to set value for ghost matrix");
#endif
    double* x = pData;
    int n = width*height;
    int i;
    __m128d XMM0 = _mm_set1_pd(value);
    for (i = 0;i < (n);i += 8) {
      _mm_store_pd((x)+i  , XMM0);
      _mm_store_pd((x)+i+2, XMM0);
      _mm_store_pd((x)+i+4, XMM0);
      _mm_store_pd((x)+i+6, XMM0);
    }
  }
  
  
  template <>
  void Matrix<double>::create(int w, int h, double value)
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
      int size = roundOut(w*h);
#ifdef	_MSC_VER
      pData = (double*)_aligned_malloc(size*sizeof(double), 16);
#elif __APPLE__
      pData = (double*) malloc(size*sizeof(double));
#else
      pData = (double*)memalign(16, size*sizeof(double));
#endif
      width = w;
      height = h;
    }
    set(value);
  }
  
  
  
  
  template <>
  void Matrix<double>::add(const Matrix &m)
  {
#ifdef _DEBUG
    if(!sameDim(m)) {
      //error
      throw std::invalid_argument("Matrix dimension must agree for addition");
    }
#endif
    
    double* y = pData;
    double* x = m.pData;
    int n = width*height;
    int i;
    for (i = 0;i < (n);i += 8)
    {
      __m128d XMM0 = _mm_load_pd((x)+i  );
      __m128d XMM1 = _mm_load_pd((x)+i+2);
      __m128d XMM2 = _mm_load_pd((x)+i+4);
      __m128d XMM3 = _mm_load_pd((x)+i+6);
      __m128d XMM4 = _mm_load_pd((y)+i  );
      __m128d XMM5 = _mm_load_pd((y)+i+2);
      __m128d XMM6 = _mm_load_pd((y)+i+4);
      __m128d XMM7 = _mm_load_pd((y)+i+6);
      XMM4 = _mm_add_pd(XMM4, XMM0);
      XMM5 = _mm_add_pd(XMM5, XMM1);
      XMM6 = _mm_add_pd(XMM6, XMM2);
      XMM7 = _mm_add_pd(XMM7, XMM3);
      _mm_store_pd((y)+i  , XMM4);
      _mm_store_pd((y)+i+2, XMM5);
      _mm_store_pd((y)+i+4, XMM6);
      _mm_store_pd((y)+i+6, XMM7);
    }
  }
  
  template <>
  void Matrix<double>::subtract(const Matrix &m)
  {
#ifdef _DEBUG
    if(!sameDim(m)) {
      //error
      throw std::invalid_argument("Matrix dimension must agree for addition");
    }
#endif
    
    double* y = pData;
    double* x = m.pData;
    int n = width*height;
    int i;
    for (i = 0;i < (n);i += 8)
    {
      __m128d XMM0 = _mm_load_pd((x)+i  );
      __m128d XMM1 = _mm_load_pd((x)+i+2);
      __m128d XMM2 = _mm_load_pd((x)+i+4);
      __m128d XMM3 = _mm_load_pd((x)+i+6);
      __m128d XMM4 = _mm_load_pd((y)+i  );
      __m128d XMM5 = _mm_load_pd((y)+i+2);
      __m128d XMM6 = _mm_load_pd((y)+i+4);
      __m128d XMM7 = _mm_load_pd((y)+i+6);
      XMM4 = _mm_sub_pd(XMM4, XMM0);
      XMM5 = _mm_sub_pd(XMM5, XMM1);
      XMM6 = _mm_sub_pd(XMM6, XMM2);
      XMM7 = _mm_sub_pd(XMM7, XMM3);
      _mm_store_pd((y)+i  , XMM4);
      _mm_store_pd((y)+i+2, XMM5);
      _mm_store_pd((y)+i+4, XMM6);
      _mm_store_pd((y)+i+6, XMM7);
    }
  }
  
  
  // this = this * value;
  template <>
  void Matrix<double>::multiply(double value)
  {
    double* y = pData;
    int n = width*height;
    int i;
    __m128d XMM7 = _mm_set1_pd(value);
    for (i = 0;i < (n);i += 4) {
      __m128d XMM0 = _mm_load_pd((y)+i  );
      __m128d XMM1 = _mm_load_pd((y)+i+2);
      XMM0 = _mm_mul_pd(XMM0, XMM7);
      XMM1 = _mm_mul_pd(XMM1, XMM7);
      _mm_store_pd((y)+i  , XMM0);
      _mm_store_pd((y)+i+2, XMM1);
    }
  }
  
  
  // this = this .* this;
  template <>
  void Matrix<double>::eltMpy(const Matrix<double> &m)
  {
    double* y = pData;
    double* x = m.pData;
    int n = width*height;
    int i;
    for (i = 0;i < (n);i += 8) {
      __m128d XMM0 = _mm_load_pd((x)+i  );
      __m128d XMM1 = _mm_load_pd((x)+i+2);
      __m128d XMM2 = _mm_load_pd((x)+i+4);
      __m128d XMM3 = _mm_load_pd((x)+i+6);
      __m128d XMM4 = _mm_load_pd((y)+i  );
      __m128d XMM5 = _mm_load_pd((y)+i+2);
      __m128d XMM6 = _mm_load_pd((y)+i+4);
      __m128d XMM7 = _mm_load_pd((y)+i+6);
      XMM4 = _mm_mul_pd(XMM4, XMM0);
      XMM5 = _mm_mul_pd(XMM5, XMM1);
      XMM6 = _mm_mul_pd(XMM6, XMM2);
      XMM7 = _mm_mul_pd(XMM7, XMM3);
      _mm_store_pd((y)+i  , XMM4);
      _mm_store_pd((y)+i+2, XMM5);
      _mm_store_pd((y)+i+4, XMM6);
      _mm_store_pd((y)+i+6, XMM7);
    }
  }
  
  template <>
  void Matrix<double>::resize(int w, int h, double value)
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
    Matrix<double> tmpMatrix(w,h,value);
    for(int col=0;col<min_w;col++)
      memcpy(tmpMatrix.get() + col*h, pData + col*height, min_h*sizeof(double));
    
    // replace old matrix with new one
    set(tmpMatrix);
  }
  
}

#endif
