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

//-------------------------------------------------------------
// Matrix Class
//-------------------------------------------------------------
// TODO: - The order of parameter is inconsistant between construtor
//         ( width, height) and access (row, column).

//*
// Constructors and Destructor
//*
#include <iostream>
#include <math.h>
// Needed by GCC 4.3
#include <climits>

#if defined _MSC_VER
#pragma warning(disable : 4996)
#endif

using namespace std;
using namespace hCRF;

template <class elType>
Matrix<elType>::Matrix()
: dispPrecision(8)
, pData(0)
, height(0)
, width(0)
{
  
}

template<class elType>
Matrix<elType>::Matrix(int w, int h, elType value)
: dispPrecision(8)
, pData(0)
, height(0)
, width(0)

{
  if (w < 0 || h < 0)
    throw std::invalid_argument("Matrix dimensions must be greater than 0.");
  create(w, h, value);
}

template<class elType>
Matrix<elType>::Matrix(const Matrix &m)
: dispPrecision(8)
, pData(0)
, height(0)
, width(0)
{
  set(m);
}

template <class elType>
Matrix<elType>::~Matrix()
{
  freeMemory();
}


//*
// Public Methods
//*

template<class elType>
Matrix<elType>& Matrix<elType>::operator=(const Matrix<elType> &m)
{
  return set(m);
}

template <class elType>
Matrix<elType> Matrix<elType>::operator+(const Matrix<elType>&m) const
{
  Matrix<elType> ans = *this;
  ans+=m;
  return ans;
}

template <class elType>
Matrix<elType> Matrix<elType>::operator-(const Matrix<elType>&m) const
{
  Matrix<elType> ans = *this;
  ans -= m;
  return ans;
}

template <class elType>
Matrix<elType>& Matrix<elType>::operator+=(const Matrix &m)
{
  add(m);
  return *this;
}

template <class elType>
Matrix<elType>& Matrix<elType>::operator-=(const Matrix &m)
{
#ifdef _DEBUG
  if(!sameDim(m)) {
    throw std::invalid_argument("Matrix dimensions do not agree");
  }
#endif
  elType* ptrSource = m.pData;
  int sizeMatrix = height * width;
  for(elType* ptrData = pData; ptrData <pData+sizeMatrix;
      ptrData++, ptrSource++)
  {
    *(ptrData) -= *(ptrSource);
  }
  return *this;
}

template <class elType>
bool Matrix<elType>::operator==(const Matrix<elType> &m) const
{
  
  if ((height != m.height) || (width != m.width))
    return false;
  elType* ptrOther = m.pData;
  for (elType* ptrData = pData; ptrData < pData + height * width; ptrData++)
  {
    if (*ptrData != *ptrOther)
      return false;
    ptrOther++;
  }
  return true;
}

template <class elType>
bool Matrix<elType>::operator!=(const Matrix<elType> &m) const {
  return !(*this == m);
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
inline elType Matrix<elType>::operator ()(int r, int c) const
{
  //	return getValue(r,c);
#ifdef _DEBUG
  if(pData==0 || c<0 || c>=width || r<0 || r>=height)
  {
    throw std::invalid_argument("Out of bound index for matrix access");
  }
#endif
  
  return pData[c*height+r];
}

template <class elType>
inline elType& Matrix<elType>::operator ()(int r, int c)
{
#ifdef _DEBUG
  if(pData==0 || c<0 || c>=width || r<0 || r>=height)
  {
    throw std::invalid_argument("Out fo bound index for matix access");
  }
#endif
  
  return pData[c*height+r];
}

template <class elType>
inline void Matrix<elType>::getRow(int rowNum, Vector<elType>& row) const
{
  int numCols = getWidth();
  if (row.getLength() != numCols){
    row = Vector<elType>(numCols);
  }
  for(int i=0;i<numCols;i++)
  {
    row.setValue(i, getValue(rowNum,i));
  }
}

template <class elType>
inline void Matrix<elType>::getCol(int colNum, Vector<elType>& col) const
{
  int numRows = getHeight();
  if (col.getLength() != numRows){
    col = Vector<elType>(numRows);
  }
  for(int i=0;i<numRows;i++)
  {
    col.setValue(i, getValue(i,colNum));
  }
}

// Matrices are column indexed, like MATLAB
template <class elType>
inline elType& Matrix<elType>::getValue(int row, int col)
{
#ifdef _DEBUG
  if(pData==0 || col<0 || col>=width || row<0 || row>=height)
    throw std::invalid_argument("Out fo bound index for matrix read access");
#endif
  return pData[col*height+row];
}

template <class elType>
inline elType Matrix<elType>::getValue(int row, int col) const
{
#ifdef _DEBUG
  if(pData==0 || col<0 || col>=width || row<0 || row>=height)
    throw std::invalid_argument("Out fo bound index for matrix read access");
#endif
  return pData[col*height+row];
}

template <class elType>
inline int Matrix<elType>::setValue(int row, int col, elType value)
{
#ifdef _DEBUG
  if(pData==0 || col<0 || col>=width || row<0 || row>=height)
    throw std::invalid_argument("Out fo bound index for matrix write access");
#endif
  pData[col*height+row] = value;
  return 0;
}

template <class elType>
inline int Matrix<elType>::addValue(int row, int col, elType value)
{
#ifdef _DEBUG
  if(pData==0 || col<0 || col>=width || row<0 || row>=height)
    throw std::invalid_argument("Out fo bound index for matrix write access");
#endif
  pData[col*height+row] += value;
  return 0;
}

template <class elType>
void Matrix<elType>::set(elType *pd, int w, int h)
{
  freeMemory();
  if(w!=0 && h!=0)
  {
    create (w,h);
    memcpy(pData, pd, w*h*sizeof(elType));
  }
  else
  {
    pData = NULL;
    width = w;
    height = h;
  }
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

template <class elType>
Matrix<elType>& Matrix<elType>::set(const Matrix &m)
{
  freeMemory();
  if (m.getWidth()!=0 && m.getWidth() !=0)
  {
    create(m.getWidth(),m.getHeight());
    memcpy(pData,m.get(),m.getWidth()*m.getHeight()*sizeof(elType));
  }
  else
  {
    pData = NULL;
    width = m.getWidth();
    height = m.getHeight();
  }
  return *this;
}

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
void Matrix<elType>::add(elType value)
{
  int sizeMatrix = height * width;
  for(elType* ptrData = pData; ptrData < pData+sizeMatrix; ptrData++)
    *(ptrData) += value;
  
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
void Matrix<elType>::subtract(elType value)
{
  int sizeMatrix = height * width;
  for(elType* ptrData = pData; ptrData < pData+sizeMatrix; ptrData++)
    *(ptrData) -= value;
}

template <class elType>
void Matrix<elType>::negate()
{
  multiply(-1);
}

// this = this * value;
template <class elType>
void Matrix<elType>::multiply(elType value)
{
  int sizeMatrix = height * width;
  for(elType* ptrData = pData; ptrData < pData+sizeMatrix; ptrData++)
    *(ptrData) *= value;
}

// this = m1 * m2; (right multiply)
template <class elType>
void Matrix<elType>::multiply(const Matrix &m1, const Matrix &m2)
{
  if(m2.getHeight()!=m1.getWidth()) {
    throw std::invalid_argument("Matrix dimension must agree for multiplication");
  }
  create(m2.getWidth(),m1.getHeight());
  elType *temp_pData = pData;
  
  elType value;
  int row,col,m_row,m_col;
  for(row=0; row<m1.getHeight(); row++) {
    for(m_col=0; m_col<m2.getWidth(); m_col++) {
      value = 0;
      for(m_row=0,col=0; col<m1.getWidth(); m_row++,col++) {
        value += m2.getValue(m_row,m_col)*m1.getValue(row,col);
      }
      temp_pData[m_col*height+row] = value;
    }
  }
}

// this = this * m; (right multiply)
template <class elType>
void Matrix<elType>::multiply(const Matrix &m)
{
  if(m.getHeight()!=width) {
    throw std::invalid_argument("Matrix dimension must agree for multiplication");
  }
  
  // perform multiply and store in temporary memory
  elType *temp_pData;
  temp_pData = new elType[height*m.getWidth()];
  
  elType value;
  int row,col,m_row,m_col;
  for(row=0; row<height; row++) {
    for(m_col=0; m_col<m.getWidth(); m_col++) {
      value = 0;
      for(m_row=0,col=0; col<width; m_row++,col++) {
        value += m.getValue(m_row,m_col)*getValue(row,col);
      }
      temp_pData[m_col*height+row] = value;
    }
  }
  
  // copy over result matrix to this
  // TODO: The height and width seem inversed... (LP)
  create(height,m.getWidth());
  memcpy(pData, temp_pData, height*width*sizeof(elType));
  
  // de-allocate memory
  delete[] temp_pData;
  temp_pData = 0;
}


// this = this * value;
template <class elType>
void Matrix<elType>::divide(elType value)
{
#ifndef _DEBUG
  if( value==0 )
    throw std::invalid_argument("Div-by-zero");
#endif
  int sizeMatrix = height * width;
  for(elType* ptrData = pData; ptrData < pData+sizeMatrix; ptrData++)
    *(ptrData) /= value;
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

// this = this ./ this;
template <class elType>
void Matrix<elType>::eltDiv(const Matrix<elType> &m)
{
  elType* ptrData = pData;
  elType* ptrSource = m.pData;
  int sizeMatrix = height * width;
  
  for(int i = 0; i < sizeMatrix; i++ )
  {
    if( *ptrSource==0 )
      throw std::invalid_argument("Div-by-zero");
    *(ptrData) /= *(ptrSource);
    ptrData++;
    ptrSource++;
  }
}

// sum(this);
template <class elType>
double Matrix<elType>::sum()
{
  double sumElement = 0;
  int sizeMatrix = height * width;
  
  for(elType* ptrData = pData; ptrData < pData+sizeMatrix; ptrData++)
  {
    sumElement += *(ptrData);
  }
  
  return sumElement;
}

template <class elType>
void Matrix<elType>::eltSqr()
{
  elType* ptrData = pData;
  int sizeMatrix = height * width;
  
  for(int i = 0; i < sizeMatrix; i++ )
  {
    *(ptrData) *= *(ptrData);
    ptrData++;
  }
}

// this = sqrt(this);
template <class elType>
void Matrix<elType>::eltSqrt()
{
  int sizeMatrix = height * width;
  
  for(elType* ptrData = pData; ptrData < pData+sizeMatrix; ptrData++ )
  {
    *(ptrData) = sqrt(*ptrData);
  }
}

// this = exp(this);
template <class elType>
void Matrix<elType>::eltExp()
{
  elType* ptrData = pData;
  int sizeMatrix = height * width;
  
  for(int i = 0; i < sizeMatrix; i++ )
  {
    *(ptrData) = exp(*(ptrData));
    ptrData++;
  }
}

// this = this .* this;
template <class elType>
void Matrix<elType>::eltLog()
{
  // this = log(this);
  elType* ptrData = pData;
  int sizeMatrix = height * width;
  
  for(int i = 0; i < sizeMatrix; i++ )
  {
    *(ptrData) = log(*(ptrData));
    ptrData++;
  }
}

template <class elType>
void Matrix<elType>::transpose()
{
  if(width==0 || height==0) {
    return;
  }
  
  Matrix<elType> m(height,width);
  
  int row, col;
  for(col = 0; col < width; col++) {
    for(row = 0; row < height; row++) {
      m(col,row) = getValue(row,col);
    }
  }
  
  set(m);
}

template <class elType>
void Matrix<elType>::abs()
{
  elType* ptrData = pData;
  
  for(int i = 0; i < width*height; i++ )
  {
    *(ptrData) = fabs(*(ptrData));
    ptrData++;
  }
}

template <class elType>
elType Matrix<elType>::absmax()
{
  elType* ptrData = pData;
  elType max_val = -DBL_MAX;
  for(int i = 0; i < width*height; i++ )
  {
    elType abs_val = fabs(*ptrData);
    if( abs_val>max_val ) max_val = abs_val;
    ptrData++;
  }
  return max_val;
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
elType Matrix<elType>::rowSum(int row) const
{
  elType rowsum = 0;
  int col;
  for(col=0; col<width; col++) {
    rowsum += getValue(row,col);
  }
  
  return rowsum;
}

template <class elType>
elType Matrix<elType>::colSum(int col) const
{
  elType colsum = 0;
  int row;
  for(row=0; row<height; row++) {
    colsum += getValue(row,col);
  }
  
  return colsum;
}


template <class elType>
elType Matrix<elType>::l1Norm() const
{
  elType sumElement(0);
  int sizeMatrix = height * width;
  for(elType* ptrData = pData; ptrData < pData+sizeMatrix; ptrData++)
  {
    if (*ptrData <0)
      sumElement -= *(ptrData);
    else
      sumElement += *(ptrData);
  }
  return sumElement;
}

template <class elType>
elType Matrix<elType>::l2Norm(bool root) const
{
  elType total(0);
  int sizeMatrix = height * width;
  for(elType* ptrData = pData; ptrData < pData+sizeMatrix; ptrData++)
  {
    total += (*ptrData) * (*ptrData);
  }
  if (root)
  {
    total = sqrt(total);
  }
  return total;
}

template <class elType>
void Matrix<elType>::rowSum(Vector<elType>& vecSum) const
{
  if(vecSum.getLength() != height)
    vecSum.create(height);
  vecSum.set(0);
  elType* pSum;
  elType* pMatrix = pData;
  
  for(int col=0; col < width; col++)
  {
    pSum = vecSum.pData;
    for(int row = 0; row < height;row++)
    {
      *pSum += *pMatrix;
      pSum++;
      pMatrix++;
    }
  }
}

template <class elType>
elType Matrix<elType>::getMaxValue()
{
  elType* tmpData = pData;
  elType MaxValue = 0;
  int sizeMatrix = width*height;
  if(sizeMatrix > 0)
    MaxValue = *(tmpData++);
  
  for(int i = 1; i < sizeMatrix; i++)
  {
    if(*tmpData > MaxValue)
      MaxValue = *tmpData;
    tmpData++;
  }
  return MaxValue;
}

template <class elType>
elType Matrix<elType>::logSumExp() const
{
  double m1, m2, sub;
  // Z = log(exp(x_0)) = x_0
  double Z = getValue(0,0);
  for(int r=0; r<getHeight(); r++) {
    for(int c=0; c<getWidth(); c++) {
      if( r==0 && c==0 ) continue;
      if( Z >= getValue(r,c) ) {
        m1 = Z;
        m2 = getValue(r,c);
      } else {
        m1 = getValue(r,c);
        m2 = Z;
      }
      sub = (m2==INT_MIN && m1==INT_MIN) ? 0 : m2 - m1;
      Z = m1 + log(1 + exp(sub));
    }
  }
  return Z;
}

template <class elType>
inline int Matrix<elType>::getWidth() const
{
  return width;
}

template <class elType>
inline int Matrix<elType>::getHeight() const
{
  return height;
}

template <class elType>
inline elType* Matrix<elType>::get() const
{
  return pData;
}

// returns 1 on error, 0 otherwise
template <class elType>
int Matrix<elType>::read(std::istream* stream)
{
  // make sure that we are not at the end of the file
  if(stream->eof()) {
    return 1;
  }
  
  // get width and height
  int w,h,bsize = MATRIX_READ_BUFF_SIZE;
  char buff[MATRIX_READ_BUFF_SIZE];
  memset(buff,0,bsize*sizeof(char));
  
  stream->getline(buff, bsize, ',');
  if(stream->fail() || sscanf(buff,"%d",&h)==0) return 1;
  
  stream->getline(buff, bsize, '\n');
  if(stream->fail() || sscanf(buff,"%d",&w)==0) return 1;
  
  // read in matrix
  double value=0;
  elType *temp_pData;
  temp_pData = new elType [w*h];
  int row, col;
  for(row=0; row<h; row++) {
    for(col=0;col<w-1;col++) {
      stream->getline(buff, bsize, ',');
      if(stream->fail() || sscanf(buff,"%lf",&value)==0) {
        delete[] temp_pData;
        temp_pData = 0;
        return 1;
      }
      
      temp_pData[row*w+col] = (elType) value;
    }
    
    stream->getline(buff, bsize, '\n');
    if(stream->fail() || sscanf(buff,"%lf",&value)==0) {
      delete[] temp_pData;
      temp_pData = 0;
      return 1;
    }
    
    temp_pData[row*w+col] = (elType) value;
  }
  
  //success! - copy contents to this matrix
  create(w,h);
  for(row=0; row<h; row++) {
    for(col=0;col<w;col++) {
      setValue(row,col,temp_pData[row*w+col]);
    }
  }
  
  // done!
  delete[] temp_pData;
  temp_pData = 0;
  return 0;
}

template <class elType>
int Matrix<elType>::write(std::ostream* stream) const
{
  //output empty matrix if necessary (width or height equal 0)
  /*
   if(height==0 && width==0) {
   return 1;
   }*/
  
  (*stream) << height << "," << width << "\n";
  if(stream->fail()) return 1;
  
  int row,col;
  for(row=0;row<height;row++) {
    for(col=0;col<width;col++) {
      //TODO: Doesn't work with unsigned char
      (*stream) << setiosflags(std::ios::fixed) <<
        std::setprecision(dispPrecision) << getValue(row,col);
      if(stream->fail()) return 1;
      
      if(col<width-1) {
        (*stream) << ",";
        if(stream->fail()) return 1;
      }
    }
    (*stream) << "\n";
    if(stream->fail()) return 1;
  }
  
  return 0;
}

template <class elType>
void Matrix<elType>::display(std::ostream* stream) const
{
  // This function is used to display to the screen. To write to a file and be
  // able to read it using read, one should use the write method
  if(height==0 && width==0) {
    (*stream) << "EMPTY MATRIX" << "\n";
    return;
  }
  int row,col;
  std::ios::fmtflags saved_flags = stream->flags();
  for(row=0;row<height;row++) {
    if (row==0) {
      (*stream) << "[ ";
    } else {
      (*stream) << "  ";
    }
    for(col=0;col<width;col++) {
      (*stream) << setiosflags(std::ios::fixed) <<
        std::setprecision(dispPrecision) << getValue(row,col) << " ";
    }
    if (row< height-1) {
      (*stream) << "\n";
    }else {
      (*stream)<<"]\n";
    }
  }
  stream->flags(saved_flags);
}

template <class elType>
void Matrix<elType>::display() const
{
  display(&(std::cout));
}

template <class elType>
void Matrix<elType>::setDisplayPrecision(int prec)
{
  dispPrecision = prec;
}

//*
// Private Methods
//*

template <class elType>
bool Matrix<elType>::validIds(int row, int col)const
{
	 return !(pData==0 || col<0 || col>=width || row<0 || row>=height);
}

template <class elType>
bool Matrix<elType>::sameDim(const Matrix &m) const
{
  return (m.getWidth()==width && m.getHeight()==height);
}


template <class elType>
void Matrix<elType>::freeMemory()
{
  if (pData) {
    delete[] pData;
  }
  pData = 0;
}



//-------------------------------------------------------------
// Vector Class
//-------------------------------------------------------------

//*
// Constructors and Destructor
//*

template <class elType>
Vector<elType>::Vector()
: Matrix<elType>()
{
  //does nothing
}

// default is column vector
template <class elType>
Vector<elType>::Vector(int len, VectorType vt, elType value)
{
  create(len, vt, value);
}

template <class elType>
Vector<elType>::Vector(const Vector& v):Matrix<elType>::Matrix()
{
  Vector<elType>::set(v);
}

template <class elType>
Vector<elType>::~Vector()
{
  //does nothing (implemented in Matrix parent class)
}

//*
// Public Methods
//*

template <class elType>
Vector<elType>& Vector<elType>::operator=(const Vector& v)
{
  set(v);
  return *this;
}


template <class elType>
void Vector<elType>::create(int len, VectorType vt, elType value)
{
  int w = 1, h = len;
  if(vt==ROWVECTOR) {
    w = len;
    h = 1;
  }
  Matrix<elType>::create(w, h, value);
}

template <class elType>
inline elType Vector<elType>::operator [](int i) const
{
  //	return getValue(i);
  return this->pData[i];
}

template <class elType>
inline elType& Vector<elType>::operator [](int i)
{
  return this->pData[i];
}

template <class elType>
void Vector<elType>::set(const Vector& v)
{
  Matrix<elType>::set(v);
}

template <class elType>
void Vector<elType>::set(elType value)
{
  Matrix<elType>::set(value);
}

template <class elType>
void Vector<elType>::setValue(int i, elType value)
{
  this->pData[i] = value;
  /*	if(getType()==ROWVECTOR) {
   return Matrix<elType>::setValue(0,i,value);
   } else {
   return Matrix<elType>::setValue(i,0,value);
   }
   */
}

template <class elType>
void Vector<elType>::addValue(int i, elType value)
{
  this->pData[i] += value;
}

template <class elType>
elType Vector<elType>::getValue(int i) const
{
  return this->pData[i];
  /*	if(getType()==ROWVECTOR) {
   return Matrix<elType>::getValue(0,i);
   } else {
   return Matrix<elType>::getValue(i,0);
   }
   */
}

//TODO: rewrite as Matrix::sum
template <class elType>
elType Vector<elType>::sum() const
{
  if(getType()==ROWVECTOR) {
    return Matrix<elType>::rowSum(0);
  } else {
    return Matrix<elType>::colSum(0);
  }
}

template <class elType>
elType Vector<elType>::max() const
{
  int length = getLength();
  elType maximum = this->pData[0];
  for (elType* current = this->pData+1; current<this->pData+length; current++) {
    if( *current > maximum)
      maximum = *current;
  }
  return maximum;
}

template <class elType>
elType Vector<elType>::min() const
{
  int length = getLength();
  elType minimum = this->pData[0];
  for (elType* current = this->pData+1; current<this->pData+length; current++) {
    if( *current < minimum)
      minimum = *current;
  }
  return minimum;
}


template <class elType>
int Vector<elType>::getLength() const
{
  return this->width*this->height;
}

template <class elType>
VectorType Vector<elType>::getType() const
{
  if(this->width==1) {
    return COLVECTOR;
  } else {
    return ROWVECTOR;
  }
}

template <class elType>
elType Vector<elType>::logSumExp() const
{
  /* This member compute log(sum_i(exp(v_i))) for the verctor
   v. This function avoid explicilty computing the sum to
   avoid overflow
   */
  double m1, m2, sub;
  // Z = log(exp(x_0)) = x_0
  double Z=getValue(0);
  for(int col=1;col<getLength();col++){
    if(Z >= getValue(col)){
      m1=Z;
      m2=getValue(col);
    }
    else {
      m1=getValue(col);
      m2=Z;
    }
    // We want to compute log(exp(m1) + exp(m2))
    // Let m1 > m2 and do some algebra :
    // log(exp(m1) + exp(m2)) = log(exp(m_1) + exp(m1+sub))
    //							log((1+exp(sub))*exp(m_1))
    //							log(exp(m1)) + log(1+exp(sub))
    //							m1 + log(1+exp(sub))
    // This avoid explicitly computing exp(m1)+exp(m2)
    // which could overflood the double type
    sub=m2-m1;
    if(m2==INT_MIN && m1==INT_MIN){ // possible problem
      sub=0;
    }
    Z=m1 + log(1 + exp(sub));
  }
  return Z;
}

//-------------------------------------------------------------
// stream io routines
//-------------------------------------------------------------

template <class elType>
std::istream& operator >>(std::istream& in, Matrix<elType>& m)
{
  m.read(&in);
  return in;
}

template <class elType>
std::istream& operator >>(std::istream& in, Vector<elType>& v)
{
  v.read(&in);
  return in;
}

template <class elType>
std::ostream& operator <<(std::ostream& out, const Matrix<elType>& m)
{
  m.display(&out);
  return out;
}


template <class elType>
MatrixSparse<elType>::MatrixSparse()
: numElement(0), numCol(0),numRow(0)
{
  pr = new Vector<elType>();
  ir = new Vector<size_t>();
  jc = new Vector<size_t>();
}

template <class elType>
MatrixSparse<elType>::MatrixSparse(const MatrixSparse<elType>& M)
: numElement(M.numElement)
, numCol(M.numCol)
, numRow(M.numRow)
{
  pr = new Vector<elType>();
  pr->set(*M.pr);
  ir = new Vector<size_t>();
  ir->set(*M.ir);
  jc = new Vector<size_t>();
  jc->set(*M.jc);
}

template <class elType>
MatrixSparse<elType>& MatrixSparse<elType>::operator=(const MatrixSparse<elType>& M)
{
  if(pr)
    delete pr;
  if(ir)
    delete ir;
  if(jc)
    delete jc;
  numElement = M.numElement;
  numCol = M.numCol;
  numRow= M.numRow;
  pr = new Vector<elType>();
  pr->set(*M.pr);
  ir = new Vector<size_t>();
  ir->set(*M.ir);
  jc = new Vector<size_t>();
  jc->set(*M.jc);
  return this;
}


template <class elType>
MatrixSparse<elType>::~MatrixSparse()
{
  if(pr)
    delete pr;
  if(ir)
    delete ir;
  if(jc)
    delete jc;
}

template <class elType>
inline size_t MatrixSparse<elType>::getNumOfElements()
{
  if(numElement == 0)
  {
    if(jc)
    {
      numElement = jc->getValue((int)numCol);
    }
  }
  return numElement;
}

template <class elType>
inline size_t MatrixSparse<elType>::getWidth() const
{
  return numCol;
}


template <class elType>
inline void MatrixSparse<elType>::setHeight(size_t n)
{
  numRow = n;
}

template <class elType>
inline size_t MatrixSparse<elType>::getHeight()
{
  return numRow;
}

template <class elType>
inline Vector<elType>* MatrixSparse<elType>::getPr() const
{
  return pr;
}

template <class elType>
inline Vector<size_t>* MatrixSparse<elType>::getIr() const
{
  return ir;
}

template <class elType>
inline Vector<size_t>* MatrixSparse<elType>::getJc() const
{
  return jc;
}

template <class elType>
void MatrixSparse<elType>::createJc(size_t numC)
{
  if(numC != numCol)
  {
    numCol = numC;
    jc->create((int)numC + 1);
  }
}

template <class elType>
void MatrixSparse<elType>::createPrIr(size_t numE)
{
  if(numE != numElement || pr->getLength() != numE || ir->getLength() != numE)
  {
    numElement = numE;
    pr->create((int)numE);
    ir->create((int)numE);
  }
}

// returns 1 on error, 0 otherwise
template <class elType>
int MatrixSparse<elType>::read(std::istream* stream)
{
  // make sure that we are not at the end of the file
  if(stream->eof()) {
    return 1;
  }
  
  // get width and height
  int h, w, i,j;
  double v;
  int nbNonZeros = 0;
  int lastColumn = -1;
  
  (*stream) >> h >> w >> v;
  
  setHeight(h);
  createJc(w);
  getJc()->set(0);
  createPrIr(1000);
  // read in matrix
  while(!stream->eof())
  {
    std::streampos prevpos = stream->tellg();
    (*stream) >> i >> j >> v;
    if (v==0)
    {
      stream->seekg(prevpos);
      break;
    }
    else
    {
      if(nbNonZeros >= getIr()->getLength())
      {
        getIr()->resize(1,nbNonZeros+1000);
        getPr()->resize(1,nbNonZeros+1000);
      }
      getIr()->setValue(nbNonZeros,i-1);
      getPr()->setValue(nbNonZeros,v);
      if(lastColumn != j)
      {
        getJc()->setValue(j-1,nbNonZeros);
        lastColumn = j;
      }
      nbNonZeros++;
      for(int k = j; k <= w; k++)
        getJc()->setValue(k,nbNonZeros);
    }
    
  }
  getIr()->resize(1,nbNonZeros);
  getPr()->resize(1,nbNonZeros);
  numElement=nbNonZeros;
  return 0;
}


