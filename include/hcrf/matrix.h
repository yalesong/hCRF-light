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

#ifndef MATRIX_H
#define MATRIX_H

#define MATRIX_READ_BUFF_SIZE 32

#include <iostream>
#include <iomanip>
#include <memory.h>
#include <stdio.h>
#include <stdexcept>
#include <float.h>

namespace hCRF
{
  //We use a not anonymous enum to avoid implict conversion from int
  //in function call. This allow to do parameter checking
  
  enum VectorType{
    ROWVECTOR,
    COLVECTOR
  };
  
  template <class elType> class Vector;
  
  template <class elType>
  class Matrix {
  public:
    Matrix();
    virtual ~Matrix();
    Matrix(int width, int height, elType value=0);
    Matrix(const Matrix &m);
    Matrix& operator=(const Matrix &m);
    
    Matrix operator+(const Matrix &m) const;
    Matrix operator-(const Matrix &m) const;
    Matrix& operator+=(const Matrix &m);
    Matrix& operator-=(const Matrix &m);
    
    bool operator==(const Matrix &m) const;
    bool operator!=(const Matrix &m) const;
    
    void create(int width, int height, elType value = 0);
    
    // access
    inline elType operator ()(int r, int c) const;
    inline elType& operator ()(int r, int c);
    inline void getRow(int row, Vector<elType>& result) const;
    inline void getCol(int col, Vector<elType>& result) const;
    inline elType& getValue(int row, int col);
    inline elType getValue(int row, int col) const;
    inline int setValue(int row, int col, elType value);
    inline int addValue(int row, int col, elType value);
    
    void set(elType *pData, int width, int height);
    Matrix& set(const Matrix &m);
    void set(elType value);
    void add(const Matrix &m);
    void add(elType value);
    void subtract(const Matrix &m);
    void subtract(elType value);
    void negate();
    void multiply(const Matrix &m);
    void multiply(const Matrix &m1,const Matrix &m2);
    void multiply(elType value);
    void divide(elType value);
    void eltMpy(const Matrix &m);
    void eltDiv(const Matrix &m);
    double sum();
    void eltSqr();
    void eltSqrt();
    void eltExp();
    void eltLog();
    void transpose();
    void abs();
    elType absmax();
    
    void resize(int width, int height, elType value=0);
    
    elType rowSum(int row) const;
    elType colSum(int col) const;
    elType l1Norm() const;
    elType l2Norm(bool roots=true) const;
    void rowSum(Vector<elType>& vecSum) const;
    
    elType getMaxValue();
    elType logSumExp() const;
    
    inline int getWidth() const;
    inline int getHeight() const;
    inline elType* get() const;
    
    // returns 1 on error, 0 otherwise
    virtual int read(std::istream* stream);
    virtual int write(std::ostream* stream) const;
    
    void display(std::ostream* stream) const;
    void display() const;
    
    void setDisplayPrecision(int prec);
    
    int dispPrecision;
    
  protected:
    elType *pData;
    int height, width;
    
  private:
    bool validIds(int row, int col)const;
    bool sameDim(const Matrix &m) const;
    void freeMemory();
  };
  
  template <class elType>
  class Vector: public Matrix<elType> {
  public:
    Vector();
    ~Vector();
    Vector(int length, VectorType vtype=COLVECTOR, elType value=0);
    Vector(const Vector& v);
    
    Vector& operator=(const Vector& v);
    
    void create(int length, VectorType vtype=COLVECTOR, elType value=0);
    
    // access
    inline elType operator [](int i) const;
    inline elType& operator [](int i);
    
    void set(const Vector& v);
    void set(elType value);
    void setValue(int i, elType value);
    void addValue(int i, elType value);
    inline elType getValue(int i) const;
    elType sum() const;
    elType max() const;
    elType min() const;
    elType logSumExp() const;
    int getLength() const;
    VectorType getType() const;
  };
  
  typedef Matrix<unsigned char> uMatrix;
  typedef Matrix<int> iMatrix;
  typedef Matrix<float> fMatrix;
  typedef Matrix<double> dMatrix;
  
  typedef Vector<unsigned char> uVector;
  typedef Vector<int> iVector;
  typedef Vector<float> fVector;
  typedef Vector<double> dVector;
  
  template <class elType>
  class MatrixSparse {
  public:
    MatrixSparse();
    MatrixSparse(const MatrixSparse<elType>&);
    MatrixSparse<elType> & operator=(const MatrixSparse<elType>&);
    ~MatrixSparse();
    
    Vector<size_t>* getJc() const;
    Vector<size_t>* getIr() const;
    Vector<elType>* getPr() const;
    
    
    void createJc(size_t numCol);
    void createPrIr(size_t numElement);
    // access
    //inline elType getValue_Sparse(int row, int nonzeroElementIndex)const;
    
    //MatrixSparse& set(const MatrixSparse &m);
    
    
    //inline: to make code directly embed in the program without using the function stack. Matrix is frequently used,
    //so it is implemented as 'inline'. this is also the reason for the matrix.inl
    inline size_t getNumOfElements();
    inline size_t getWidth() const;
    inline void setHeight(size_t);
    inline size_t getHeight();
    // returns 1 on error, 0 otherwise
    //virtual int read(std::istream* stream);
    //virtual int write(std::ostream* stream) const;
    //int display(std::ostream* stream) const;
    //void display() const;
    //void setDisplayPrecision(int prec);
    //int dispPrecision;
    
    // returns 1 on error, 0 otherwise
    int read(std::istream* stream);
    //int write(std::ostream* stream) const;
    
    
  protected:
    Vector<elType> *pr;
    Vector<size_t> *ir;
    Vector<size_t> *jc;
    
    size_t numElement;
    size_t numCol;
    size_t numRow;
    
    //static const int vtlConst=50;
    //int vtlHeight;
    //static const int upper_vtl=1000; // the upper bound of row number, if big than this, then auto make virtual matrix
    //static const int memoryRowNum=50; // the real rows of the matrix, this is strictly required to make it vulnerable
    
    //private:
    
  };
  typedef MatrixSparse<double> dMatrixSparse;
}


// stream io routines
template <class elType>
std::istream& operator >>(std::istream& in, hCRF::Matrix<elType>& m);
template <class elType>
std::istream& operator >>(std::istream& in, hCRF::Vector<elType>& v);
template <class elType>
std::ostream& operator <<(std::ostream& out, const hCRF::Matrix<elType>& m);

// include inline source
#include "../src/hcrf/matrix.inl"

#endif
