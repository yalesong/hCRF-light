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


#include "inferenceengine.h"
#ifdef _OPENMP
#include <omp.h>
#endif
//-------------------------------------------------------------
// InferenceEngineBP Class
//-------------------------------------------------------------

//*
// Constructor and Destructor
//*

InferenceEngine::InferenceEngine()
{
	nbThreadsMP = 1;
}

InferenceEngine::~InferenceEngine()
{
}

void InferenceEngine::setMaxNumberThreads(int maxThreads)
{
	if (nbThreadsMP < maxThreads)
	{
		nbThreadsMP = maxThreads;
	}
}


void InferenceEngine::computeLogMi(FeatureGenerator* fGen, Model* model, DataSequence* X, 
	int i, int seqLabel, dMatrix& Mi_YY, dVector& Ri_Y, bool takeExp, bool bUseStatePerNodes)
{
	int k, offset, mv_offset, hss_offset;
	Ri_Y.set(0);
	Mi_YY.set(0);

	dVector* lambda = model->getWeights();

	feature* f;
	featureVector vecFeatures;

	// this will break if multi-view and HSS are used together
	mv_offset  = model->getCurrentView() * X->length();
	hss_offset = model->getCurrentFeatureLayer() * X->length(); 
	offset =  mv_offset + hss_offset; 

	// Singleton
	fGen->getFeatures(vecFeatures,X,model,i+offset,-1,seqLabel);
	f = vecFeatures.getPtr();
	for( k=0; k<vecFeatures.size(); k++,f++ ) 
		Ri_Y.addValue(f->nodeState, (*lambda)[f->globalId]*f->value);
	
	// Pairwise
	if( i>0 ) {
		fGen->getFeatures(vecFeatures,X,model,i+offset,i+offset-1,seqLabel);
		f = vecFeatures.getPtr();
		for( k=0; k<vecFeatures.size(); k++,f++ )
			Mi_YY.addValue(f->prevNodeState,f->nodeState, (*lambda)[f->globalId]*f->value);
	}

	double maskValue = -INF_VALUE;
	if( takeExp ) {
		Ri_Y.eltExp();
		Mi_YY.eltExp();
		maskValue = 0;
	}

	if(bUseStatePerNodes) {
		// This take into account the sharing of the state.
		iMatrix* pStatesPerNodes = model->getStateMatrix(X);
		for(int s = 0; s < Ri_Y.getLength(); s++) {
			if(pStatesPerNodes->getValue(s,i) == 0)
				Ri_Y.setValue(s,maskValue);
		}
	}
}

void InferenceEngine::LogMultiply(dMatrix& Potentials, dVector& Beli, 
								  dVector& LogAB)
{
	// The output is stored in LogAB. . This function compute
	// log(exp(P) * exp(B)).
	// It is safe to have Beli and LogAB reference the same variable
	//TODO: Potential may not be correctly initilized when we enter
	int row;
	int col;
	dMatrix temp;
	double m1;
	double m2;
	double sub;
	temp.create(Potentials.getWidth(),Potentials.getHeight());
	for(row=0;row<Potentials.getHeight();row++){
		for(col=0;col<Potentials.getWidth();col++){
			temp.setValue(row, col, Potentials.getValue(row,col) + 
						  Beli.getValue(col));
		}
	}
	// WARNING: Beli should not be used after this point as it can be the same
	// as LogAB and thus may be overwritten by the results.
	for(row=0;row<Potentials.getHeight();row++){
		LogAB.setValue(row,temp.getValue(row,0));
		for(col=1;col<temp.getWidth();col++){
			if(LogAB.getValue(row) >= temp.getValue(row,col)){
				m1=LogAB.getValue(row);
				m2=temp.getValue(row,col);
			} else {
				m1=temp.getValue(row,col);
				m2=LogAB.getValue(row);
			}
			sub=m2-m1;
			LogAB.setValue(row,m1 + log(1 + exp(sub)));
		}

	}
}

void InferenceEngine::logMultiply(dVector src_Vi, dVector src_Vj, dMatrix& dst_Mij)
{
	for(int r=0; r<src_Vi.getLength(); r++) 
		for(int c=0; c<src_Vj.getLength(); c++)
			dst_Mij(r,c) += (src_Vi[r]+src_Vj[c]);
}

void InferenceEngine::logMultiply(dVector src_Vi, dMatrix src_Mij, dVector& dst_Vj)
{ 
	dVector tmp_Vi;
	for(int c=0; c<src_Mij.getWidth(); c++) {
		tmp_Vi.set(src_Vi);
		for(int r=0; r<src_Mij.getHeight(); r++)
			tmp_Vi[r] += src_Mij(r,c);
		dst_Vj[c] = tmp_Vi.logSumExp();
	}
}

int InferenceEngine::logMultiplyMaxProd(dVector src_Vi, dMatrix src_Mij, dVector& dst_Vj)
{ 
	int max_idx=-1; 
	double max_val=-DBL_MAX;
	
	dVector tmp_Vi;
	for(int c=0; c<src_Mij.getWidth(); c++) {
		max_idx = -1; max_val=-DBL_MAX; 
		tmp_Vi.set(src_Vi); 
		for(int r=0; r<src_Mij.getHeight(); r++) {
			tmp_Vi[r] += src_Mij(r,c);
			if( tmp_Vi[r] > max_val ) { 
				max_val = tmp_Vi[r]; 
				max_idx = r; 
			}
		}
		dst_Vj[c] = tmp_Vi[max_idx];
	}
	return max_idx;
} 
