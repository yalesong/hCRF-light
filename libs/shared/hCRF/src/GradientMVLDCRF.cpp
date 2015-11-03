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


#include "gradient.h"  

GradientMVLDCRF::GradientMVLDCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen):
Gradient(infEngine, featureGen)
{}

double GradientMVLDCRF::computeGradient(dVector& vecGradient, Model* m, DataSequence* X)
{ 
	return computeGradientMLE(vecGradient,m,X);
}
 
double GradientMVLDCRF::computeGradientMLE(dVector& vecGradient, Model* m, DataSequence* X)
{
	////////////////////////////////////////////////////////////////////////////////////
	// Step 1 : Run Inference in each network to compute marginals
	Beliefs bel, belMasked;
	pInfEngine->computeBeliefs(bel, pFeatureGen, X, m, true, -1, false);
	pInfEngine->computeBeliefs(belMasked, pFeatureGen, X, m, true, -1, true);

	// This is the value to be returned
	double f_value =  bel.partition - belMasked.partition;

	
	////////////////////////////////////////////////////////////////////////////////////
	// Step 2 : Update the gradient
	int nbFeatures = pFeatureGen->getNumberOfFeatures();
	if( vecGradient.getLength() != nbFeatures )
		vecGradient.create(nbFeatures); 
	
	feature* f; 
	featureVector vecFeatures;

	iMatrix adjMat;
	m->getAdjacencyMatrixMV(adjMat, X);
	
	int V = m->getNumberOfViews();
	int T = X->length(); 
	int nbNodes= V*T;

	
	// For xxLDCRF
	int seqLabel = -1;

	// Loop over nodes to compute features and update the gradient
	for(int xi=0; xi<nbNodes; xi++) {
		pFeatureGen->getFeatures(vecFeatures,X,m,xi,-1,seqLabel);			
		f = vecFeatures.getPtr();						
		for(int k=0; k<vecFeatures.size(); k++, f++) {  				
			// p(h^v_t=a|x,y) * f_k(v,t,a,x,y)
			double gain = bel.belStates[xi][f->nodeState] - belMasked.belStates[xi][f->nodeState];
			vecGradient[f->globalId] -= gain * f->value;
		} 
	} 

	// Loop over edges to compute features and update the gradient
	for(int xi=0; xi<nbNodes; xi++) {
		for(int xj=xi+1; xj<nbNodes; xj++) {
			if( !adjMat(xi,xj) ) continue;
			pFeatureGen->getFeatures(vecFeatures,X,m,xj,xi,seqLabel);
			f = vecFeatures.getPtr();				
			for(int k=0; k<vecFeatures.size(); k++, f++) {
				// p(h^vi_ti=a,h^vj_tj=b|x,y) * f_k(vi,ti,vj,tj,x,y)
				double gain = bel.belEdges[adjMat(xi,xj)-1](f->prevNodeState,f->nodeState)
							- belMasked.belEdges[adjMat(xi,xj)-1](f->prevNodeState,f->nodeState);
				vecGradient[f->globalId] -= gain * f->value;
			} 
		} 
	} 	
 
	return f_value;
}

 