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

GradientMVHCRF::GradientMVHCRF(InferenceEngine* ie, FeatureGenerator* fg): Gradient(ie, fg)
{
}

double GradientMVHCRF::computeGradient(dVector& vecGradient, Model* m, DataSequence* X) 
{
	return computeGradientMLE(vecGradient,m,X);
}

double GradientMVHCRF::computeGradientMLE(dVector& vecGradient, Model* m, DataSequence* X)
{    
	double f_value=0; // return value

	////////////////////////////////////////////////////////////////////////////////////
	// Step 1 : Run Inference in each network to compute marginals conditioned on Y
 	int nbSeqLabels = m->getNumberOfSequenceLabels();
	std::vector<Beliefs> condBeliefs(nbSeqLabels);	
	dVector Partition(nbSeqLabels);
	 
	for(int y=0; y<nbSeqLabels; y++) 
	{ 
		pInfEngine->computeBeliefs(condBeliefs[y], pFeatureGen, X, m, true, y);
		Partition[y] = condBeliefs[y].partition;; 
	} 
	
	////////////////////////////////////////////////////////////////////////////////////
	// Step 2 : Compute expected values for node/edge features conditioned on Y
	int nbFeatures = pFeatureGen->getNumberOfFeatures();
	dMatrix condEValues(nbFeatures, nbSeqLabels);
	
	feature* f;
	featureVector vecFeatures;

	iMatrix adjMat;
	m->getAdjacencyMatrixMV(adjMat, X);
	
	int V = m->getNumberOfViews();
	int T = X->length(); 
	int nbNodes= V*T;

	double val;
	int y, k, xi, xj;
	
	for(y=0; y<nbSeqLabels; y++) 
	{ 
		// Loop over nodes to compute features and update the gradient
		for(xi=0; xi<nbNodes; xi++) {
			pFeatureGen->getFeatures(vecFeatures,X,m,xi,-1,y);			
			f = vecFeatures.getPtr();						
			for(k=0; k<vecFeatures.size(); k++, f++) {  				
				// p(h^v_t=a|x,y) * f_k(v,t,a,x,y)
				val = condBeliefs[y].belStates[xi][f->nodeState] * f->value;
				condEValues.addValue(y, f->globalId, val);
			} 
		} 

		// Loop over edges to compute features and update the gradient
		for(xi=0; xi<nbNodes; xi++) {
			for(xj=xi+1; xj<nbNodes; xj++) {
				if( !adjMat(xi,xj) ) continue;
				pFeatureGen->getFeatures(vecFeatures,X,m,xj,xi,y);
				f = vecFeatures.getPtr();				
				for(k=0; k<vecFeatures.size(); k++, f++) {
					// p(h^vi_ti=a,h^vj_tj=b|x,y) * f_k(vi,ti,vj,tj,x,y)
					val = condBeliefs[y].belEdges[adjMat(xi,xj)-1]
							(f->prevNodeState,f->nodeState) * f->value;
					condEValues.addValue(y, f->globalId, val);
				} 
			} 
		} 	
	} 

	////////////////////////////////////////////////////////////////////////////////////
	// Step 3: Compute Joint Expected Values
	dVector JointEValues(nbFeatures);
	dVector rowJ(nbFeatures);  // expected value conditioned on seqLabel Y
	double sumZLog = Partition.logSumExp();
	for (int y=0; y<nbSeqLabels; y++) 
	{
		condEValues.getRow(y, rowJ);
		rowJ.multiply( exp(Partition[y]-sumZLog) );
		JointEValues.add(rowJ);
	}
	
	////////////////////////////////////////////////////////////////////////////////////
	// Step 4 Compute Gradient as Exi[i,*,*] - Exi[*,*,*], that is the difference between 
	// expected values conditioned on seqLabel Y and joint expected values	
	if( vecGradient.getLength() != nbFeatures )
		vecGradient.create(nbFeatures);

	condEValues.getRow(X->getSequenceLabel(), rowJ); 
	JointEValues.negate();
	rowJ.add(JointEValues);
	vecGradient.add(rowJ);  

	// MLE: return log(sum_y' p(y'|xi)) - log(p(yi|xi)})	
	f_value = Partition.logSumExp() - Partition[X->getSequenceLabel()]; 
	return f_value;
}


