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

#include "evaluator.h"
#include <assert.h> 

/////////////////////////////////////////////////////////////////////
// EvaluatorMVLDCRF Class
/////////////////////////////////////////////////////////////////////

EvaluatorMVLDCRF::EvaluatorMVLDCRF(): Evaluator()
{}

EvaluatorMVLDCRF::EvaluatorMVLDCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen) 
: Evaluator(infEngine, featureGen)
{}

EvaluatorMVLDCRF::~EvaluatorMVLDCRF()
{}


// Computes OVERALL error given X
double EvaluatorMVLDCRF::computeError(DataSequence* X, Model* m)
{
	if(!pInfEngine || !pFeatureGen)
		throw HcrfBadPointer("In EvaluatorMVLDCRF::computeError");
 
	double partition = pInfEngine->computePartition(pFeatureGen, X, m, -1, false);
	double partitionMasked = pInfEngine->computePartition(pFeatureGen, X, m, -1, true);

	// return log(Z(h|x)) - log(Z(h*|x)) 
	return partition - partitionMasked;
}


// Compute the probability of each nodes in the datasequence given the model. 
// Returns a label vector and a probability matrix
void EvaluatorMVLDCRF::computeStateLabels(
	DataSequence* X, Model* m, iVector* vecStateLabels, dMatrix* prob)
{
	if(!pInfEngine || !pFeatureGen)
		throw HcrfBadPointer("EvaluatorMVLDCRF::computeStateLabels");

	Beliefs bel;
	pInfEngine->computeBeliefs(bel, pFeatureGen, X, m, false, -1, false);

	int nbNodes = (int) bel.belStates.size();
	int nbLabels = m->getNumberOfStateLabels();
	int seqLength = X->length();
	int nbViews = m->getNumberOfViews();

	vecStateLabels->create(seqLength);
	if( prob ) prob->create(seqLength, nbLabels);

	dMatrix sumBeliefsPerLabel(nbLabels,nbViews);
	dVector avgBeliefsPerLabel(nbLabels);

	// Belief at each frame is computed as an average of multi-view beliefs
	for(int xt=0; xt<seqLength; xt++) {
		sumBeliefsPerLabel.set(0);
		avgBeliefsPerLabel.set(0);
		// Sum up beliefs
		for(int v=0; v<nbViews; v++) {
			int xi = v*seqLength + xt;
			for(int h=0; h<m->getNumberOfStatesMV(v); h++)
				sumBeliefsPerLabel(v,m->getLabelPerStateMV(v)[h]) += bel.belStates[xi][h];
		}
		// Compute the average beliefs across views
		for(int y=0; y<nbLabels; y++) 
			avgBeliefsPerLabel[y] = sumBeliefsPerLabel.colSum(y) / nbViews;
		// Find the max belief
		int yt = 0;
		vecStateLabels->setValue(xt,yt);
		double maxBelief = avgBeliefsPerLabel[yt];
		if( prob ) prob->setValue(yt,xt,avgBeliefsPerLabel[yt]);
		for( yt=1; yt<nbLabels; yt++ ) {
			if( prob ) prob->setValue(yt,xt,avgBeliefsPerLabel[yt]);
			if( maxBelief < avgBeliefsPerLabel[yt] ) {
				vecStateLabels->setValue(xt,yt);
				maxBelief = avgBeliefsPerLabel[yt];
			}
		}
	}
}
