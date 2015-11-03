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
using namespace std;

/////////////////////////////////////////////////////////////////////
// Evaluator HCRF Class
/////////////////////////////////////////////////////////////////////

// *
// Constructor and Destructor
// *

EvaluatorLDCRF::EvaluatorLDCRF() 
:Evaluator()
{}

EvaluatorLDCRF::EvaluatorLDCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen) 
:Evaluator(infEngine, featureGen)
{}

EvaluatorLDCRF::~EvaluatorLDCRF()
{}

// *
// Public Methods
// *

//computes OVERALL error of the datasequence
double EvaluatorLDCRF::computeError(DataSequence* X, Model* m)
{
	if(!pInfEngine || !pFeatureGen){
		throw HcrfBadPointer("In EvaluatorLDCRF::computeError");
	}
	double partition, partitionMasked;

	partitionMasked = pInfEngine->computePartition(pFeatureGen, X, m, -1, true);
	partition = pInfEngine->computePartition(pFeatureGen, X, m, -1, false);
	
	// return log(Z(h|x)) - log(Z(h*|x)) 
	return partition - partitionMasked;
}


void EvaluatorLDCRF::computeStateLabels(DataSequence* X, Model* m, iVector* vecStateLabels, dMatrix * probabilities)
{
	if(!pInfEngine || !pFeatureGen){
		throw HcrfBadPointer("EvaluatorLDCRF::computeStateLabels");
	}
	Beliefs bel;
	pInfEngine->computeBeliefs(bel, pFeatureGen, X, m, false);

	int nbNodes = (int)bel.belStates.size();
	int nbLabels = m->getNumberOfStateLabels();
	int nbStates = 0;
	if(nbNodes > 0)
		nbStates = bel.belStates[0].getLength();

	vecStateLabels->create(nbNodes);
	if(probabilities)
		probabilities->create(nbNodes,nbLabels);

	dVector sumBeliefsPerLabel(nbLabels);

	for(int n = 0; n<nbNodes; n++) 
	{
		sumBeliefsPerLabel.set(0);
		// Sum beliefs
		for (int s = 0; s<nbStates; s++) 
		{
			sumBeliefsPerLabel[m->getLabelPerState()[s]] += bel.belStates[n][s];
		}
		// find max value
		vecStateLabels->setValue(n, 0);
		double MaxBel = sumBeliefsPerLabel[0];
		if(probabilities)
			probabilities->setValue(0,n,sumBeliefsPerLabel[0]);
		for (int cur_label = 1; cur_label < nbLabels; cur_label++) 
		{
			if(probabilities)
				probabilities->setValue(cur_label, n, sumBeliefsPerLabel[cur_label]);
			if(MaxBel < sumBeliefsPerLabel[cur_label]) {
				vecStateLabels->setValue(n, cur_label);
				MaxBel = sumBeliefsPerLabel[cur_label];
			}
		}
	}
}
