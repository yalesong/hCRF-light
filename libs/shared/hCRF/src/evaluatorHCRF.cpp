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

EvaluatorHCRF::EvaluatorHCRF() 
:Evaluator()
{}

EvaluatorHCRF::EvaluatorHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen) 
:Evaluator(infEngine, featureGen)
{}

EvaluatorHCRF::~EvaluatorHCRF()
{}

// *
// Public Methods
// *

//computes OVERALL error of the datasequence
double EvaluatorHCRF::computeError(DataSequence* X, Model* m)
{
	if(!pInfEngine || !pFeatureGen){
		throw HcrfBadPointer("In EvaluatorHCRF::computeError");
	}

	int y, nbSeqLabels = m->getNumberOfSequenceLabels(); 
	double score;
	dVector Partition(nbSeqLabels);

	for(y=0; y<nbSeqLabels; y++) 
		Partition[y] = pInfEngine->computePartition(pFeatureGen,X,m,y,false);

	if( m->isMaxMargin() ) 
	{
		//return max_y' log(F(y'|x)) - log(F(y|x))
		int max_y=-1; double max_val=-DBL_MAX;
		for(y=0; y<nbSeqLabels; y++) {
			if( Partition[y] > max_val ) {
				max_y=y; max_val = Partition[y];
			}
		}
		score = max_val - Partition[X->getSequenceLabel()];
	}
	else 
	{
		//return log(Sum_y' p(y'|x)) - log(p(y|x)) 
		score = Partition.logSumExp() - Partition[X->getSequenceLabel()];
	}

	return score;
}


int EvaluatorHCRF::computeSequenceLabel(DataSequence* X, Model* m, dVector* prob, dMatrix* hprob)
{
	if(!pInfEngine || !pFeatureGen){
		throw HcrfBadPointer("In EvaluatorHCRF::computeSequenceLabel");
	}

	int t, h, y, y_star, nbSeqLabels, nbNodes, nbHiddenStates;
	double bestScore, lZx;

	nbSeqLabels = m->getNumberOfSequenceLabels();
	nbNodes = X->length();
	nbHiddenStates = m->getNumberOfStates();

	if( prob ) prob->create(nbSeqLabels);

	bestScore = -DBL_MAX;
	std::vector<Beliefs> condBeliefs(nbSeqLabels);
	dVector Partition(nbSeqLabels);
	for(y=0; y<nbSeqLabels; y++)
	{
		pInfEngine->computeBeliefs(condBeliefs[y], pFeatureGen, X, m, true, y);
		Partition[y] = condBeliefs[y].partition;
		
		if( Partition[y] > bestScore ){
			y_star = y;
			bestScore = Partition[y];
		}
	}
	lZx = Partition.logSumExp();

	for(y=0; y<nbSeqLabels; y++) 
		prob->setValue(y, exp(Partition[y]-lZx));

	if( hprob ) hprob->create(nbNodes,nbHiddenStates);
	for(t=0; t<nbNodes; t++)
		for(h=0; h<nbHiddenStates; h++)
			hprob->setValue(h,t,condBeliefs[y_star].belStates[t][h]);

	return y_star;
}

