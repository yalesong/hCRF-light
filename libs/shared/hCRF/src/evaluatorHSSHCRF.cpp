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

#define CLASS_CONDITIONAL 0

/////////////////////////////////////////////////////////////////////
// Evaluator HCRF Class
/////////////////////////////////////////////////////////////////////

// *
// Constructor and Destructor
// *

EvaluatorHSSHCRF::EvaluatorHSSHCRF() 
: Evaluator() {}

EvaluatorHSSHCRF::EvaluatorHSSHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen) 
: Evaluator(infEngine, featureGen) {}

EvaluatorHSSHCRF::~EvaluatorHSSHCRF() {}

// *
// Public Methods
// *

//computes OVERALL error of the datasequence
double EvaluatorHSSHCRF::computeError(DataSequence* X, Model* m)
{ 
	throw HcrfNotImplemented("EvaluatorHSSHCRF::computeError()");
}

int EvaluatorHSSHCRF::computeSequenceLabel(DataSequence* X, Model* m, dVector* prob, dMatrix* hprob)
{
	if(!pInfEngine || !pFeatureGen)
		throw HcrfBadPointer("In EvaluatorHCRF::computeSequenceLabel");

	int y, y_star, dimY, num_ccs, prev_num_ccs, layer;
	int *labels = 0;
	double Z, maxZ;
	segment segmenter;
  
	dimY = m->getNumberOfSequenceLabels();
	prev_num_ccs = num_ccs = X->length();
	if(prob) prob->create(dimY);
	prob->set(0);

	std::vector<Beliefs> condBeliefs(dimY); 
	dVector partition(m->getNumberOfSequenceLabels());

	std::vector<std::vector<int> > groupLabelSet;
	for(int j=0; j<X->length(); j++) {
		std::vector<int> groupLabel;
		groupLabel.push_back(j);
		groupLabelSet.push_back(groupLabel);
	}
	X->getDeepSeqGroupLabels()->push_back(groupLabelSet);

	// Compute posterior probability p(y|x)
	for(layer=0;layer<m->getMaxFeatureLayer();layer++) 
	{ 
		m->setCurrentFeatureLayer(layer);

		for( y=0; y<dimY; y++ ) {
			pInfEngine->computeBeliefs(condBeliefs[y],pFeatureGen,X,m,true,y,false);
			partition[y] = condBeliefs[y].partition;
		}
		Z = partition.logSumExp();
		for( y=0; y<dimY; y++ )
			prob->addValue(y,partition[y]-Z);

		// Summarize for the next iteration
		std::vector<std::vector<int> > super_x = X->getDeepSeqGroupLabels()->at(layer);
		labels = new int[(int)super_x.size()];
		segmenter.segment_sequence(condBeliefs,m->getSegmentConst(),&num_ccs,&labels);

		if( prev_num_ccs==num_ccs || num_ccs==1 ) break;
		prev_num_ccs = num_ccs;

		// Record group labels
		std::vector<std::vector<int> > groupLabelSet;
		for(int t=0; t<num_ccs; t++) {
			std::vector<int> groupLabel;
			groupLabelSet.push_back(groupLabel);
		}
		for(int t=0; t<(int)super_x.size(); t++) 
			for(int k=0; k<(int)super_x[t].size(); k++)
				groupLabelSet.at(labels[t]).push_back(super_x[t].at(k));
		X->getDeepSeqGroupLabels()->push_back(groupLabelSet);
		
		delete[] labels; labels=0;
	}

	Z = prob->logSumExp();
	for( y=0; y<dimY; y++ )
		prob->setValue(y,exp(prob->getValue(y)-Z));

	// Find y_star
	maxZ = -DBL_MAX;
	for( y=0; y<dimY; y++ ) {
		Z = prob->getValue(y);  
		if( maxZ<Z ) { maxZ = Z; y_star = y; }
	}

	delete[] labels; labels=0;
	return y_star; 
} 