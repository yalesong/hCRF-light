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

#include <assert.h>

#include "hcrf/evaluator.h"

using namespace std;

/////////////////////////////////////////////////////////////////////
// Evaluator MVHCRF Class
/////////////////////////////////////////////////////////////////////

// *
// Constructor and Destructor
// *

EvaluatorMVHCRF::EvaluatorMVHCRF()
: Evaluator()
{}

EvaluatorMVHCRF::EvaluatorMVHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen)
: Evaluator(infEngine, featureGen)
{}

EvaluatorMVHCRF::~EvaluatorMVHCRF()
{}

// *
// Public Methods
// *
double EvaluatorMVHCRF::computeError(DataSequence* X, Model* m)
{
  if(!pInfEngine || !pFeatureGen)
    throw HcrfBadPointer("In EvaluatorHCRF::computeError()");
  
  double groundTruthLabel = X->getSequenceLabel();
  double groundTruthPartition = -DBL_MAX;
  double maxPartition = -DBL_MAX;
  
  int nbSeqLabels = m->getNumberOfSequenceLabels();
  dVector Partition(nbSeqLabels);
  
  // For each class label, compute the partition of the data sequence, and add up all these partitions
  for(int seqLabel=0; seqLabel<nbSeqLabels; seqLabel++)
  {
    Partition[seqLabel] = pInfEngine->computePartition(pFeatureGen,X,m,seqLabel,false);
    if( seqLabel==groundTruthLabel )
      groundTruthPartition = Partition[seqLabel];
    if( Partition[seqLabel] > maxPartition )
      maxPartition = Partition[seqLabel];
  }
  
  if( m->isMaxMargin() )
    return maxPartition - groundTruthPartition;
  else
    return Partition.logSumExp() - groundTruthPartition;
}


int EvaluatorMVHCRF::computeSequenceLabel(DataSequence* X, Model* m, dVector* prob, dMatrix* hprob)
{
  if(!pInfEngine || !pFeatureGen)
    throw HcrfBadPointer("In EvaluatorHCRF::computeSequenceLabel()");
  
  int y, y_star, nbSeqLabels;
  double partition, bestScore, lZx;
  
  nbSeqLabels = m->getNumberOfSequenceLabels();
  prob->create(nbSeqLabels);
  
  y_star = -1;
  bestScore = -DBL_MAX;
  for(y=0; y<nbSeqLabels; y++)
  {
    partition = pInfEngine->computePartition(pFeatureGen,X,m,y);
    prob->setValue(y,partition);
    if( partition > bestScore) {
      y_star = y;
      bestScore = partition;
    }
  }
  
  lZx = prob->logSumExp();
  for(y=0; y<nbSeqLabels; y++)
    prob->setValue(y, exp(prob->getValue(y)-lZx));
  
  return y_star;
}

