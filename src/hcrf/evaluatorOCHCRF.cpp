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

EvaluatorOCHCRF::EvaluatorOCHCRF(): Evaluator() {}

EvaluatorOCHCRF::EvaluatorOCHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen)
: Evaluator(infEngine, featureGen) {}

EvaluatorOCHCRF::~EvaluatorOCHCRF() {}

double EvaluatorOCHCRF::computeError(DataSequence* X, Model* m)
{
  throw HcrfBadPointer("In EvaluatorOCHCRF::computeError()");
}

int EvaluatorOCHCRF::computeSequenceLabel(DataSequence* X, Model* m, dVector* prob, dMatrix* hprob)
{
  if(!pInfEngine || !pFeatureGen)
    throw HcrfBadPointer("In EvaluatorOCHCRF::computeSequenceLabel()");
  
  int nbSeqLabels, nbNodes, nbHiddenStates, t, h, y, y_pos, y_neg;
  double f_val, lZx;
  
  nbSeqLabels = m->getNumberOfSequenceLabels();
  nbNodes = X->length();
  nbHiddenStates = m->getNumberOfStates();
  
  if( prob ) prob->create(nbSeqLabels);
  
  std::vector<Beliefs> condBeliefs(nbSeqLabels);
  dVector Partition(nbSeqLabels);
  for(y=0; y<nbSeqLabels; y++) {
    pInfEngine->computeBeliefs(condBeliefs[y], pFeatureGen, X, m, true, y);
    Partition[y] = condBeliefs[y].partition;
  }
  lZx = Partition.logSumExp();
  
  y_pos = 0; y_neg = 1;
  prob->setValue(y_pos,exp(Partition[y_pos]-lZx));
  prob->setValue(y_neg,exp(Partition[y_neg]-lZx));
  f_val = (prob->getValue(y_pos)-prob->getValue(y_neg)) + 1.0e-6; // for numerical stability
  
  y = ((f_val-m->getRho())>=0) ? 0 : 1;
  
  if( hprob ) hprob->create(nbNodes,nbHiddenStates);
  for(t=0; t<nbNodes; t++)
    for(h=0; h<nbHiddenStates; h++)
      hprob->setValue(h,t,condBeliefs[y].belStates[t][h]);
  
  return y;
}
