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

#include "hcrf/inferenceengine.h"

double InferenceEngineFB::computePartition(FeatureGenerator* fGen,
                                           DataSequence* X, Model* model, int seqLabel,bool bUseStatePerNodes)
{
  Beliefs bel;
  computeBeliefsLog(bel, fGen,X, model, true,seqLabel, bUseStatePerNodes);
  return bel.partition;
}

// Inference Functions
void InferenceEngineFB::computeBeliefs(Beliefs& bel, FeatureGenerator* fGen,
                                       DataSequence* X, Model* model, int bComputePartition, int seqLabel, bool bUseStatePerNodes) {
  computeBeliefsLog(bel, fGen,X, model, bComputePartition, seqLabel, bUseStatePerNodes);
}


void InferenceEngineFB::computeBeliefsLinear(Beliefs& bel, FeatureGenerator* fGen,
                                             DataSequence* X, Model* model, int, int seqLabel, bool bUseStatePerNodes)
{
  if(model->getAdjacencyMatType()!=CHAIN)
    throw HcrfBadModel("InferenceEngineFB need a model based on a Chain");
  
  int i, NNODES, NEDGES, NSTATES;
  
  NNODES = X->length();
  NEDGES = NNODES-1;
  NSTATES = model->getNumberOfStates();
  
  bel.belStates.resize(NNODES);
  for(i=0; i<NNODES; i++) {
    bel.belStates[i].create(NSTATES);
    bel.belStates[i].set(0);
  }
  bel.belEdges.resize(NEDGES);
  for(int i=0;i<NEDGES;i++)
  {
    bel.belEdges[i].create(NSTATES,NSTATES);
    bel.belEdges[i].set(0);
  }
  
  dMatrix Mi_YY (NSTATES,NSTATES);
  dVector Ri_Y (NSTATES);
  dVector alpha_Y(NSTATES);
  dVector newAlpha_Y(NSTATES);
  dVector tmp_Y(NSTATES);
  
  alpha_Y.set(1);
  bel.belStates[NNODES-1].set(1.0);
  
  for(i=NNODES-1; i>0; i--)
  {
    // compute the Mi matrix
    computeLogMi(fGen,model,X,i,seqLabel,Mi_YY,Ri_Y,true,bUseStatePerNodes);
    
    tmp_Y.set(bel.belStates[i]);
    tmp_Y.eltMpy(Ri_Y);
    bel.belStates[i-1].multiply(Mi_YY,tmp_Y);
  }
  for(i=0; i<NNODES; i++)
  {
    // compute the Mi matrix
    computeLogMi(fGen,model,X,i,seqLabel,Mi_YY,Ri_Y,true,bUseStatePerNodes);
    
    if (i > 0) {
      tmp_Y.set(alpha_Y);
      Mi_YY.transpose();
      newAlpha_Y.multiply(Mi_YY,tmp_Y);
      newAlpha_Y.eltMpy(Ri_Y);
    }
    else
    {
      newAlpha_Y.set(Ri_Y);
    }
    
    if (i > 0)
    {
      tmp_Y.set(Ri_Y);
      tmp_Y.eltMpy(bel.belStates[i]);
      tmp_Y.transpose();
      bel.belEdges[i-1].multiply(alpha_Y,tmp_Y);
      Mi_YY.transpose();
      bel.belEdges[i-1].eltMpy(Mi_YY);
    }
    
    bel.belStates[i].eltMpy(newAlpha_Y);
    alpha_Y.set(newAlpha_Y);
  }
  
  double Zx = alpha_Y.sum();
  for(i=0; i<NNODES; i++)
    bel.belStates[i].multiply(1.0/Zx);
  for(i=0; i<NEDGES; i++)
    bel.belEdges[i].multiply(1.0/Zx);
  bel.partition = log(Zx);
}

void InferenceEngineFB::computeBeliefsLog(Beliefs& bel, FeatureGenerator* fGen,
                                          DataSequence* X, Model* model, int, int seqLabel, bool bUseStatePerNodes)
{
  if(model->getAdjacencyMatType()!=CHAIN)
    throw HcrfBadModel("InferenceEngineFB need a model based on a Chain");
  
  int xi, NNODES, NSTATES, NEDGES;
  
  NNODES  = (model->getMaxFeatureLayer()>1)
		? (int) X->getDeepSeqGroupLabels()->at(model->getCurrentFeatureLayer()).size()
		: X->length();
  NSTATES = model->getNumberOfStates();
  NEDGES  = NNODES-1;
  
  bel.belStates.resize(NNODES);
  for( xi=0; xi<NNODES; xi++ )
    bel.belStates[xi].create(NSTATES);
  
  bel.belEdges.resize(NEDGES);
  for( xi=0; xi<NEDGES; xi++)
    bel.belEdges[xi].create(NSTATES,NSTATES, 0);
  
  dVector Ri_Y(NSTATES);
  dVector alpha_Y(NSTATES);
  dVector newAlpha_Y(NSTATES);
  dVector tmp_Y(NSTATES);
  dMatrix Mi_YY(NSTATES,NSTATES);
  
  // compute beta values in a backward scan.
  // also scale beta-values to 1 to avoid numerical problems.
  bel.belStates[NNODES-1].set(0);
  for( xi=NNODES-1; xi>0; xi--)
  {
    // compute the Mi matrix
    computeLogMi(fGen, model, X, xi, seqLabel, Mi_YY, Ri_Y, false, bUseStatePerNodes);
    tmp_Y.set(bel.belStates[xi]);
    tmp_Y.add(Ri_Y);
    LogMultiply(Mi_YY,tmp_Y,bel.belStates[xi-1]);
  }
  
  // Compute Alpha values
  alpha_Y.set(0);
  {
    xi = 0;
    computeLogMi(fGen,model, X, xi, seqLabel, Mi_YY, Ri_Y, false, bUseStatePerNodes);
    newAlpha_Y.set(Ri_Y);
    bel.belStates[xi].add(newAlpha_Y);
    alpha_Y.set(newAlpha_Y);
  }
  for( xi=1; xi<NNODES; xi++)
  {
    computeLogMi(fGen,model, X, xi, seqLabel, Mi_YY, Ri_Y, false, bUseStatePerNodes);
    tmp_Y.set(alpha_Y);
    Mi_YY.transpose();
    LogMultiply(Mi_YY, tmp_Y, newAlpha_Y);
    newAlpha_Y.add(Ri_Y);
    
    tmp_Y.set(Ri_Y);
    tmp_Y.add(bel.belStates[xi]);
    Mi_YY.transpose();
    bel.belEdges[xi-1].set(Mi_YY);
    for(int yprev = 0; yprev < NSTATES; yprev++)
      for(int yp = 0; yp < NSTATES; yp++)
        bel.belEdges[xi-1](yprev,yp) += tmp_Y[yp] + alpha_Y[yprev];
    
    bel.belStates[xi].add(newAlpha_Y);
    alpha_Y.set(newAlpha_Y);
  } 
  
  double lZx = alpha_Y.logSumExp();
  for( xi=0; xi<NNODES; xi++ )
  {
    bel.belStates[xi].add(-lZx);
    bel.belStates[xi].eltExp();
  }
  for( xi=0; xi<NEDGES; xi++ )
  {
    bel.belEdges[xi].add(-lZx);
    bel.belEdges[xi].eltExp();
  }
  bel.partition = lZx;
}	







