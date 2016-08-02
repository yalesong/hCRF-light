/*
 hCRF-light Library 3.0 (full version http://hcrf.sf.net)
 Copyright (C) Yale Song (yalesong@mit.edu)
 
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


#include "hcrf/gradient.h"
#include "hcrf/features.h"

using namespace std;

GradientOCHCRF::GradientOCHCRF(InferenceEngine* ie, FeatureGenerator* fg): Gradient(ie, fg){}

double GradientOCHCRF::computeGradient(dVector& vecGradient, Model* m, DataSequence* X)
{
  return mle(vecGradient,m,X);
}

// MLE-like
double GradientOCHCRF::mle(dVector& vecGradient, Model* m, DataSequence* X)
{
  int k, xi, xj, y, gidx, y_pos, y_neg, seqLabel, nbSeqLabels, nbFeatures, nbGates;
  double val, loss, lZx;
  
  seqLabel    = X->getSequenceLabel();
  nbSeqLabels = m->getNumberOfSequenceLabels();
  nbFeatures  = pFeatureGen->getNumberOfFeatures();
  nbGates     = m->getNbGates();
  
  std::vector<Beliefs> condBeliefs(nbSeqLabels);
  dVector Partition(nbSeqLabels);
  dVector prob(nbSeqLabels);
  
  dMatrix condEValues(nbFeatures, nbSeqLabels);
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Step 1 : Run Inference in each network to compute marginals conditioned on Y
  for(y=0; y<nbSeqLabels; y++) {
    pInfEngine->computeBeliefs(condBeliefs[y], pFeatureGen, X, m, true, y);
    Partition[y] = condBeliefs[y].partition;
  }
  lZx = Partition.logSumExp();
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Step 2 : Compute expected values for node/edge features conditioned on Y
  feature* f;
  featureVector vecFeatures,myPreGateVecFeat;
  dVector* W = m->getWeights();
  
  // Neural gate features
  dVector gateProbWeightSum;
  GateNodeFeatures* gateF;
  
  // Loop over nodes to compute features and update the gradient
  if( nbGates==0 ) {
    for(y=0; y<nbSeqLabels; y++) {
      for(xi=0; xi<X->length(); xi++) {
        pFeatureGen->getFeatures(vecFeatures,X,m,xi,-1,y);
        f = vecFeatures.getPtr();
        for(k=0; k<vecFeatures.size(); k++, f++) {
          // p(h^v_t=a|x,y) * f_k(v,t,a,x,y)
          val = condBeliefs[y].belStates[xi][f->nodeState] * f->value;
          condEValues.addValue(y, f->globalId, val);
        }
      }
    }
  }
  else {
    gateProbWeightSum.resize(1,nbGates);
    gateF = (GateNodeFeatures*) pFeatureGen->getFeatureById(GATE_NODE_FEATURE_ID);
    
    for(y=0; y<nbSeqLabels; y++) {
      for(xi=0; xi<X->length(); xi++) {
        pFeatureGen->getFeatures(vecFeatures,X,m,xi,-1,y);
        f = vecFeatures.getPtr();
        for(k=0; k<vecFeatures.size(); k++, f++) {
          //printf("[s] typeID=%d,gid=%d,id=%d,idx=[%d,%d],state=[%d,%d],view=[%d,%d],y=%d,val=%f\n",f->featureTypeId,f->globalId,f->id,f->prevNodeIndex,f->nodeIndex,f->prevNodeState,f->nodeState,f->prevNodeView,f->nodeView,f->sequenceLabel,f->value);
          // p(h^v_t=a|x,y) * f_k(v,t,a,x,y)
          val = condBeliefs[y].belStates[xi][f->nodeState] * f->value;
          condEValues.addValue(y, f->globalId, val);
          if( f->featureTypeId == GATE_NODE_FEATURE_ID ) {
            val = W->getValue(f->globalId) * ((1.0-f->value)*f->value);
            gidx = f->prevNodeState; // quick-and-dirty solution
            gateProbWeightSum.addValue(gidx, condBeliefs[y].belStates[xi][f->nodeState]*val);
          }
        }
        gateF->getPreGateFeatures(myPreGateVecFeat,X,m,xi,-1,y);
        f = myPreGateVecFeat.getPtr();
        for(k=0; k<myPreGateVecFeat.size(); k++, f++) {
          //printf("[s] typeID=%d,gid=%d,id=%d,idx=[%d,%d],state=[%d,%d],view=[%d,%d],y=%d,val=%f\n",f->featureTypeId,f->globalId,f->id,f->prevNodeIndex,f->nodeIndex,f->prevNodeState,f->nodeState,f->prevNodeView,f->nodeView,f->sequenceLabel,f->value);
          gidx = f->prevNodeState;
          val = f->value*gateProbWeightSum[gidx];
          condEValues.addValue(y, f->globalId, val);
        }
      }
    }
  }
  
  // Loop over edges to compute features and update the gradient
  for(y=0; y<nbSeqLabels; y++) {
    for(xi=0; xi<X->length()-1; xi++) {
      xj = xi+1;
      pFeatureGen->getFeatures(vecFeatures,X,m,xj,xi,y);
      f = vecFeatures.getPtr();
      for(k=0; k<vecFeatures.size(); k++, f++) {
        //printf("[p] typeID=%d,gid=%d,id=%d,idx=[%d,%d],state=[%d,%d],view=[%d,%d],y=%d,val=%f\n",f->featureTypeId,f->globalId,f->id,f->prevNodeIndex,f->nodeIndex,f->prevNodeState,f->nodeState,f->prevNodeView,f->nodeView,f->sequenceLabel,f->value);
        // p(h^vi_ti=a,h^vj_tj=b|x,y) * f_k(vi,ti,vj,tj,x,y)
        val = condBeliefs[y].belEdges[xi](f->prevNodeState,f->nodeState) * f->value;
        condEValues.addValue(y, f->globalId, val);
      }
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Step 3: Compute loss and gradient
  dVector rowJ(nbFeatures), JointEValues(nbFeatures);
  
  if( vecGradient.getLength() != nbFeatures )
    vecGradient.create(nbFeatures);
  
  y_pos = 0; y_neg = 1;
  prob[y_pos] = exp(Partition[y_pos]-lZx);
  prob[y_neg] = exp(Partition[y_neg]-lZx);
  
  // Compute loss (log scale)
  loss = MAX(0,log((1+m->getRho())/(1-m->getRho())) - (Partition[y_pos]-Partition[y_neg]));
  
  condEValues.getRow(y_pos,rowJ);
  rowJ.multiply( prob[y_pos] );
  JointEValues.add( rowJ );
  
  condEValues.getRow(y_neg,rowJ);
  rowJ.multiply( prob[y_neg] );
  JointEValues.add( rowJ );
  
  condEValues.getRow(y_pos,rowJ);
  rowJ.subtract(JointEValues);
  
  vecGradient.add(rowJ);
  
  return loss;
}


// MaxMargin-like
double GradientOCHCRF::max_margin(dVector& vecGradient, Model* m, DataSequence* X)
{
  int k, xi, xj, y, y_pos, y_neg, nbSeqLabels, nbFeatures;
  double val, loss, lZx;
  
  y_pos = X->getSequenceLabel();
  nbSeqLabels = m->getNumberOfSequenceLabels();
  nbFeatures = pFeatureGen->getNumberOfFeatures();
  
  std::vector<Beliefs> condBeliefs(nbSeqLabels);
  dMatrix condEValues(nbFeatures, nbSeqLabels);
  dVector prob(nbSeqLabels);
  dVector Partition(nbSeqLabels);
  
  // Assume all training samples are from the positive class (label=0)
  y_pos = 0; y_neg = 1;
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Step 1 : Run Inference in each network to compute marginals conditioned on Y
  for(y=0; y<nbSeqLabels; y++) {
    pInfEngine->computeBeliefs(condBeliefs[y], pFeatureGen, X, m, true, y);
    Partition[y] = condBeliefs[y].partition;
  }
  lZx = Partition.logSumExp();
  prob[y_pos] = exp(Partition[y_pos]-lZx);
  prob[y_neg] = exp(Partition[y_neg]-lZx);
  
  // Compute loss (log scale)
  loss = MAX(0,log((1+m->getRho())/(1-m->getRho())) - (Partition[y_pos]-Partition[y_neg]));
  
  // Viterbi decoding
  iVector h_star, h_true;
  dVector ph_star, ph_true;
  viterbiDecoding(condBeliefs[y_pos],h_true,ph_true); // true
  viterbiDecoding(condBeliefs[y_neg],h_star,ph_star); // star
  
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Step 2 : Compute expected values for node/edge features conditioned on Y
  feature* f;
  featureVector vecFeatures;
  
  bool noBel = true;
  
  // Loop over nodes to compute features and update the gradient
  for(xi=0; xi<X->length(); xi++) {
    pFeatureGen->getFeatures(vecFeatures,X,m,xi,-1,y_pos);
    f = vecFeatures.getPtr();
    for(k=0; k<vecFeatures.size(); k++, f++) {
      // p(h^v_t=a|x,y) * f_k(v,t,a,x,y)
      if( f->nodeState==h_true[xi] ) {
        val = (noBel) ? f->value : condBeliefs[y_pos].belStates[xi][f->nodeState] * f->value;
        condEValues.addValue(y_pos, f->globalId, val);
      }
    }
  }
  
  for(xi=0; xi<X->length(); xi++) {
    pFeatureGen->getFeatures(vecFeatures,X,m,xi,-1,y_neg);
    f = vecFeatures.getPtr();
    for(k=0; k<vecFeatures.size(); k++, f++) {
      // p(h^v_t=a|x,y) * f_k(v,t,a,x,y)
      if( f->nodeState==h_star[xi] ) {
        val = (noBel) ? f->value : condBeliefs[y_neg].belStates[xi][f->nodeState] * f->value;
        condEValues.addValue(y_neg, f->globalId, val);
      }
    }
  }
  
  // Loop over edges to compute features and update the gradient
  for(xi=0; xi<X->length()-1; xi++) {
    xj = xi+1;
    pFeatureGen->getFeatures(vecFeatures,X,m,xj,xi,y_pos);
    f = vecFeatures.getPtr();
    for(k=0; k<vecFeatures.size(); k++, f++) {
      // p(h^vi_ti=a,h^vj_tj=b|x,y) * f_k(vi,ti,vj,tj,x,y)
      if( f->prevNodeState==h_true[xi] && f->nodeState==h_true[xj] ) {
        val = (noBel) ? f->value : condBeliefs[y_pos].belEdges[xi](f->prevNodeState,f->nodeState) * f->value;
        condEValues.addValue(y_pos, f->globalId, val);
      }
    }
  }
  
  for(xi=0; xi<X->length()-1; xi++) {
    xj = xi+1;
    pFeatureGen->getFeatures(vecFeatures,X,m,xj,xi,y_neg);
    f = vecFeatures.getPtr();
    for(k=0; k<vecFeatures.size(); k++, f++) {
      // p(h^vi_ti=a,h^vj_tj=b|x,y) * f_k(vi,ti,vj,tj,x,y)
      if( f->prevNodeState==h_star[xi] && f->nodeState==h_star[xj] ) {
        val = (noBel) ? f->value : condBeliefs[y_neg].belEdges[xi](f->prevNodeState,f->nodeState) * f->value;
        condEValues.addValue(y_neg, f->globalId, val);
      }
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Step 3: Compute gradient
  if( vecGradient.getLength() != nbFeatures )
    vecGradient.create(nbFeatures);
  
  dVector rowJ(nbFeatures);  // expected value conditioned on Y
  
  condEValues.getRow(y_neg,rowJ);
  rowJ.multiply( prob[y_pos] );
  vecGradient.add(rowJ);
  
  condEValues.getRow(y_pos,rowJ);
  rowJ.multiply( prob[y_neg] );
  rowJ.negate();
  vecGradient.add(rowJ);
  
  return loss;
}
