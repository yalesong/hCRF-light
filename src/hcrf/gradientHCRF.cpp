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


#include "hcrf/gradient.h"
#include "hcrf/features.h"

GradientHCRF::GradientHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen)
: Gradient(infEngine, featureGen)
{}

double GradientHCRF::computeGradient(dVector& vecGradient, Model* m, DataSequence* X)
{
  double fval = (m->isMaxMargin())
  ? computeGradientMaxMargin(vecGradient,m,X)
  : computeGradientMLE(vecGradient,m,X);
  return fval;
}

double GradientHCRF::computeGradientMLE(dVector& vecGradient, Model* m, DataSequence* X)
{
  int k, xi, y, nbSeqLabels, nbFeatures;
  int gidx, nbGates; // Neural layer params
  double val, f_val, lZx;
  
  nbSeqLabels  = m->getNumberOfSequenceLabels();
  nbFeatures   = pFeatureGen->getNumberOfFeatures();
  nbGates      = m->getNbGates();
  
  std::vector<Beliefs> condBeliefs(nbSeqLabels);
  dMatrix condEValues(nbFeatures, nbSeqLabels);
  dVector Partition(nbSeqLabels);
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Step 1 : Run Inference in each network to compute marginals conditioned on Y
  for(y=0; y<nbSeqLabels; y++) {
    pInfEngine->computeBeliefs(condBeliefs[y], pFeatureGen, X, m, true, y);
    Partition[y] = condBeliefs[y].partition;
  }
  lZx = Partition.logSumExp();
  f_val = lZx - Partition[X->getSequenceLabel()];
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Step 2 : Compute expected values for node/edge features conditioned on Y
  feature* f;
  featureVector vecFeatures;
  
  // Neural gate features
  dVector* W = m->getWeights();
  dVector gateProbWeightSum;
  GateNodeFeatures* gateF;
  
  // Compute gradient for singleton features
  if( nbGates==0 )
  {
    for(y=0; y<nbSeqLabels; y++) {
      // Loop over nodes to compute features and update the gradient
      for(xi=0; xi<X->length(); xi++) {
        pFeatureGen->getFeatures(vecFeatures,X,m,xi,-1,y);
        f = vecFeatures.getPtr();
        for(k=0; k<vecFeatures.size(); k++, f++)
        {
          //{printf("[s] typeID=%d,gid=%d,id=%d,idx=[%d,%d],state=[%d,%d],view=[%d,%d],y=%d,val=%f\n",f->featureTypeId,f->globalId,f->id,f->prevNodeIndex,f->nodeIndex,f->prevNodeState,f->nodeState,f->prevNodeView,f->nodeView,f->sequenceLabel,f->value);}
          // p(h^v_t=a|x,y) * f_k(v,t,a,x,y)
          val = condBeliefs[y].belStates[xi][f->nodeState] * f->value;
          condEValues.addValue(y, f->globalId, val);
        }
      }
    }
  }
  else
  {
    gateProbWeightSum.resize(1,nbGates);
    gateF = (GateNodeFeatures*) pFeatureGen->getFeatureById(GATE_NODE_FEATURE_ID);
    
    for(y=0; y<nbSeqLabels; y++) {
      for(xi=0; xi<X->length(); xi++) {
        gateProbWeightSum.set(0); // don't forget to reset this to zero
        pFeatureGen->getFeatures(vecFeatures,X,m,xi,-1,y);
        f = vecFeatures.getPtr();
        for(k=0; k<vecFeatures.size(); k++, f++)
        {
          //{printf("[s] typeID=%d,gid=%d,id=%d,idx=[%d,%d],state=[%d,%d],view=[%d,%d],y=%d,val=%f\n",f->featureTypeId,f->globalId,f->id,f->prevNodeIndex,f->nodeIndex,f->prevNodeState,f->nodeState,f->prevNodeView,f->nodeView,f->sequenceLabel,f->value);}
          // p(h^v_t=a|x,y) * f_k(v,t,a,x,y)
          val = condBeliefs[y].belStates[xi][f->nodeState] * f->value;
          condEValues.addValue(y, f->globalId, val);
          if( f->featureTypeId == GATE_NODE_FEATURE_ID ) {
            val = W->getValue(f->globalId) * ((1.0-f->value)*f->value);
            gidx = f->prevNodeState; // gate index (quick-and-dirty solution)
            gateProbWeightSum.addValue(gidx, condBeliefs[y].belStates[xi][f->nodeState]*val);
          }
        }
        
        gateF->getPreGateFeatures(vecFeatures,X,m,xi,-1,y);
        f = vecFeatures.getPtr();
        for(k=0; k<vecFeatures.size(); k++, f++)
        {
          //{printf("[s] typeID=%d,gid=%d,id=%d,idx=[%d,%d],state=[%d,%d],view=[%d,%d],y=%d,val=%f\n",f->featureTypeId,f->globalId,f->id,f->prevNodeIndex,f->nodeIndex,f->prevNodeState,f->nodeState,f->prevNodeView,f->nodeView,f->sequenceLabel,f->value);}
          gidx = f->prevNodeState; // gate index
          val = f->value*gateProbWeightSum[gidx];
          condEValues.addValue(y, f->globalId, val);
        }
      }
    }
  }
  
  // Compute gradient for pairwise features
  for(y=0; y<nbSeqLabels; y++)
  {
    // Loop over edges to compute features and update the gradient
    for(xi=1; xi<X->length(); xi++) {
      pFeatureGen->getFeatures(vecFeatures,X,m,xi,xi-1,y);
      f = vecFeatures.getPtr();
      for(k=0; k<vecFeatures.size(); k++, f++)
      {
        //{printf("[p] typeID=%d,gid=%d,id=%d,idx=[%d,%d],state=[%d,%d],view=[%d,%d],y=%d,val=%f\n",f->featureTypeId,f->globalId,f->id,f->prevNodeIndex,f->nodeIndex,f->prevNodeState,f->nodeState,f->prevNodeView,f->nodeView,f->sequenceLabel,f->value);}
        // p(h^vi_ti=a,h^vj_tj=b|x,y) * f_k(vi,ti,vj,tj,x,y)
        val = condBeliefs[y].belEdges[xi-1](f->prevNodeState,f->nodeState) * f->value;
        condEValues.addValue(y, f->globalId, val);
      }
    }
  }
  
  // Step 3: Compute Joint Expected Values
  dVector JointEValues(nbFeatures), rowJ(nbFeatures);
  for(y=0; y<nbSeqLabels; y++) {
    condEValues.getRow(y, rowJ);
    rowJ.multiply( exp(Partition[y]-lZx) );
    JointEValues.add(rowJ);
  }
  
  // Step 4 Compute Gradient as Exi[i,*,*] -Exi[*,*,*], that is difference
  // between expected values conditioned on Sequence Labels and Joint expected values
  condEValues.getRow(X->getSequenceLabel(), rowJ); // rowJ=Expected value conditioned on Sequence label Y
  JointEValues.negate();
  rowJ.add(JointEValues);
  
  vecGradient.add(rowJ);
  return f_val;
}


double GradientHCRF::computeGradientMaxMargin(dVector& vecGradient, Model* m, DataSequence* X)
{
  int k, xi, xj, y, y_true, y_star, nbSeqLabels, nbFeatures;
  double val, max_val, loss, lZx, phi_true=0, phi_star=0;
  
  y_true = X->getSequenceLabel();
  nbSeqLabels = m->getNumberOfSequenceLabels();
  nbFeatures = pFeatureGen->getNumberOfFeatures();
  
  std::vector<Beliefs> condBeliefs(nbSeqLabels);
  dMatrix condEValues(nbFeatures, nbSeqLabels);
  dVector Partition(nbSeqLabels);
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Step 1 : Run Inference in each network
  for(y=0; y<nbSeqLabels; y++) {
    pInfEngine->computeBeliefs(condBeliefs[y], pFeatureGen, X, m, true, y);
    Partition[y] = condBeliefs[y].partition;
  }
  lZx = Partition.logSumExp();
  
  // Find y_star
  y_star = -1; max_val = -DBL_MAX;
  for(y=0; y<nbSeqLabels; y++) {
    if( Partition[y] > max_val ) {
      y_star = y; max_val = Partition[y];
    }	}
  if( y_star==y_true ) return 0;
  
  loss = (y_star!=y_true) +  MAX(0,Partition[y_star] - Partition[y_true]);
  
  // Viterbi Decoding
  iVector hstar, htrue;
  dVector phstar, phtrue;
  viterbiDecoding(condBeliefs[y_star],hstar,phstar);
  viterbiDecoding(condBeliefs[y_true],htrue,phtrue);
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Step 2 : Compute subgradient
  feature* f;
  featureVector vecFeatures;
  
  dVector *w = m->getWeights();
  
  // ---------- Singleton Features ----------
  for(xi=0; xi<X->length(); xi++)
  {
    pFeatureGen->getFeatures(vecFeatures,X,m,xi,-1,y_star);
    for(k=0, f=vecFeatures.getPtr(); k<vecFeatures.size(); k++, f++)  {
      if( f->nodeState == hstar[xi] ) {
        phi_star += w->getValue(f->globalId)*f->value;
        //val = condBeliefs[y_star].belStates[xi][f->nodeState] * f->value;
        val = f->value;
        condEValues.addValue(y_star,f->globalId,val);
      }
    }
    
    pFeatureGen->getFeatures(vecFeatures,X,m,xi,-1,y_true);
    for(k=0, f=vecFeatures.getPtr(); k<vecFeatures.size(); k++, f++) {
      if( f->nodeState == htrue[xi] )  {
        phi_true += w->getValue(f->globalId)*f->value;
        //val = condBeliefs[y_true].belStates[xi][f->nodeState] * f->value;
        val = f->value;
        condEValues.addValue(y_true,f->globalId,val);
      }
    }
  }
  // ---------- Pairwise Features ----------
  for(xi=0, xj=1; xj<X->length(); xi++, xj++)
  {
    pFeatureGen->getFeatures(vecFeatures,X,m,xj,xi,y_star);
    for(k=0, f=vecFeatures.getPtr(); k<vecFeatures.size(); k++, f++) {
      if( f->prevNodeState==hstar[xi] && f->nodeState==hstar[xj] ) {
        phi_star += w->getValue(f->globalId)*f->value;
        //val = condBeliefs[y_star].belEdges[xi](f->prevNodeState,f->nodeState) * f->value;
        val = f->value;
        condEValues.addValue(y_star,f->globalId,val);
      }
    }
    
    pFeatureGen->getFeatures(vecFeatures,X,m,xj,xi,y_true);
    for(k=0, f=vecFeatures.getPtr(); k<vecFeatures.size(); k++, f++) {
      if( f->prevNodeState==htrue[xi] && f->nodeState==htrue[xj] ) {
        phi_true = w->getValue(f->globalId)*f->value;
        //val = condBeliefs[y_true].belEdges[xi](f->prevNodeState,f->nodeState) * f->value;
        val = f->value;
        condEValues.addValue(y_true,f->globalId,val);
      }
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Step 3: Compute loss & gradient
  if( vecGradient.getLength() != nbFeatures )
    vecGradient.create(nbFeatures);
  
  dVector rowJ(nbFeatures);
  
  condEValues.getRow(y_star,rowJ);
  //rowJ.multiply( exp(Partition[y_star]-lZx) );
  vecGradient.add(rowJ);
  
  condEValues.getRow(y_true,rowJ);
  //rowJ.multiply( exp(Partition[y_true]-lZx) );
  rowJ.negate();
  vecGradient.add(rowJ);
  
  return loss;
}

