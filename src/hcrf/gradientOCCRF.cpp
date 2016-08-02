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

using namespace std;

GradientOCCRF::GradientOCCRF(InferenceEngine* ie, FeatureGenerator* fg): Gradient(ie, fg){}

double GradientOCCRF::computeGradient(dVector& vecGradient, Model* m, DataSequence* X)
{
  return mle(vecGradient,m,X);
}

double GradientOCCRF::mle(dVector& vecGradient, Model* m, DataSequence* X)
{
  int xi, xj, yi, yj, k, nbFeatures;
  double phi_true=0, phi_star=0, loss=0, val;
  
  Beliefs bel;
  pInfEngine->computeBeliefs(bel,pFeatureGen, X, m, false);
  
  iVector ystar; dVector pystar;
  viterbiDecoding(bel,ystar,pystar);
  ystar.set(1);
		
  feature* f;
  featureVector vecFeatures;
  
  nbFeatures = pFeatureGen->getNumberOfFeatures();
  if(vecGradient.getLength() != nbFeatures)
    vecGradient.create(nbFeatures);
  
  dVector *w = m->getWeights();
  dVector localGrad(nbFeatures);
  
  // Loop over nodes to compute features and update the gradient
  for(xi=0; xi<X->length(); xi++)
  {
    // Read the label for this state
    yi = X->getStateLabels(xi);
    
    //Get nodes features
    pFeatureGen->getFeatures(vecFeatures,X,m,xi,-1);
    f = vecFeatures.getPtr();
    for(k=0; k<vecFeatures.size(); k++, f++) {
      if(f->nodeState==yi) {
        phi_true += w->getValue(f->id) * f->value;
        localGrad[f->id] -= f->value;
      }
      else if(f->nodeState==ystar[xi]) {
        phi_star += w->getValue(f->id) * f->value;
        localGrad[f->id] += f->value;
      }
      val = bel.belStates[xi][f->nodeState]*f->value;
      localGrad[f->id] -= val;
    }
  }
  
  //Loop over edges to compute features and update the gradient
  for(xi=0; xi<X->length()-1; xi++) {
    xj = xi+1;
    yi = X->getStateLabels(xi);
    yj = X->getStateLabels(xj);
    
    //Get nodes features
    pFeatureGen->getFeatures(vecFeatures,X,m,xj,xi);
    f = vecFeatures.getPtr();
    for(k=0; k<vecFeatures.size(); k++, f++){
      if(f->prevNodeState == yi && f->nodeState == yj) {
        phi_true += w->getValue(f->id) * f->value;
        localGrad[f->id] -= f->value;
      }
      else if(f->prevNodeState==ystar[xi] && f->nodeState==ystar[xj]) {
        phi_star += w->getValue(f->id) * f->value;
        localGrad[f->id] += f->value;
      }
      val = bel.belEdges[xi](f->prevNodeState,f->nodeState)*f->value;
      localGrad[f->id] -= val;
    }
  }
  
  vecGradient.add(localGrad);
  dVector prob(2);
  prob[0] = exp(phi_true-bel.partition);
  prob[1] = exp(phi_star-bel.partition);
  loss = MAX(0, m->getRho()-(prob[0]-prob[1]));
  //printf("(+,-,Z)=(%.6f,%.6f,%.6f), %.6f ( = %.6f - %.6f)\n", phi_true,phi_star,bel.partition, prob[0]-prob[1], prob[0], prob[1]);
  
  return loss;
}



double GradientOCCRF::max_margin(dVector& vecGradient, Model* m, DataSequence* X)
{
  int xi, xj, yi, yj, k, nbFeatures;
  double hamming_loss=0;
  dVector phi_loss(2); // 0:true, 1:star
  
  nbFeatures = pFeatureGen->getNumberOfFeatures();
  
  Beliefs bel;
  pInfEngine->computeBeliefs(bel,pFeatureGen, X, m, false);
  
  // Compute Hamming loss
  iVector ystar; dVector pystar;
  viterbiDecoding(bel,ystar,pystar);
  for(xi=0; xi<X->length(); xi++)
    if( X->getStateLabels(xi) != ystar[xi] )
      hamming_loss++;
  
  // Compute gradients
  // Check the size of vecGradient
  feature* f;
  featureVector vecFeatures;
  
  if(vecGradient.getLength() != nbFeatures)
    vecGradient.create(nbFeatures);
  
  dVector localGrad(nbFeatures);
  dVector *w = m->getWeights();
  
  // Loop over nodes to compute features and update the gradient
  for(xi=0; xi<X->length(); xi++)
  {
    // Read the label for this state
    yi = X->getStateLabels(xi);
    
    //Get nodes features
    pFeatureGen->getFeatures(vecFeatures,X,m,xi,-1);
    f = vecFeatures.getPtr();
    for(k=0; k<vecFeatures.size(); k++, f++)
    {
      if(f->nodeState==yi) {
        phi_loss[0] += w->getValue(f->id) * f->value;
        localGrad[f->id] -= f->value;
      }
      else if(f->nodeState==ystar[xi]) {
        phi_loss[1] += w->getValue(f->id) * f->value;
        localGrad[f->id] += f->value;
      }
    }
  }
  
  //Loop over edges to compute features and update the gradient
  for(xi=0; xi<X->length()-1; xi++) {
    xj = xi+1;
    yi = X->getStateLabels(xi);
    yj = X->getStateLabels(xj);
    
    //Get nodes features
    pFeatureGen->getFeatures(vecFeatures,X,m,xj,xi);
    f = vecFeatures.getPtr();
    for(k=0; k<vecFeatures.size(); k++, f++)
    {
      if(f->prevNodeState == yi && f->nodeState == yj) {
        phi_loss[0] += w->getValue(f->id) * f->value;
        localGrad[f->id] -= f->value;
      }
      else if(f->prevNodeState==ystar[xi] && f->nodeState==ystar[xj]) {
        phi_loss[1] += w->getValue(f->id) * f->value;
        localGrad[f->id] += f->value;
      }
    }
  }
  
  // Taskar et al. (2004) vs Tsochantaridis et al. (2005)
  bool useTaskar = false;
  double scale = (useTaskar) ? 1 : hamming_loss;
  
  // Done!
  localGrad.multiply(scale);
  vecGradient.add(localGrad);
  
  phi_loss.multiply(1/bel.partition);
  return hamming_loss + scale*(phi_loss[1]-phi_loss[0]);
}

