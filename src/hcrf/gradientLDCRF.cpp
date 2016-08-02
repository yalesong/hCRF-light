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

GradientLDCRF::GradientLDCRF(InferenceEngine* infEngine,FeatureGenerator* featureGen):
Gradient(infEngine, featureGen)
{}

double GradientLDCRF::computeGradient(dVector& vecGradient, Model* m,DataSequence* X)
{
  //compute beliefs
  Beliefs bel;
  Beliefs belMasked;
  pInfEngine->computeBeliefs(bel,pFeatureGen, X, m, true,-1, false);
  pInfEngine->computeBeliefs(belMasked,pFeatureGen, X, m, true,-1, true);
  
  //Check the size of vecGradient
  int nbFeatures = pFeatureGen->getNumberOfFeatures();
  if(vecGradient.getLength() != nbFeatures)
    vecGradient.create(nbFeatures);
  
  featureVector vecFeatures;
  
  //Loop over nodes to compute features and update the gradient
  for(int i = 0; i < X->length(); i++)
  {
    //Get nodes features
    pFeatureGen->getFeatures(vecFeatures, X,m,i,-1);
    feature* pFeature = vecFeatures.getPtr();
    for(int j = 0; j < vecFeatures.size(); j++, pFeature++)
    {
      //p(y_i=s|x)*f_k(i,s,x)
      vecGradient[pFeature->globalId] -= bel.belStates[i][pFeature->nodeState]*pFeature->value;
      vecGradient[pFeature->globalId] += belMasked.belStates[i][pFeature->nodeState]*pFeature->value;
    }
  }
  
  for(int i = 0; i < X->length()-1; i++) // Loop over all rows (the previous node index)
  {
    //Get nodes features
    pFeatureGen->getFeatures(vecFeatures, X,m,i+1,i);
    feature* pFeature = vecFeatures.getPtr();
    for(int j = 0; j < vecFeatures.size(); j++, pFeature++)
    {
      //p(y_i=s1,y_j=s2|x)*f_k(i,j,s1,s2,x) is subtracted from the gradient
      double a = bel.belEdges[i](pFeature->prevNodeState,pFeature->nodeState)*pFeature->value;
      double b = belMasked.belEdges[i](pFeature->prevNodeState,pFeature->nodeState)*pFeature->value;
      vecGradient[pFeature->globalId] -= a;
      vecGradient[pFeature->globalId] += b;
    }
  }
  
  return bel.partition - belMasked.partition;
}
