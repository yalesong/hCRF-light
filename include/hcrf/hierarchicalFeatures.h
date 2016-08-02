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

#ifndef HIERARCHICAL_FEATURES_H
#define HIERARCHICAL_FEATURES_H

#include "hcrf/featuregenerator.h"

class HSSRawFeatures: public FeatureType
{
public:
  HSSRawFeatures(int layer);
  ~HSSRawFeatures();
  
  void init(const DataSet& dataset, const Model& m);
  
  void getFeatures(featureVector& listFeatures, DataSequence* X, Model* M,
                   int nodeIndex, int prevNodeIndex, int seqLabel = -1);
  void getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures);
  
  bool isEdgeFeatureType();
};

class HSSGateNodeFeatures : public FeatureType
{
public:
  HSSGateNodeFeatures(int layer, int nbGates, int windowSize = 0);
  
  virtual void init(const DataSet& dataset, const Model& m);
  virtual void getFeatures(featureVector& listFeatures, DataSequence* X, Model* m,
                           int nodeIndex, int prevNodeIndex, int seqLabel = -1);
  virtual bool isEdgeFeatureType();
  
  void getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures);
  
  int getNbFeaturesPerGate() {return nbFeaturesPerGate;}
  int getNbGates() {return nbGates;}
  
  //Function used by gradient to have direct access to raw features.
  void getPreGateFeatures(featureVector& listFeatures, DataSequence* X, Model* m,
                          int nodeIndex, int prevNodeIndex, int seqLabel = -1);
  
private:
  //Gating function of the neural network. h(x) = 1/(1+exp(x))
  double gate(double sum);
  
  int nbFeaturesPerGate;
  int windowSize;
  int nbGates;
};

class HSSEdgeFeatures: public FeatureType
{
public:
  HSSEdgeFeatures(int layer);
  
  void init(const DataSet& dataset, const Model& m);
  
  void getFeatures(featureVector& listFeatures, DataSequence* X, Model* M,
                   int nodeIndex, int prevNodeIndex, int seqLabel = -1);
  void getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures);
  
  void computeFeatureMask(iMatrix& matFeautureMask, const Model& m);
  bool isEdgeFeatureType();
};

class HSSLabelEdgeFeatures: public FeatureType
{
public:
  HSSLabelEdgeFeatures(int layer);
  
  void init(const DataSet& dataset, const Model& m);
  
  void getFeatures(featureVector& listFeatures, DataSequence* X, Model* M,
                   int nodeIndex, int prevNodeIndex, int seqLabel = -1);
  void getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures);
  
  void computeFeatureMask(iMatrix& matFeautureMask, const Model& m);
  bool isEdgeFeatureType();
};

#endif



