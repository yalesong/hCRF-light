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

#ifndef MULTIVIEW_FEATURES_H
#define MULTIVIEW_FEATURES_H

#include "hcrf/featuregenerator.h"

class GaussianWindowRawFeaturesMV: public FeatureType
{
public:
  GaussianWindowRawFeaturesMV(int viewIdx, int windowSize=0);
  ~GaussianWindowRawFeaturesMV();
  
  void init(const DataSet& dataset, const Model& m);
  
  void getFeatures(featureVector& listFeatures, DataSequence* X, Model* M,
                   int nodeIndex, int prevNodeIndex, int seqLabel = -1);
  void getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures);
  
  bool isEdgeFeatureType();
  
private:
  int viewIdx;
  int windowSize;
  double* weights;
};


class EdgeFeaturesMV: public FeatureType
{
public:
  EdgeFeaturesMV(int prevViewIdx, int prevTimeIdx);
  
  void init(const DataSet& dataset, const Model& m);
  
  void getFeatures(featureVector& listFeatures, DataSequence* X, Model* M,
                   int nodeIndex, int prevNodeIndex, int seqLabel = -1);
  void getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures);
  
  void computeFeatureMask(iMatrix& matFeautureMask, const Model& m);
  bool isEdgeFeatureType();
  
private:
  int prevViewIdx, curViewIdx;
};


class LabelEdgeFeaturesMV: public FeatureType
{
public:
  LabelEdgeFeaturesMV(int viewIdx);
  
  void init(const DataSet& dataset, const Model& m);
  
  void getFeatures(featureVector& listFeatures, DataSequence* X, Model* M,
                   int nodeIndex, int prevNodeIndex, int seqLabel = -1);
  void getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures);
  
  void computeFeatureMask(iMatrix& matFeautureMask, const Model& m);
  bool isEdgeFeatureType();
  
private:
  int viewIdx;
};


#endif



