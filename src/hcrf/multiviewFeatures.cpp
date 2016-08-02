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

#include "hcrf/multiviewFeatures.h"

///////////////////////////////////////////////////////////////////////////////
// GaussianWindowRawFeaturesMV
// Encodes dependency between h^{c}_{t} and x^{c}_{t-w:t+w}
///////////////////////////////////////////////////////////////////////////////

GaussianWindowRawFeaturesMV::GaussianWindowRawFeaturesMV(int vid, int ws): FeatureType(),
viewIdx(vid), windowSize(1+2*ws)
{
  strFeatureTypeName = "Multiview Gaussian Window Raw Feature Type";
  featureTypeId = MV_GAUSSIAN_WINDOW_RAW_FEATURE_ID;
  basicFeatureType = NODE_FEATURE;
  
  // Compute a normalized Gaussian kernel.
  weights = new double[windowSize];
  double sum=0;
  if( windowSize == 1 )
  {
    weights[0] = 1.0;
  }
  else
  {
    double alpha = 2.5;
    int N = windowSize-1;
    for( int i=0-N/2; i<=N-N/2; i++ )
    {
      double x = alpha * (i / (0.5*N));
      weights[i+N/2] = exp(-0.5*x*x);
      sum += weights[i+N/2];
    }
    for( int i=0; i<windowSize; i++ )
    {
      weights[i] /= sum;
    }
  }
}

GaussianWindowRawFeaturesMV::~GaussianWindowRawFeaturesMV()
{
  if( weights ) { delete [] weights; weights = 0; }
}

void GaussianWindowRawFeaturesMV::getFeatures(featureVector& listFeatures, DataSequence* X,
                                              Model* m, int nodeIndex, int prevNodeIndex, int seqLabel)
{
  int nodeView = nodeIndex/X->length();
  if( nodeView!=viewIdx ) return;
  
  // listValidIndex contains raw feature indices for this particular view
  std::vector<int> listValidIndex = m->getRawFeatureIndexMV(viewIdx);
  int nbStateLabels = m->getNumberOfStatesMV(viewIdx);
  int nbFeaturesDense = (int) listValidIndex.size();
  
  if( X->getPrecomputedFeatures()!=NULL && prevNodeIndex==-1 )
  {
    dMatrix *preFeatures = X->getPrecomputedFeatures();
    int nbNodes = preFeatures->getWidth();
    feature* pFeature;
    
    for( int s=0; s<nbStateLabels; s++ )
    {
      for( int f=0; f<nbFeaturesDense; f++ )
      {
        pFeature = listFeatures.addElement();
        pFeature->id = getIdOffset(seqLabel) + f + s*nbFeaturesDense;
        pFeature->globalId = getIdOffset() + f + s*nbFeaturesDense;
        pFeature->nodeView = nodeView;
        pFeature->nodeIndex = nodeIndex;
        pFeature->nodeState = s;
        pFeature->prevNodeView = -1;
        pFeature->prevNodeIndex = -1;
        pFeature->prevNodeState = -1;
        pFeature->sequenceLabel = seqLabel;
        pFeature->value = 0;
        
        for( int w=0; w<windowSize; w++ )
        {
          int idx = nodeIndex%X->length() - (w-(int)(windowSize/2));
          if( idx<0 || idx>nbNodes-1 ) continue;
          pFeature->value += weights[w] * preFeatures->getValue(listValidIndex.at(f),idx);
        }
      }
    }
  }
}


void GaussianWindowRawFeaturesMV::getAllFeatures(
                                                 featureVector& listFeatures, Model* m, int nbRawFeatures)
{
  int nbStateLabels = m->getNumberOfStatesMV(viewIdx);
  feature* pFeature;
  
  for( int s=0; s<nbStateLabels; s++ )
  {
    for( int f=0; f<nbRawFeatures; f++ )
    {
      pFeature = listFeatures.addElement();
      pFeature->id = getIdOffset() + f + s*nbRawFeatures;
      pFeature->globalId = getIdOffset() + f + s*nbRawFeatures;
      pFeature->nodeView = viewIdx;
      pFeature->nodeIndex = featureTypeId;
      pFeature->nodeState = s;
      pFeature->prevNodeView = -1;
      pFeature->prevNodeIndex = -1;
      pFeature->prevNodeState = -1;
      pFeature->sequenceLabel = -1;
      pFeature->value = f;
    }
  }
}

void GaussianWindowRawFeaturesMV::init(const DataSet& dataset, const Model& m)
{
  FeatureType::init(dataset, m);
  
  if( dataset.size() > 0 )
  {
    // listValidIndex contains raw feature indices for this particular view
    std::vector<int> listValidIndex = m.getRawFeatureIndexMV(viewIdx);
    
    int nbFeaturesPerState = (int)listValidIndex.size(); // |X^{c}|
    int nbStates = m.getNumberOfStatesMV(viewIdx); // |H^{c}|
    int nbSeqLabels = m.getNumberOfSequenceLabels(); // |Y|
    
    nbFeatures = nbStates * nbFeaturesPerState;
    for( int i=0; i<nbSeqLabels; i++ )
    {
      nbFeaturesPerLabel[i] = nbFeatures;
    }
  }
}

bool GaussianWindowRawFeaturesMV::isEdgeFeatureType()
{
  return false;
}


///////////////////////////////////////////////////////////////////////////////
// EdgeFeaturesMV
// Encodes dependency between h^{c}_{s} and h^{d}_{t}.
///////////////////////////////////////////////////////////////////////////////
EdgeFeaturesMV::EdgeFeaturesMV(int v1, int v2):FeatureType(),
prevViewIdx(v1), curViewIdx(v2)
{
  strFeatureTypeName = "Multiview Edge Feature Type";
  featureTypeId = MV_EDGE_FEATURE_ID;
  basicFeatureType = EDGE_FEATURE;
}

void EdgeFeaturesMV::getFeatures(featureVector& listFeatures,
                                 DataSequence* X, Model* m, int nodeIndex, int prevNodeIndex, int seqLabel)
{
  if( prevNodeIndex == -1 ) return;
  
  int T = X->length();
  int nodeView = nodeIndex/T;
  int prevNodeView = prevNodeIndex/T;
  if( prevViewIdx!=prevNodeView || curViewIdx!=nodeView ) return;
		
  feature* pFeature;
  int nbStateLabels1 = m->getNumberOfStatesMV(prevViewIdx);
  int nbStateLabels2 = m->getNumberOfStatesMV(curViewIdx);
  
  for( int s1=0; s1<nbStateLabels1; s1++ )
  {
    for( int s2=0; s2<nbStateLabels2; s2++ )
    {
      pFeature = listFeatures.addElement();
      pFeature->id = getIdOffset(seqLabel) + s2 + s1*nbStateLabels2;
      pFeature->globalId = getIdOffset() + s2 + s1*nbStateLabels2 + seqLabel*nbStateLabels1*nbStateLabels2;
      pFeature->nodeView = nodeView;
      pFeature->nodeIndex = nodeIndex;
      pFeature->nodeState = s2;
      pFeature->prevNodeView = prevNodeView;
      pFeature->prevNodeIndex = prevNodeIndex;
      pFeature->prevNodeState = s1;
      pFeature->sequenceLabel = seqLabel;
      pFeature->value = 1.0f;
    }
  }
}

void EdgeFeaturesMV::getAllFeatures(featureVector& listFeatures, Model* m, int)
{
  int nbStateLabels1 = m->getNumberOfStatesMV(prevViewIdx);
  int nbStateLabels2 = m->getNumberOfStatesMV(curViewIdx);
  
  int nbSeqLabels = m->getNumberOfSequenceLabels();
  if( nbSeqLabels==0 ) nbSeqLabels = 1;
  
  feature* pFeature;
  for( int seqLabel=0; seqLabel<nbSeqLabels; seqLabel++ )
  {
    for(int s1=0; s1<nbStateLabels1; s1++ )
    {
      for(int s2=0; s2<nbStateLabels2; s2++ )
      {
        pFeature = listFeatures.addElement();
        pFeature->id = getIdOffset() + s2 + s1*nbStateLabels2 + seqLabel*nbStateLabels1*nbStateLabels2;
        pFeature->globalId = getIdOffset() + s2 + s1*nbStateLabels2 + seqLabel*nbStateLabels1*nbStateLabels2;
        pFeature->nodeView = featureTypeId;
        pFeature->nodeIndex = featureTypeId;
        pFeature->nodeState = s2;
        pFeature->prevNodeView = -1;
        pFeature->prevNodeIndex = -1;
        pFeature->prevNodeState = s1;
        pFeature->sequenceLabel = seqLabel;
        pFeature->value = 1.0f;
      }
    }
  }
}


void EdgeFeaturesMV::init(const DataSet& dataset, const Model& m)
{
  FeatureType::init(dataset,m);
  int nbStateLabels1 = m.getNumberOfStatesMV(prevViewIdx);
  int nbStateLabels2 = m.getNumberOfStatesMV(curViewIdx);
  
  int nbSeqLabels = m.getNumberOfSequenceLabels();
  if( nbSeqLabels==0 )
  {
    nbFeatures = nbStateLabels1 * nbStateLabels2;
  }
  else
  {
    nbFeatures = nbSeqLabels * nbStateLabels1 * nbStateLabels2;
    for( int i=0; i<nbSeqLabels; i++ )
      nbFeaturesPerLabel[i] = nbStateLabels1*nbStateLabels2;
  }
}

void EdgeFeaturesMV::computeFeatureMask(iMatrix& matFeautureMask, const Model& m)
{
  int nbLabels = m.getNumberOfSequenceLabels();
  int firstOffset = idOffset;
  
  for( int j=0; j<nbLabels; j++ )
  {
    int lastOffset = firstOffset + nbFeaturesPerLabel[j];
    
    for( int i=firstOffset; i<lastOffset; i++ )
      matFeautureMask(i,j) = 1;
    
    firstOffset += nbFeaturesPerLabel[j];
  }
}

bool EdgeFeaturesMV::isEdgeFeatureType()
{
  return true;
}


///////////////////////////////////////////////////////////////////////////////
// LabelEdgeFeaturesMV
// Encodes dependency between y and h^{c}_{t}.
///////////////////////////////////////////////////////////////////////////////

LabelEdgeFeaturesMV::LabelEdgeFeaturesMV(int vid): FeatureType(),viewIdx(vid)
{
  strFeatureTypeName = "Multiview Label Edge Feature Type";
  featureTypeId = MV_LABEL_EDGE_FEATURE_ID;
  basicFeatureType = LABEL_EDGE_FEATURE;
}


void LabelEdgeFeaturesMV::getFeatures(featureVector& listFeatures,
                                      DataSequence* X, Model* m, int nodeIndex, int prevNodeIndex, int seqLabel)
{
  int nodeView = nodeIndex/X->length();
  if( nodeView!=viewIdx ) return;
  
  if( seqLabel!=-1 && m->getNumberOfSequenceLabels()>0 && prevNodeIndex==-1)
  {
    feature* pFeature;
    int nbStateLabels = m->getNumberOfStatesMV(viewIdx);
    for( int s=0; s<nbStateLabels; s++ )
    {
      pFeature = listFeatures.addElement();
      pFeature->id = getIdOffset(seqLabel) + s;
      pFeature->globalId = getIdOffset() + s + seqLabel*nbStateLabels ;
      pFeature->nodeView = nodeView;
      pFeature->nodeIndex = nodeIndex;
      pFeature->nodeState = s;
      pFeature->prevNodeView = -1;
      pFeature->prevNodeIndex = -1;
      pFeature->prevNodeState = -1;
      pFeature->sequenceLabel = seqLabel;
      pFeature->value = 1.0f;
    }
  }
}


void LabelEdgeFeaturesMV::getAllFeatures(featureVector& listFeatures, Model* m, int)
{
  if(m->getNumberOfSequenceLabels() > 0)
  {
    int nbStateLabels = m->getNumberOfStatesMV(viewIdx);
    int nbSeqLabels = m->getNumberOfSequenceLabels();
    feature* pFeature;
    for(int seqLabel = 0; seqLabel < nbSeqLabels; seqLabel++)
    {
      for(int s = 0; s < nbStateLabels; s++)
      {
        pFeature = listFeatures.addElement();
        pFeature->id = getIdOffset() + s + seqLabel*nbStateLabels ;
        pFeature->globalId = getIdOffset() + s + seqLabel*nbStateLabels ;
        pFeature->nodeView = featureTypeId;
        pFeature->nodeIndex = featureTypeId;
        pFeature->nodeState = s;
        pFeature->prevNodeView = -1;
        pFeature->prevNodeIndex = -1;
        pFeature->prevNodeState = -1;
        pFeature->sequenceLabel = seqLabel;
        pFeature->value = 1.0f;
      }
    }
  }
}


void LabelEdgeFeaturesMV::init(const DataSet& dataset, const Model& m)
{
  FeatureType::init(dataset,m);
  int nbStateLabels = m.getNumberOfStatesMV(viewIdx);
  int nbSeqLabels = m.getNumberOfSequenceLabels();
  
  if(nbSeqLabels == 0)
    nbFeatures = 0;
  else
  {
    nbFeatures = nbStateLabels*nbSeqLabels;
    for(int i = 0; i < nbSeqLabels; i++)
      nbFeaturesPerLabel[i] = nbStateLabels;
  }
}

void LabelEdgeFeaturesMV::computeFeatureMask(iMatrix& matFeautureMask, const Model& m)
{
  int nbLabels = m.getNumberOfSequenceLabels();
  int firstOffset = idOffset;
  
  for(int j = 0; j < nbLabels; j++)
  {
    int lastOffset = firstOffset + nbFeaturesPerLabel[j];
    
    for(int i = firstOffset; i < lastOffset; i++)
      matFeautureMask(i,j) = 1;
    
    firstOffset += nbFeaturesPerLabel[j];
  }
}

bool LabelEdgeFeaturesMV::isEdgeFeatureType()
{ 
  return false;
}

