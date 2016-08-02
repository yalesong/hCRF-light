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

#include "hcrf/hierarchicalFeatures.h"


///////////////////////////////////////////////////////////////
// HSSRawFeatures

HSSRawFeatures::HSSRawFeatures(int l)
: FeatureType(l)
{
  strFeatureTypeName = "Hierarchical Raw Feature Type";
  featureTypeId = HSS_RAW_FEATURE_ID;
  basicFeatureType = NODE_FEATURE;
}

HSSRawFeatures::~HSSRawFeatures()
{}

void HSSRawFeatures::getFeatures(featureVector& listFeatures, DataSequence* X,
                                 Model* m, int nodeIndex, int prevNodeIndex, int y)
{
  if( X->getPrecomputedFeatures()==NULL || prevNodeIndex != -1 ) return;
  int s, f, k, nodeLayer, nbStates, nbFeaturesDense;
  
  nodeLayer = nodeIndex/X->length();
  nodeIndex = nodeIndex%X->length();
  if( nodeLayer != layer) return;
  
  dMatrix *preFeatures = X->getPrecomputedFeatures();
  feature* pFeature;
  
  nbStates = m->getNumberOfStates();
  nbFeaturesDense = preFeatures->getHeight();
  
  std::vector<int> super_x = X->getDeepSeqGroupLabels()->at(layer).at(nodeIndex);
  
  for( s=0; s<nbStates; s++ )
  {
    for( f=0; f<nbFeaturesDense; f++ )
    {
      pFeature = listFeatures.addElement();
      pFeature->id = getIdOffset(y) + f + s*nbFeaturesDense;
      pFeature->globalId = getIdOffset() + f + s*nbFeaturesDense;
      pFeature->nodeIndex = nodeIndex;
      pFeature->nodeState = s;
      pFeature->prevNodeIndex = -1;
      pFeature->prevNodeState = -1;
      pFeature->sequenceLabel = y;
      pFeature->value = 0;
      for( k=0; k<(int)super_x.size(); k++ )
        pFeature->value += preFeatures->getValue(f,super_x[k]);
      pFeature->value /= (double)super_x.size();
    }
  }
}


void HSSRawFeatures::getAllFeatures(
                                    featureVector& listFeatures, Model* m, int nbRawFeatures)
{
  int s, f, nbStates;
  feature* pFeature;
  
  nbStates = m->getNumberOfStates();
  
  for( s=0; s<nbStates; s++ )
  {
    for( f=0; f<nbRawFeatures; f++ )
    {
      pFeature = listFeatures.addElement();
      pFeature->id = getIdOffset() + f + s*nbRawFeatures;
      pFeature->globalId = getIdOffset() + f + s*nbRawFeatures;
      pFeature->nodeIndex = featureTypeId;
      pFeature->nodeState = s;
      pFeature->prevNodeIndex = -1;
      pFeature->prevNodeState = -1;
      pFeature->sequenceLabel = -1;
      pFeature->value = f;
    }
  }
}

void HSSRawFeatures::init(const DataSet& dataset, const Model& m)
{
  int i, nbStates, nbSeqLabels, nbFeaturesPerState;
  FeatureType::init(dataset, m);
  if( dataset.size() > 0 )
  {
    nbStates = m.getNumberOfStates();
    nbFeaturesPerState = dataset.at(0)->getPrecomputedFeatures()->getHeight();
    nbFeatures = nbStates * nbFeaturesPerState;
    nbSeqLabels = m.getNumberOfSequenceLabels();
    
    for( i=0; i<nbSeqLabels; i++ )
      nbFeaturesPerLabel[i] = nbFeatures;
  }
}

bool HSSRawFeatures::isEdgeFeatureType()
{
  return false;
}

///////////////////////////////////////////////////////////////////////////////////
// HSSGateNodeFeatures
//
HSSGateNodeFeatures::HSSGateNodeFeatures(int l, int g, int w)
: FeatureType(l),windowSize(w),nbGates(g)
{
  strFeatureTypeName = "Hierarchical Gate Node Feature Type";
  featureTypeId = HSS_GATE_NODE_FEATURE_ID;
  basicFeatureType = NODE_FEATURE;
}

void HSSGateNodeFeatures::getFeatures(featureVector& listFeatures,
                                      DataSequence* X, Model* m, int nodeIndex, int prevNodeIndex, int seqLabel)
{
  if( X->getPrecomputedFeatures()==NULL || prevNodeIndex!=-1 ) return;
  
  int idx,s,f,g,k,nidx,nodeLayer,nbNodes,nbStates,nbSG,nbFeaturesDense,gXfpg,sXG;
  double gateSum = 0;
  
  nodeLayer = nodeIndex/X->length();
  nodeIndex = nodeIndex%X->length();
  if( nodeLayer != layer) return;
  
  dMatrix *preFeatures = X->getPrecomputedFeatures();
  feature* pFeature;
  
  nbStates = m->getNumberOfStates();
  nbSG = nbStates*nbGates;
  nbFeaturesDense = preFeatures->getHeight();
  nbNodes = preFeatures->getWidth();
  
  dVector* lambda = m->getWeights();
  dMatrix* gates = X->getGateMatrix();
  
  std::vector<int> super_x = X->getDeepSeqGroupLabels()->at(layer).at(nodeIndex);
  
  // Precompute gate values, (row,col) = (gate,node)
  if( gates->getHeight()!=nbGates || gates->getWidth()!=nbNodes )
    gates->resize(nbNodes,nbGates);
  
  // 1/size(x_t) * \sum_{x' \in x_t} g(w*x')
  for( g=0; g<nbGates; g++ )
  {
    gates->setValue(g,nodeIndex,0.0);
    
    gXfpg = g*(nbFeaturesDense+1); // 1: bias
    
    // gate sum
    for( k=0; k<(int)super_x.size(); k++ )
    {
      nidx = super_x[k];
      
      // gate bias
      idx = getIdOffset() + nbSG + gXfpg + nbFeaturesDense;
      gateSum = (*lambda)[idx];
      
      for( f=0; f<nbFeaturesDense; f++ )
      {
        idx = getIdOffset() + nbSG + gXfpg + f;
        gateSum += (*lambda)[idx] * preFeatures->getValue(f,nidx);
      }
      gates->addValue(g,nodeIndex,gate(gateSum));
    }
    gates->setValue(g,nodeIndex, gates->getValue(g,nodeIndex) / (double)super_x.size());
  }
  
  // Then compute node features from gate values
  for( s=0; s<nbStates; s++ )
  {
    sXG = s*nbGates;
    for( g=0; g<nbGates; g++ )
    {
      pFeature = listFeatures.addElement();
      pFeature->id = getIdOffset(seqLabel) + g + sXG;
      pFeature->globalId = getIdOffset() + g + sXG;
      pFeature->nodeIndex = nodeIndex;
      pFeature->nodeState = s;
      pFeature->prevNodeIndex = -1;
      pFeature->prevNodeState = g; // quick-and-dirty fix
      pFeature->sequenceLabel = seqLabel;
      pFeature->value = gates->getValue(g,nodeIndex);
      pFeature->featureTypeId = HSS_GATE_NODE_FEATURE_ID;
    }
  }
}

void HSSGateNodeFeatures::getPreGateFeatures(featureVector& listFeatures,
                                             DataSequence* X, Model* m, int nodeIndex,
                                             int prevNodeIndex, int seqLabel)
{
  if( X->getPrecomputedFeatures()==NULL || prevNodeIndex!=-1 ) return;
  listFeatures.clear();
  
  int f,g,k,nodeLayer,nbNodes,nbStates,nbFeaturesDense,nbSG,gXfpg;
  
  nodeLayer = nodeIndex/X->length();
  nodeIndex = nodeIndex%X->length();
  if( nodeLayer != layer) return;
  
  dMatrix* preFeatures = X->getPrecomputedFeatures();
  feature* pFeature;
  
  nbNodes = preFeatures->getWidth();
  nbStates = m->getNumberOfStates();
  nbFeaturesDense = preFeatures->getHeight();
  nbSG = nbStates*nbGates;
  
  std::vector<int> super_x = X->getDeepSeqGroupLabels()->at(layer).at(nodeIndex);
  
  for( g=0; g<nbGates; g++ )
  {
    gXfpg = g*nbFeaturesPerGate;
    for( f=0; f<nbFeaturesDense; f++ )
    {
      pFeature = listFeatures.addElement();
      pFeature->id = getIdOffset(seqLabel) + nbSG + gXfpg + f;
      pFeature->globalId = getIdOffset() + nbSG + gXfpg + f;
      pFeature->nodeIndex = nodeIndex;
      pFeature->nodeState = -1;
      pFeature->prevNodeIndex = -1;
      pFeature->prevNodeState = g; // this is a quick and dirty fix
      pFeature->sequenceLabel = seqLabel;
      pFeature->value = 0;
      for( k=0; k<(int)super_x.size(); k++ )
        pFeature->value += preFeatures->getValue(f,super_x[k]);
      pFeature->value /= (double)super_x.size();
    }
    // Gate bias.
    pFeature = listFeatures.addElement();
    pFeature->id = getIdOffset(seqLabel) + nbSG + gXfpg + nbFeaturesDense;
    pFeature->globalId = getIdOffset() + nbSG + gXfpg + nbFeaturesDense;
    pFeature->nodeIndex = nodeIndex;
    pFeature->nodeState = -1;
    pFeature->prevNodeIndex = -1;
    pFeature->prevNodeState = g; // this is a quick and dirty fix
    pFeature->sequenceLabel = seqLabel;
    pFeature->value = 1.0;
  }
}


void HSSGateNodeFeatures::getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures)
{
  int s, f, g, nbStates, nbSG, sXG, gXfpg;
  feature* pFeature;
  
  nbStates = m->getNumberOfStates();
  nbSG = nbStates*nbGates;
  
  // Node features.
  for( s=0; s<nbStates; s++ ) {
    sXG = s*nbGates;
    for( g=0; g<nbGates; g++ ) {
      pFeature = listFeatures.addElement();
      pFeature->id = getIdOffset() + g + sXG;
      pFeature->globalId = getIdOffset() + g + sXG;
      pFeature->nodeIndex = HSS_GATE_NODE_FEATURE_ID;
      pFeature->nodeState = s;
      pFeature->prevNodeIndex = -1;
      pFeature->prevNodeState = -1;
      pFeature->sequenceLabel = -1;
      pFeature->value = g;
    }
  }
  
  // Gate features.
  for( g=0; g<nbGates; g++ ) {
    gXfpg = g*nbFeaturesPerGate;
    for( f=0; f<nbFeaturesPerGate; f++ ) {
      pFeature = listFeatures.addElement();
      pFeature->id = getIdOffset() + nbSG + gXfpg+ f;
      pFeature->globalId = getIdOffset() + nbSG + gXfpg+ f;
      pFeature->nodeIndex = HSS_GATE_NODE_FEATURE_ID;
      pFeature->nodeState = -1;
      pFeature->prevNodeIndex = -1;
      pFeature->prevNodeState = -1;
      pFeature->sequenceLabel = -1;
      pFeature->value = f;
    }
  }
}

void HSSGateNodeFeatures::init(const DataSet& dataset, const Model& m)
{
  int y, nbStates, nbSeqLabels;
  FeatureType::init(dataset,m);
  if( dataset.size() > 0 )
  {
    nbStates = m.getNumberOfStates();
    nbSeqLabels = m.getNumberOfSequenceLabels();
    
    // Total number of features needed by each gate, dimX+1 (+1 for bias).
    nbFeaturesPerGate = (*dataset.begin())->getPrecomputedFeatures()->getHeight() + 1;
    
    // This number represents the number of weights needed.
    // A call to getFeatures will only return (nbStates*nbGates) features.
    nbFeatures = nbGates * (nbStates + nbFeaturesPerGate);
    for( y=0; y<nbSeqLabels; y++)
      nbFeaturesPerLabel[y] = nbFeatures;
  }
}

bool HSSGateNodeFeatures::isEdgeFeatureType()
{
  return false;
}

double HSSGateNodeFeatures::gate(double sum)
{
  return 1.0/(1.0+exp(-sum));
}


///////////////////////////////////////////////////////////////
// HSSEdgeFeatures

HSSEdgeFeatures::HSSEdgeFeatures(int l):FeatureType(l)
{
  strFeatureTypeName = "Hierarchical Edge Feature Type";
  featureTypeId = HSS_EDGE_FEATURE_ID;
  basicFeatureType = EDGE_FEATURE;
}

void HSSEdgeFeatures::getFeatures(featureVector& listFeatures,
                                  DataSequence* X, Model* m, int nodeIndex,
                                  int prevNodeIndex, int y)
{
  if( prevNodeIndex==-1 || prevNodeIndex+1!=nodeIndex ) return;
  int s1, s2, nbNodes, nbStates, nodeLayer;
  
  nodeLayer = nodeIndex/X->length();
  nodeIndex = nodeIndex%X->length();
  prevNodeIndex = prevNodeIndex%X->length();
  
  if( nodeLayer != layer) return;
  
  nbNodes = (X->getPrecomputedFeatures())
		? X->getPrecomputedFeatures()->getWidth()
		: (int)X->getPrecomputedFeaturesSparse()->getWidth();
  
  feature* pFeature;
  nbStates = m->getNumberOfStates();
  for( s1=0; s1<nbStates; s1++)
  {
    for( s2=0; s2<nbStates; s2++)
    {
      pFeature = listFeatures.addElement();
      pFeature->id = getIdOffset(y) + s2 + s1*nbStates ;
      pFeature->globalId = getIdOffset() + s2 + s1*nbStates + y*nbStates*nbStates ;
      pFeature->nodeIndex = nodeIndex;
      pFeature->nodeState = s2;
      pFeature->prevNodeIndex = prevNodeIndex;
      pFeature->prevNodeState = s1;
      pFeature->sequenceLabel = y;
      pFeature->value = 1.0f;
    }
  }
}

void HSSEdgeFeatures::getAllFeatures(featureVector& listFeatures, Model* m, int)
{
  int y, s1, s2, nbStates, nbSeqLabels;
  feature* pFeature;
  
  nbStates = m->getNumberOfStates();
  nbSeqLabels = m->getNumberOfSequenceLabels();
  if(nbSeqLabels == 0) nbSeqLabels = 1;
  
  for( y=0; y<nbSeqLabels; y++ ) {
    for( s1=0; s1<nbStates; s1++ ) {
      for( s2=0; s2<nbStates; s2++ ) {
        pFeature = listFeatures.addElement();
        pFeature->id = getIdOffset() + s2 + s1*nbStates + y*nbStates*nbStates;
        pFeature->globalId = getIdOffset() + s2 + s1*nbStates + y*nbStates*nbStates ;
        pFeature->nodeIndex = featureTypeId;
        pFeature->nodeState = s2;
        pFeature->prevNodeIndex = -1;
        pFeature->prevNodeState = s1;
        pFeature->sequenceLabel = y;
        pFeature->value = 1.0f;
      }
    }
  }
}


void HSSEdgeFeatures::init(const DataSet& dataset, const Model& m)
{
  int i, nbStates, nbSeqLabels;
  
  FeatureType::init(dataset,m);
  nbStates = m.getNumberOfStates();
  nbSeqLabels = m.getNumberOfSequenceLabels();
  
  if(nbSeqLabels == 0)
    nbFeatures = nbStates*nbStates;
  else {
    nbFeatures = nbStates*nbStates*nbSeqLabels;
    for( i=0; i<nbSeqLabels; i++ )
      nbFeaturesPerLabel[i] = nbStates*nbStates;
  }
}

void HSSEdgeFeatures::computeFeatureMask(iMatrix& matFeautureMask, const Model& m)
{
  int i, j, nbSeqLabels, firstOffset, lastOffset;
  
  nbSeqLabels = m.getNumberOfSequenceLabels();
  firstOffset = idOffset;
  
  for( j=0; j<nbSeqLabels; j++) {
    lastOffset = firstOffset + nbFeaturesPerLabel[j];
    
    for( i=firstOffset; i<lastOffset; i++)
      matFeautureMask(i,j) = 1;
    firstOffset += nbFeaturesPerLabel[j];
  }
}

bool HSSEdgeFeatures::isEdgeFeatureType()
{
  return true;
}



///////////////////////////////////////////////////////////////
// HSSLabelEdgeFeatures

HSSLabelEdgeFeatures::HSSLabelEdgeFeatures(int l): FeatureType(l)
{
  strFeatureTypeName = "Hierarchical Label Edge Feature Type";
  featureTypeId = HSS_LABEL_EDGE_FEATURE_ID;
  basicFeatureType = LABEL_EDGE_FEATURE;
}

void HSSLabelEdgeFeatures::getFeatures(featureVector& listFeatures,
                                       DataSequence* X, Model* m, int nodeIndex,
                                       int prevNodeIndex, int y)
{
  if( y==-1 || prevNodeIndex!=-1 || m->getNumberOfSequenceLabels()<1 ) return;
  
  int s, nodeLayer, nbStates;
  nodeLayer = nodeIndex/X->length();
  nodeIndex = nodeIndex%X->length();
  if( nodeLayer!=layer ) return;
  
  feature* pFeature;
  nbStates = m->getNumberOfStates();
  for( s=0; s<nbStates; s++)
  {
    pFeature = listFeatures.addElement();
    pFeature->id = getIdOffset(y) + s;
    pFeature->globalId = getIdOffset() + s + y*nbStates;
    pFeature->nodeIndex = nodeIndex;
    pFeature->nodeState = s;
    pFeature->prevNodeIndex = -1;
    pFeature->prevNodeState = -1;
    pFeature->sequenceLabel = y;
    pFeature->value = 1.0f;
  }
}

void HSSLabelEdgeFeatures::getAllFeatures(featureVector& listFeatures, Model* m, int)
{
  if( m->getNumberOfSequenceLabels()<1 ) return;
  
  int y, s, nbStates, nbSeqLabels;
  feature* pFeature;
  
  nbStates = m->getNumberOfStates();
  nbSeqLabels = m->getNumberOfSequenceLabels();
  for( y=0; y<nbSeqLabels; y++)
  {
    for( s=0; s<nbStates; s++)
    {
      pFeature = listFeatures.addElement();
      pFeature->id = getIdOffset() + s + y*nbStates;
      pFeature->globalId = getIdOffset() + s + y*nbStates;
      pFeature->nodeIndex = featureTypeId;
      pFeature->nodeState = s;
      pFeature->prevNodeIndex = -1;
      pFeature->prevNodeState = -1;
      pFeature->sequenceLabel = y;
      pFeature->value = 1.0f;
    }
  }
}


void HSSLabelEdgeFeatures::init(const DataSet& dataset, const Model& m)
{
  int i, nbStates, nbSeqLabels;
  FeatureType::init(dataset,m);
  
  nbStates = m.getNumberOfStates();
  nbSeqLabels = m.getNumberOfSequenceLabels();
  
  if(nbSeqLabels == 0)
    nbFeatures = 0;
  else {
    nbFeatures = nbStates*nbSeqLabels;
    for( i=0; i<nbSeqLabels; i++)
      nbFeaturesPerLabel[i] = nbStates;
  }
}

void HSSLabelEdgeFeatures::computeFeatureMask(iMatrix& matFeautureMask, const Model& m)
{
  int i, j, nbStates, firstOffset, lastOffset;
  
  nbStates = m.getNumberOfSequenceLabels();
  firstOffset = idOffset;
  
  for( j=0; j<nbStates; j++)
  {
    lastOffset = firstOffset + nbFeaturesPerLabel[j];
    
    for( i=firstOffset; i<lastOffset; i++)
      matFeautureMask(i,j) = 1;
    
    firstOffset += nbFeaturesPerLabel[j];
  }
}

bool HSSLabelEdgeFeatures::isEdgeFeatureType()
{
  return false;
}

