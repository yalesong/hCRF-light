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


#include "hcrf/features.h"

using namespace std;


///////////////////////////////////////////////////////////////////////////////////
// RawFeatures
//
RawFeatures::RawFeatures():FeatureType()
{
  strFeatureTypeName = "Raw Feature Type";
  featureTypeId = RAW_FEATURE_ID;
  basicFeatureType = NODE_FEATURE;
}

void RawFeatures::getFeatures(featureVector& listFeatures, DataSequence* X, Model* m,
                              int nodeIndex, int prevNodeIndex, int seqLabel)
{
  if(X->getPrecomputedFeatures() != NULL && prevNodeIndex == -1)
  {
    dMatrix * preFeatures = X->getPrecomputedFeatures();
    int nbFeatures = preFeatures->getHeight();
    feature* pFeature;
    int idState = 0;
    int nbStateLabels = m->getNumberOfStates();
    
    for(int s = 0; s < nbStateLabels; s++)
    {
      for(int f = 0; f < nbFeatures; f++)
      {
        pFeature = listFeatures.addElement();
        pFeature->id = getIdOffset(seqLabel) + f + idState*nbFeatures;
        pFeature->globalId = getIdOffset() + f + idState*nbFeatures;
        pFeature->nodeIndex = nodeIndex;
        pFeature->nodeState = s;
        pFeature->prevNodeIndex = -1;
        pFeature->prevNodeState = -1;
        pFeature->sequenceLabel = seqLabel;
        pFeature->value = preFeatures->getValue(f,nodeIndex);
      }
      idState ++;
    }
  }
}

void RawFeatures::getAllFeatures(featureVector& listFeatures, Model* m, int NbRawFeatures)
{
  int nbStateLabels = m->getNumberOfStates();
  feature* pFeature;
  
  for(int s = 0; s < nbStateLabels; s++)
  {
    for(int f = 0; f < NbRawFeatures; f++)
    {
      pFeature = listFeatures.addElement();
      pFeature->id = getIdOffset() + f + s*NbRawFeatures;
      pFeature->globalId = getIdOffset() + f + s*NbRawFeatures;
      pFeature->nodeIndex = featureTypeId;
      pFeature->nodeState = s;
      pFeature->prevNodeIndex = -1;
      pFeature->prevNodeState = -1;
      pFeature->sequenceLabel = -1;
      pFeature->value = f;
    }
  }
}



void RawFeatures::init(const DataSet& dataset, const Model& m)
{
  FeatureType::init(dataset,m);
  if(dataset.size() > 0)
  {
    int nbStates = m.getNumberOfStates();
    int nbSeqLabels = m.getNumberOfSequenceLabels();
    int nbFeaturesPerStates = dataset.at(0)->getPrecomputedFeatures()->getHeight();
    
    nbFeatures = nbStates * nbFeaturesPerStates;
    for(int i = 0; i < nbSeqLabels; i++)
      nbFeaturesPerLabel[i] = nbFeatures;
  }
}

bool RawFeatures::isEdgeFeatureType()
{
  return false;
}



///////////////////////////////////////////////////////////////////////////////////
// LabelEdgeFeatures
//
LabelEdgeFeatures::LabelEdgeFeatures():FeatureType()
{
  strFeatureTypeName = "Label Edge Feature Type";
  featureTypeId = LABEL_EDGE_FEATURE_ID;
  basicFeatureType = LABEL_EDGE_FEATURE;
}


void LabelEdgeFeatures::getFeatures(featureVector& listFeatures, DataSequence*, Model* m,
                                    int nodeIndex, int prevNodeIndex, int seqLabel)
{
  if(seqLabel != -1 && m->getNumberOfSequenceLabels() > 0 && prevNodeIndex == -1)
  {
    feature* pFeature;
    int nbStateLabels = m->getNumberOfStates();
    for(int s = 0; s < nbStateLabels; s++)
    {
      pFeature = listFeatures.addElement();
      pFeature->id = getIdOffset(seqLabel) + s;
      pFeature->globalId = getIdOffset() + s + seqLabel*nbStateLabels ;
      pFeature->nodeIndex = nodeIndex;
      pFeature->nodeState = s;
      pFeature->prevNodeIndex = -1;
      pFeature->prevNodeState = -1;
      pFeature->sequenceLabel = seqLabel;
      pFeature->value = 1.0f;
    }
  }
}

void LabelEdgeFeatures::getAllFeatures(featureVector& listFeatures, Model* m, int)
{
  if(m->getNumberOfSequenceLabels() > 0)
  {
    int nbStateLabels = m->getNumberOfStates();
    int nbSeqLabels = m->getNumberOfSequenceLabels();
    feature* pFeature;
    for(int seqLabel = 0; seqLabel < nbSeqLabels; seqLabel++)
    {
      for(int s = 0; s < nbStateLabels; s++)
      {
        pFeature = listFeatures.addElement();
        pFeature->id = getIdOffset() + s + seqLabel*nbStateLabels ;
        pFeature->globalId = getIdOffset() + s + seqLabel*nbStateLabels ;
        pFeature->nodeIndex = featureTypeId;
        pFeature->nodeState = s;
        pFeature->prevNodeIndex = -1;
        pFeature->prevNodeState = -1;
        pFeature->sequenceLabel = seqLabel;
        pFeature->value = 1.0f;
      }
    }
  }
}


void LabelEdgeFeatures::init(const DataSet& dataset, const Model& m)
{
  FeatureType::init(dataset,m);
  int nbStateLabels = m.getNumberOfStates();
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

void LabelEdgeFeatures::computeFeatureMask(iMatrix& matFeautureMask, const Model& m)
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

bool LabelEdgeFeatures::isEdgeFeatureType()
{
  return false;
}



///////////////////////////////////////////////////////////////////////////////////
// EdgeFeatures
//
EdgeFeatures::EdgeFeatures():FeatureType()
{
  strFeatureTypeName = "Edge Feature Type";
  featureTypeId = EDGE_FEATURE_ID;
  basicFeatureType = EDGE_FEATURE;
}

void EdgeFeatures::getFeatures(featureVector& listFeatures, DataSequence* X,
                               Model* m, int nodeIndex, int prevNodeIndex,
                               int seqLabel)
{
  // These features are only used for adjacent edge in the chain
  int nbNodes = -1;
  
  if(X->getPrecomputedFeatures())
    nbNodes = X->getPrecomputedFeatures()->getWidth();
  else
    nbNodes = (int)X->getPrecomputedFeaturesSparse()->getWidth();
  
  if( ((prevNodeIndex == nodeIndex-1) || prevNodeIndex == nodeIndex-1 + nbNodes)
     && (prevNodeIndex != -1))
  {
    feature* pFeature;
    int nbStateLabels = m->getNumberOfStates();
    for(int s1 = 0; s1 < nbStateLabels;s1++)
    {
      for(int s2 = 0; s2 < nbStateLabels;s2++)
      {
        pFeature = listFeatures.addElement();
        pFeature->id = getIdOffset(seqLabel) + s2 + s1*nbStateLabels ;
        pFeature->globalId = getIdOffset() + s2 + s1*nbStateLabels +
        seqLabel*nbStateLabels*nbStateLabels ;
        pFeature->nodeIndex = nodeIndex;
        pFeature->nodeState = s2;
        pFeature->prevNodeIndex = prevNodeIndex;
        pFeature->prevNodeState = s1;
        pFeature->sequenceLabel = seqLabel;
        pFeature->value = 1.0f;
      }
    }
  }
}

void EdgeFeatures::getAllFeatures(featureVector& listFeatures, Model* m, int)
{
  int nbStateLabels = m->getNumberOfStates();
  int nbSeqLabels = m->getNumberOfSequenceLabels();
  feature* pFeature;
  if(nbSeqLabels == 0)
    nbSeqLabels = 1;
  for(int seqLabel = 0; seqLabel < nbSeqLabels;seqLabel++)
  {
    for(int s1 = 0; s1 < nbStateLabels;s1++)
    {
      for(int s2 = 0; s2 < nbStateLabels;s2++)
      {
        pFeature = listFeatures.addElement();
        pFeature->id = getIdOffset() + s2 + s1*nbStateLabels + seqLabel*nbStateLabels*nbStateLabels ;
        pFeature->globalId = getIdOffset() + s2 + s1*nbStateLabels + seqLabel*nbStateLabels*nbStateLabels ;
        pFeature->nodeIndex = featureTypeId;
        pFeature->nodeState = s2;
        pFeature->prevNodeIndex = -1;
        pFeature->prevNodeState = s1;
        pFeature->sequenceLabel = seqLabel;
        pFeature->value = 1.0f;
      }
    }
  }
}


void EdgeFeatures::init(const DataSet& dataset, const Model& m)
{
  FeatureType::init(dataset,m);
  int nbStateLabels = m.getNumberOfStates();
  int nbSeqLabels = m.getNumberOfSequenceLabels();
  
  if(nbSeqLabels == 0)
    nbFeatures = nbStateLabels*nbStateLabels;
  else
  {
    nbFeatures = nbStateLabels*nbStateLabels*nbSeqLabels;
    for(int i = 0; i < nbSeqLabels; i++)
      nbFeaturesPerLabel[i] = nbStateLabels*nbStateLabels;
  }
}

void EdgeFeatures::computeFeatureMask(iMatrix& matFeautureMask, const Model& m)
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

bool EdgeFeatures::isEdgeFeatureType()
{
  return true;
}




///////////////////////////////////////////////////////////////////////////////////
// GateNodeFeatures
//


GateNodeFeatures::GateNodeFeatures(int nbGates, int winSize)
: FeatureType(),windowSize(winSize),nbGates(nbGates)
{
  strFeatureTypeName = "Gate Node Feature Type";
  featureTypeId = GATE_NODE_FEATURE_ID;
  basicFeatureType = NODE_FEATURE;
}

void GateNodeFeatures::getFeatures(featureVector& listFeatures,
                                   DataSequence* X, Model* m, int nodeIndex,
                                   int prevNodeIndex, int seqLabel)
{
  if( X->getPrecomputedFeatures()==NULL || prevNodeIndex!=-1 ) return;
  
  int idx,s,f,g,nbNodes,nbStates,nbSG,nbFeaturesDense,gXfpg,sXG;
  double gateSum;
  
  dMatrix *preFeatures = X->getPrecomputedFeatures();
  feature* pFeature;
  
  nbStates = m->getNumberOfStates();
  nbSG = nbStates*nbGates;
  nbFeaturesDense = preFeatures->getHeight();
  nbNodes = preFeatures->getWidth();
  
  dVector* lambda = m->getWeights();
  dMatrix* gates = X->getGateMatrix();
  
  // Precompute gate values, (row,col) = (gate,node)
  if( gates->getHeight()!=nbGates || gates->getWidth()!=nbNodes )
    gates->resize(nbNodes,nbGates);
  for( g=0; g<nbGates; g++ ) {
    gXfpg = g*(nbFeaturesDense+1); // 1: bias
    
    // gate bias
    idx = getIdOffset() + nbSG + gXfpg + nbFeaturesDense;
    gateSum = (*lambda)[idx];
    
    // gate sum
    for( f=0; f<nbFeaturesDense; f++ ) {
      idx = getIdOffset() + nbSG + gXfpg + f;
      gateSum += (*lambda)[idx] * preFeatures->getValue(f,nodeIndex);
    }
    gates->setValue(g,nodeIndex,gate(gateSum));
  }
  
  // Then compute node features from gate values
  for( s=0; s<nbStates; s++ ) {
    sXG = s*nbGates;
    for( g=0; g<nbGates; g++ ) {
      pFeature = listFeatures.addElement();
      pFeature->id = getIdOffset(seqLabel) + g + sXG;
      pFeature->globalId = getIdOffset() + g + sXG;
      pFeature->nodeIndex = nodeIndex;
      pFeature->nodeState = s;
      pFeature->prevNodeIndex = -1;
      pFeature->prevNodeState = g; // quick-and-dirty fix
      pFeature->sequenceLabel = seqLabel;
      pFeature->value = gates->getValue(g,nodeIndex);
      pFeature->featureTypeId = GATE_NODE_FEATURE_ID;
    }
  }
}


void GateNodeFeatures::getPreGateFeatures(featureVector& listFeatures,
                                          DataSequence* X, Model* m, int nodeIndex,
                                          int prevNodeIndex, int seqLabel)
{
  if( X->getPrecomputedFeatures()==NULL || prevNodeIndex!=-1 ) return;
  listFeatures.clear();
  
  int f,g,nbNodes,nbStates,nbFeaturesDense,nbSG,gXfpg;
  
  dMatrix* preFeatures = X->getPrecomputedFeatures();
  feature* pFeature;
  
  nbNodes = preFeatures->getWidth();
  nbStates = m->getNumberOfStates();
  nbFeaturesDense = preFeatures->getHeight();
  nbSG = nbStates*nbGates;
  
  for( g=0; g<nbGates; g++ ) {
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
      pFeature->value = preFeatures->getValue(f,nodeIndex);
    }
    // Gate bias.
    pFeature = listFeatures.addElement();
    pFeature->id = getIdOffset(seqLabel) + nbSG + gXfpg + nbFeaturesDense;
    pFeature->globalId = getIdOffset() + nbSG + gXfpg + nbFeaturesDense;
    pFeature->nodeIndex = nodeIndex;
    pFeature->nodeState =
    pFeature->prevNodeIndex = -1;
    pFeature->prevNodeState = g; // this is a quick and dirty fix
    pFeature->sequenceLabel = seqLabel;
    pFeature->value = 1.0;
  }
}


void GateNodeFeatures::getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures)
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
      pFeature->nodeIndex = GATE_NODE_FEATURE_ID;
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
      pFeature->nodeIndex = GATE_NODE_FEATURE_ID;
      pFeature->nodeState = -1;
      pFeature->prevNodeIndex = -1;
      pFeature->prevNodeState = -1;
      pFeature->sequenceLabel = -1;
      pFeature->value = f;
    }
  }
}

void GateNodeFeatures::init(const DataSet& dataset, const Model& m)
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

bool GateNodeFeatures::isEdgeFeatureType()
{
  return false;
}

double GateNodeFeatures::gate(double sum)
{
  return 1.0/(1.0+exp(-sum));
}
