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

#include "hcrf/toolbox.h"

ToolboxHSSHCRF::ToolboxHSSHCRF(): ToolboxHCRF() {}

ToolboxHSSHCRF::ToolboxHSSHCRF(int nbHiddenStates)
: ToolboxHCRF(), numberOfHiddenStates(nbHiddenStates) {}

ToolboxHSSHCRF::~ToolboxHSSHCRF() {}

void ToolboxHSSHCRF::initToolbox()
{
  // Create components
  if(!pFeatureGenerator)
    pFeatureGenerator = new FeatureGenerator();
  if(!pOptimizer)
    pOptimizer = new OptimizerLBFGS();
  if(!pInferenceEngine)
    setInferenceEngine(new InferenceEngineFB());
  if(!pGradient)
    pGradient = new GradientHSSHCRF (pInferenceEngine, pFeatureGenerator);
  if(!pEvaluator)
    pEvaluator = new EvaluatorHSSHCRF (pInferenceEngine, pFeatureGenerator);
  
  // Add features
  // Note: features should be added layer-wise; otherwise the program will break.
  //       see FeatureGenerator::initFeatures(...) for the reason
  int level, nbGates, wndSize = 0;
  nbGates = getModel()->getNbGates();
  if(!pFeatureGenerator->getFeatureByBasicType(NODE_FEATURE) &&
     !pFeatureGenerator->getFeatureByBasicType(LABEL_EDGE_FEATURE) &&
     !pFeatureGenerator->getFeatureByBasicType(EDGE_FEATURE) ) {
    for( level=0; level<pModel->getMaxFeatureLayer(); level++ ) {
      if( nbGates>0 )
        pFeatureGenerator->addFeature(new HSSGateNodeFeatures(level,nbGates,wndSize));
      else
        pFeatureGenerator->addFeature(new HSSRawFeatures(level));
      pFeatureGenerator->addFeature(new HSSLabelEdgeFeatures(level));
      pFeatureGenerator->addFeature(new HSSEdgeFeatures(level));
    }
  }
}

void ToolboxHSSHCRF::initModel(DataSet &X)
{
  // Find number of states and initialize Model
  pModel->setNumberOfStates(numberOfHiddenStates);
  pModel->setNumberOfSequenceLabels(X.searchNumberOfSequenceLabels());
  
  // Initialize feature generator
  pFeatureGenerator->initFeatures(X,*pModel);
}

void ToolboxHSSHCRF::train(DataSet &X, bool bInitWeights)
{
  initToolbox();
  if(bInitWeights) {
    initModel(X);
    initWeights(X);
  }
  
  for(int i=0; i<(int)X.size(); i++) {
    std::vector<std::vector<int> > groupLabelSet;
    for(int j=0; j<X.at(i)->length(); j++) {
      std::vector<int> groupLabel;
      groupLabel.push_back(j);
      groupLabelSet.push_back(groupLabel);
    }
    X.at(i)->getDeepSeqGroupLabels()->push_back(groupLabelSet);
  }
  
  // Incremental Optimization. Train layer-by-layer, bottom-up
  for(int layer=0; layer<pModel->getMaxFeatureLayer(); layer++) {
    pModel->setCurrentFeatureLayer(layer);
    pOptimizer->optimizeBlock(pModel, &X, pEvaluator, pGradient);
    if(layer < pModel->getMaxFeatureLayer()-1)
      buildDeepSequence(X, layer);
  }
}

void ToolboxHSSHCRF::buildDeepSequence(DataSet& X, int cur_layer)
{
  int i,y,dimY,num_ccs,new_layer;
  int *labels = 0;
  std::vector<std::vector<int> > super_x;
  
  dimY = pModel->getNumberOfSequenceLabels();
  new_layer = cur_layer+1;
  
  segment segmenter;
  DataSequence *x;
  std::vector<Beliefs> condBeliefs(dimY);
  
  for( i=0; i<(int)X.size(); i++ )
  {
    x=X.at(i);
    
    // Skip this sequence if there was no grouping at the last iteration.
    if( new_layer != x->getDeepSeqGroupLabels()->size() ) continue;
    super_x = x->getDeepSeqGroupLabels()->at(cur_layer);
    
    for( y=0; y<dimY; y++ )
      pInferenceEngine->computeBeliefs(condBeliefs[y],pFeatureGenerator,x,pModel,true,y);
    
    labels = new int[(int)super_x.size()];
    segmenter.segment_sequence(condBeliefs,pModel->getSegmentConst(),&num_ccs,&labels);
    
    // If no segmentation, continue to the next sequence
    if( num_ccs==1 || num_ccs==(int)super_x.size() )
      continue;
    
    // Record group labels
    std::vector<std::vector<int> > groupLabelSet; // group label set for the new layer
    
    // create empty label sets
    for(int t=0; t<num_ccs; t++) {
      std::vector<int> groupLabel;
      groupLabelSet.push_back(groupLabel);
    }
    // create union sets
    for(int t=0; t<(int)super_x.size(); t++)
      for(int k=0; k<(int)super_x[t].size(); k++)
        groupLabelSet.at(labels[t]).push_back(super_x[t].at(k));
    x->getDeepSeqGroupLabels()->push_back(groupLabelSet);
    
    delete[] labels; labels=0;
  }
}
