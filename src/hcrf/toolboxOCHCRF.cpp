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

using namespace std;
ToolboxOCHCRF::ToolboxOCHCRF():ToolboxHCRF() {}
ToolboxOCHCRF::ToolboxOCHCRF(int nbHiddenStates):ToolboxHCRF(), numberOfHiddenStates(nbHiddenStates) {}
ToolboxOCHCRF::~ToolboxOCHCRF() {}

void ToolboxOCHCRF::initToolbox()
{
  if(!pFeatureGenerator)
    pFeatureGenerator = new FeatureGenerator();
  if(!pInferenceEngine)
    setInferenceEngine(new InferenceEngineFB());
  if(!pOptimizer)
    setOptimizer(OPTIMIZER_LBFGS);
  if(!pGradient)
    pGradient = new GradientOCHCRF(pInferenceEngine, pFeatureGenerator);
  if(!pEvaluator)
    pEvaluator = new EvaluatorOCHCRF(pInferenceEngine, pFeatureGenerator);
  
  if(!pFeatureGenerator->getFeatureByBasicType(NODE_FEATURE)) {
    if( getModel()->getNbGates()>0 )
      pFeatureGenerator->addFeature(new GateNodeFeatures(getModel()->getNbGates()));
    else
      pFeatureGenerator->addFeature(new RawFeatures);
  }
  if(!pFeatureGenerator->getFeatureByBasicType(EDGE_FEATURE))
    pFeatureGenerator->addFeature(new EdgeFeatures());
  if(!pFeatureGenerator->getFeatureByBasicType(LABEL_EDGE_FEATURE))
    pFeatureGenerator->addFeature(new LabelEdgeFeatures());
}

void ToolboxOCHCRF::initModel(DataSet &X)
{
  // Find number of states and initialize Model
  pModel->setNumberOfStates(numberOfHiddenStates);
  pModel->setNumberOfSequenceLabels(2);
  // Initialize feature generator
  pFeatureGenerator->initFeatures(X,*pModel);
}


void ToolboxOCHCRF::train(DataSet &X, bool bInitWeights)
{
  initToolbox();
  if(bInitWeights)
  {
    initModel(X);
    initWeights(X);
  }
  //Start training
  pOptimizer->optimize(pModel, &X, pEvaluator, pGradient);
}

