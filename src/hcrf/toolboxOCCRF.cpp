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
ToolboxOCCRF::ToolboxOCCRF():ToolboxCRF() {}
ToolboxOCCRF::~ToolboxOCCRF() {}

void ToolboxOCCRF::initToolbox()
{
  if(!pFeatureGenerator)
    pFeatureGenerator = new FeatureGenerator();
  if(!pInferenceEngine)
    setInferenceEngine(new InferenceEngineFB());
  if(!pOptimizer)
    setOptimizer(OPTIMIZER_NRBM);
  if(!pGradient)
    pGradient = new GradientOCCRF(pInferenceEngine, pFeatureGenerator);
  if(!pEvaluator)
    pEvaluator = new EvaluatorOCCRF(pInferenceEngine, pFeatureGenerator);
  
  if(!pFeatureGenerator->getFeatureByBasicType(NODE_FEATURE))
    pFeatureGenerator->addFeature(new RawFeatures);
  
  if(!pFeatureGenerator->getFeatureByBasicType(EDGE_FEATURE))
    pFeatureGenerator->addFeature(new EdgeFeatures());
}

void ToolboxOCCRF::initModel(DataSet &X)
{
  pModel->setNumberOfStates(2);
  pModel->setNumberOfStateLabels(2);
  
  int dimX = 0;
  if((*X.begin())->getPrecomputedFeatures() != NULL)
    dimX+= (*X.begin())->getPrecomputedFeatures()->getHeight();
  pModel->setNumberOfRawFeaturesPerFrame(dimX);
  
  // Initialize feature generator
  pFeatureGenerator->initFeatures(X,*pModel);
}

void ToolboxOCCRF::train(DataSet &X, bool bInitWeights)
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
