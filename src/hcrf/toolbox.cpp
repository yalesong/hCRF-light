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

#include <time.h>

#include "hcrf/toolbox.h"
#include "hcrf/optimizer.h"

Toolbox::Toolbox(bool isContinuousModel)
: bContinuousModel(isContinuousModel), weightInitType(INIT_RANDOM),
nbThreadsMP(-1), seed(0), minRangeWeights(-1.0), maxRangeWeights(1.0),
pOptimizer(NULL), pGradient(NULL), pEvaluator(NULL), pModel(NULL),
pInferenceEngine(NULL), pFeatureGenerator(NULL)
{
  // Create model and feature generator right away.
  // They don't depend on anything and we might need them before the call to initToolbox.
  pModel = new Model();
  pFeatureGenerator = new FeatureGenerator();
}

void Toolbox::initToolbox()
{}

Toolbox::~Toolbox()
{
  if(pOptimizer)
  {
    delete pOptimizer;
    pOptimizer = NULL;
  }
  if(pGradient)
  {
    delete pGradient;
    pGradient = NULL;
  }
  if(pEvaluator)
  {
    delete pEvaluator;
    pEvaluator = NULL;
  }
  if(pModel)
  {
    delete pModel;
    pModel = NULL;
  }
  if(pInferenceEngine)
  {
    delete pInferenceEngine;
    pInferenceEngine = NULL;
  }
  if(pFeatureGenerator)
  {
    delete pFeatureGenerator;
    pFeatureGenerator = NULL;
  }
}

void Toolbox::train(DataSet &X, bool bInitWeights)
{
  initToolbox();
  if(bInitWeights)
  {
    initModel(X);
    initWeights(X);
  }
  
  pGradient->setMaxNumberThreads(nbThreadsMP);
  pEvaluator->setMaxNumberThreads(nbThreadsMP);
  
  //Start training
  pOptimizer->optimize(pModel, &X, pEvaluator, pGradient);
}


void Toolbox::setMaxNumberThreads(int maxThreads)
{
  if (nbThreadsMP < maxThreads)
  {
    nbThreadsMP = maxThreads;
  }
}

double Toolbox::computeError(DataSet& X)
{
  return pEvaluator->computeError(&X, pModel);
}


void Toolbox::setRangeWeights(double minRange, double maxRange)
{
  minRangeWeights = minRange;
  maxRangeWeights = maxRange;
}

double Toolbox::getMinRangeWeights()
{
  return minRangeWeights;
}

double Toolbox::getMaxRangeWeights()
{
  return maxRangeWeights;
}

void Toolbox::setMinRangeWeights(double minRange)
{
  minRangeWeights = minRange;
}

void Toolbox::setMaxRangeWeights(double maxRange)
{
  maxRangeWeights = maxRange;
}

int Toolbox::getWeightInitType()
{
  return weightInitType;
}

void Toolbox::setWeightInitType(int initType)
{
  weightInitType = initType;
}

void Toolbox::setRandomSeed(long in_seed)
{
  seed = in_seed;
}

long Toolbox::getRandomSeed()
{
  return seed;
}

void Toolbox::setInitWeights(const dVector& w)
{
  setWeightInitType(INIT_PREDEFINED);
  initW.set(w);
}

void Toolbox::setWeights(const dVector& w)
{
  pModel->setWeights(w);
}

dVector& Toolbox::getInitWeights()
{
  return initW;
}

void Toolbox::addFeatureFunction(int featureFunctionId, int iParam1, int iParam2)
{
  if (!pFeatureGenerator)
    pFeatureGenerator = new FeatureGenerator();
  
  if (!getModel()->isMultiViewMode() && pFeatureGenerator->getFeatureById(featureFunctionId) != NULL)
    throw HcrfBadModel("Toolbox::addFeatureFunction() - Feature already added.");
  
  switch(featureFunctionId)
  {
    case RAW_FEATURE_ID:
      pFeatureGenerator->addFeature(new RawFeatures());
      break;
    case EDGE_FEATURE_ID:
      pFeatureGenerator->addFeature(new EdgeFeatures());
      break;
    case LABEL_EDGE_FEATURE_ID:
      pFeatureGenerator->addFeature(new LabelEdgeFeatures());
      break;
    case GATE_NODE_FEATURE_ID:
      pFeatureGenerator->addFeature(new GateNodeFeatures(iParam1, iParam2));
      break;
    case MV_GAUSSIAN_WINDOW_RAW_FEATURE_ID:
      pFeatureGenerator->addFeature(new GaussianWindowRawFeaturesMV(iParam1, iParam2));
      break;
    case MV_EDGE_FEATURE_ID:
      pFeatureGenerator->addFeature(new EdgeFeaturesMV(iParam1, iParam2));
      break;
    case MV_LABEL_EDGE_FEATURE_ID:
      pFeatureGenerator->addFeature(new LabelEdgeFeaturesMV(iParam1));
      break;
    default:
      break;
  }
}

void Toolbox::setInferenceEngine(InferenceEngine* engine)
{
  pInferenceEngine = engine;
  if (pEvaluator)
    pEvaluator->setInferenceEngine(engine);
  if (pGradient)
    pGradient->setInferenceEngine(engine);
}

void Toolbox::setOptimizer(int opt)
{
  if(opt == OPTIMIZER_LBFGS) {
    pOptimizer = new OptimizerLBFGS();
    pModel->setMaxMargin(false);
  }
  else if(opt == OPTIMIZER_NRBM) {
#if USENRBM
    pOptimizer = new OptimizerNRBM();
    pModel->setMaxMargin(true);
#else
    throw InvalidOptimizer("NRBM is disabled.");
#endif
  }
  else {
    throw InvalidOptimizer("Invalid optimizer specified");
  }
}

void Toolbox::initWeights(DataSet &X)
{
  initWeightsRandom();
}

void Toolbox::initWeightsRandom()
{
  if (seed==0)
    srand( (unsigned)time( NULL ) );
  else
    srand(seed);
  
  dVector w(pFeatureGenerator->getNumberOfFeatures());
  double widthRangeWeight = fabs(maxRangeWeights - minRangeWeights);
  for(int i = 0; i < w.getLength(); i++)
    w.setValue(i,(((double)rand())/(double)RAND_MAX)*widthRangeWeight+minRangeWeights);
  pModel->setWeights(w);
}



int Toolbox::getMaxNbIteration()
{
  if(pOptimizer)
    return pOptimizer->getMaxNumIterations();
  else
    return 0;
}

double Toolbox::getRegularizationL2()
{
  if(pModel)
    return pModel->getRegL2Sigma();
  else
    return 0.0;
}

double Toolbox::getRegularizationL1()
{
  if(pModel)
    return pModel->getRegL1Sigma();
  else
    return 0.0;
}

void Toolbox::setMaxNbIteration(int maxit)
{
  if(pOptimizer)
    pOptimizer->setMaxNumIterations(maxit);
}

void Toolbox::setRegularizationL2(double regFactorL2, eFeatureTypes typeFeature)
{
  if(pModel)
    pModel->setRegL2Sigma(regFactorL2, typeFeature);
}

void Toolbox::setRegularizationL1(double regFactorL1, eFeatureTypes typeFeature)
{
  if(pModel)
    pModel->setRegL1Sigma(regFactorL1, typeFeature);
}

int Toolbox::getDebugLevel()
{
  if(pModel)
    return pModel->getDebugLevel();
  else
    return -1;
}
void Toolbox::setDebugLevel(int newDebugLevel)
{
  if(pModel)
    pModel->setDebugLevel(newDebugLevel);
}

Model* Toolbox::getModel()
{
  return pModel;
}

FeatureGenerator* Toolbox::getFeatureGenerator()
{
  return pFeatureGenerator;
}

Optimizer* Toolbox::getOptimizer()
{
  return pOptimizer;
}

void Toolbox::load(char* filenameModel, char* filenameFeatures)
{
  pFeatureGenerator->load(filenameFeatures);
  pModel->load(filenameModel);
}

void Toolbox::save(char* filenameModel, char* filenameFeatures)
{
  pModel->save(filenameModel);
  pFeatureGenerator->save(filenameFeatures);
}

void Toolbox::validate(DataSet& dataTrain, DataSet& dataValidate, 
  double& optimalRegularisation, char* filenameStats)
{
  double MaxF1value = 0.0;
  for(int r = -1; r <= 2;r ++)
  {
    double regFactor = pow(10.0,r);
    setRegularizationL2(regFactor);
    train(dataTrain);
    double F1Value = test(dataValidate,NULL,filenameStats);
    if(F1Value > MaxF1value)
    {
      MaxF1value = F1Value;
      optimalRegularisation = regFactor;
    }
  }
}

featureVector* Toolbox::getAllFeatures(DataSet &X)
{
  if(pFeatureGenerator->getNumberOfFeatures() == 0)
    initModel(X);
  return pFeatureGenerator->getAllFeatures(pModel,X.getNumberofRawFeatures());
}

#ifdef _OPENMP
void Toolbox::set_num_threads(int nt){
  omp_set_num_threads(nt);
}
#else
void Toolbox::set_num_threads(int){
  //Do nothing if not OpenMP
}
#endif
