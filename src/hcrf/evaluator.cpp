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

#include <assert.h>

#include "hcrf/evaluator.h"

using namespace std;

/////////////////////////////////////////////////////////////////////
// Evaluator Class
/////////////////////////////////////////////////////////////////////

// *
// Constructor and Destructor
// *

Evaluator::Evaluator(): pInfEngine(NULL), pFeatureGen(NULL), nbThreadsMP(1)
{}

Evaluator::Evaluator(InferenceEngine* infEngine, FeatureGenerator* featureGen)
:pInfEngine(infEngine), pFeatureGen(featureGen), nbThreadsMP(1)
{}

Evaluator::~Evaluator()
{
  pInfEngine = NULL;
  pFeatureGen = NULL;
}

Evaluator::Evaluator(const Evaluator& other)
:pInfEngine(other.pInfEngine), pFeatureGen(other.pFeatureGen), nbThreadsMP(other.nbThreadsMP)
{}

Evaluator& Evaluator::operator=(const Evaluator& other)
{
  nbThreadsMP = other.nbThreadsMP;
  pInfEngine = other.pInfEngine;
  pFeatureGen = other.pFeatureGen;
  return *this;
}

// *
// Public Methods
// *

void Evaluator::setMaxNumberThreads(int maxThreads)
{
  if (nbThreadsMP < maxThreads)
    nbThreadsMP = maxThreads;
}

void Evaluator::init(InferenceEngine* infEngine, FeatureGenerator* featureGen)
{
  pInfEngine = infEngine;
  pFeatureGen = featureGen;
}

double Evaluator::computeError(DataSet* X, Model* m)
{
  if(!pInfEngine || !pFeatureGen){
    throw HcrfBadPointer("In Evaluator::computeError(DataSet*, model*)");
  }
  
  int TID, nbIters = (int)X->size();
  double val = 0.0;
  
  // Initialize the buffers (vecFeaturesMP) for each thread
#ifdef _OPENMP
  if( nbThreadsMP < 1 ) nbThreadsMP = omp_get_max_threads();
  setMaxNumberThreads(nbThreadsMP);
  pInfEngine->setMaxNumberThreads(nbThreadsMP);
  pFeatureGen->setMaxNumberThreads(nbThreadsMP);
#endif
#pragma omp parallel default(none) private(TID) \
  shared(std::cout, X, m, nbIters, val)
  { // BEGIN parallel region
#ifdef _OPENMP
    TID = omp_get_thread_num();
#else
    TID = 0;
#endif
#pragma omp for schedule(dynamic) reduction(+:val)
    for(int i=0; i<nbIters; i++)
    {
      if (m->getDebugLevel()>=4)
#pragma omp critical(output)
        printf("Thread %d computing error for sequence %d out of %d (Size: %d)\n",
               TID, i, (int)X->size(), X->at(i)->length());
      val+= this->computeError(X->at(i), m) * X->at(i)->getWeightSequence();
    }
  } // END parallel region
  
  if(!m->isMaxMargin() && m->getRegL2Sigma() != 0.0f)
  {
    double weightNorm = m->getWeights()->l2Norm(false);
    val += weightNorm / (2.0*m->getRegL2Sigma()*m->getRegL2Sigma());
  }
  return val;
}


void Evaluator::computeStateLabels(DataSequence* X, Model* m, iVector* ystar, dMatrix* pystar)
{
  if(!pInfEngine || !pFeatureGen)
    throw HcrfBadPointer("In Evaluator::computeStateLabels");
  
  Beliefs bel;
  pInfEngine->computeBeliefs(bel, pFeatureGen, X, m, 0, -1, false);
  computeLabels(bel,ystar,pystar);
}

int Evaluator::computeSequenceLabel(DataSequence* , Model* , dVector *, dMatrix *)
{
  /* This is only valide for model that have a sequence label, such as HCRF */
  if(!pInfEngine || !pFeatureGen){
    throw HcrfBadPointer("In Evaluator::computeSequenceLabel");
  }
  // To be implemented
  throw HcrfNotImplemented("Evaluator::computeSequenceLabel");
}


// *
// Private Methods
// *
void Evaluator::computeLabels(Beliefs& bel, iVector* ystar, dMatrix * pystar)
{
  int nbNodes, nbStates, xi, yi;
  double max_val;
  
  nbNodes = (int)bel.belStates.size();
  nbStates = 0;
  if(nbNodes > 0)
    nbStates = bel.belStates[0].getLength();
  
  ystar->create(nbNodes);
  if(pystar)
    pystar->create(nbNodes,nbStates);
  
  // Viterbi decoding
  for( xi=0; xi<nbNodes; xi++) {
    ystar->setValue(xi, 0);
    max_val = bel.belStates[xi][0];
    pystar->setValue(0, xi, bel.belStates[xi][0]);
    for( yi=1; yi<nbStates; yi++) {
      pystar->setValue(yi, xi, bel.belStates[xi][yi]);
      if(max_val < bel.belStates[xi][yi]) {
        ystar->setValue(xi,yi);
        max_val = bel.belStates[xi][yi];
      }
    }
  }
}
