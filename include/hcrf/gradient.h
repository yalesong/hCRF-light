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

#ifndef GRADIENT_H
#define GRADIENT_H

#include <vector>
#include <assert.h>

#include "hcrf/dataset.h"
#include "hcrf/model.h"
#include "hcrf/inferenceengine.h"
#include "hcrf/featuregenerator.h"
#include "hcrf/evaluator.h"
#include "hcrf/segment.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

class Gradient {
public:
  Gradient(InferenceEngine* infEngine, FeatureGenerator* featureGen);
  Gradient(const Gradient&);
  Gradient& operator=(const Gradient&);
  virtual double computeGradient(dVector& vecGradient, Model* m, DataSequence* X) = 0;
  virtual double computeGradient(dVector& vecGradient, Model* m, DataSet* X);
  virtual ~Gradient();
  virtual void setMaxNumberThreads(int maxThreads);
  void setInferenceEngine(InferenceEngine* inEngine){pInfEngine = inEngine;}
  
  virtual void viterbiDecoding(Beliefs bel, iVector& ystar, dMatrix& pystar);
protected:
  InferenceEngine* pInfEngine;
  FeatureGenerator* pFeatureGen;
  int nbThreadsMP;
};

class GradientCRF : public Gradient
{
public:
  GradientCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
  double computeGradient(dVector& vecGradient, Model* m, DataSequence* X);
  using Gradient::computeGradient;
private:
  double computeGradientMLE(dVector& vecGradient, Model* m, DataSequence* X);
  double computeGradientMaxMargin(dVector& vecGradient, Model* m, DataSequence* X);
};

class GradientHCRF : public Gradient
{
public:
  GradientHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
  double computeGradient(dVector& vecGradient, Model* m, DataSequence* X);
  using Gradient::computeGradient;
private:
  double computeGradientMLE(dVector& vecGradient, Model* m, DataSequence* X);
  double computeGradientMaxMargin(dVector& vecGradient, Model* m, DataSequence* X);
};

class GradientHSSHCRF : public Gradient
{
public:
  GradientHSSHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
  double computeGradient(dVector& vecGradient, Model* m, DataSequence* X);
  using Gradient::computeGradient;
private:
  double computeGradientMLE(dVector& vecGradient, Model* m, DataSequence* X);
};

class GradientLDCRF : public Gradient
{
public:
  GradientLDCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
  double computeGradient(dVector& vecGradient, Model* m, DataSequence* X);
};

class GradientMVHCRF: public Gradient
{
public:
  GradientMVHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
  double computeGradient(dVector& vecGradient, Model* m, DataSequence* X);
  using Gradient::computeGradient;
  
private:
  double computeGradientMLE(dVector& vecGradient, Model* m, DataSequence* X);
};

class GradientMVLDCRF: public Gradient
{
public:
  GradientMVLDCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
  double computeGradient(dVector& vecGradient, Model* m, DataSequence* X);
  using Gradient::computeGradient;
private:
  double computeGradientMLE(dVector& vecGradient, Model* m, DataSequence* X);
};

class GradientOCCRF: public Gradient
{
public:
  GradientOCCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
  double computeGradient(dVector& vecGradient, Model* m, DataSequence* X);
  using Gradient::computeGradient;
  double mle(dVector& vecGradient, Model* m, DataSequence* X);
  double max_margin(dVector& vecGradient, Model* m, DataSequence* X);
};

class GradientOCHCRF: public Gradient
{
public:
  GradientOCHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
  double computeGradient(dVector& vecGradient, Model* m, DataSequence* X);
  using Gradient::computeGradient;
private:
  double mle(dVector& vecGradient, Model* m, DataSequence* X);
  double max_margin(dVector& vecGradient, Model* m, DataSequence* X);
};

#endif
