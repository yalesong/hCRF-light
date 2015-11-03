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

#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <list>

#include "featuregenerator.h"
#include "inferenceengine.h"
#include "dataset.h"
#include "model.h"
#include "hcrfExcep.h"
#include "segment.h"

#ifdef _OPENMP
#include <omp.h>
#endif

class Evaluator 
{
  public:
    Evaluator();
    Evaluator(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    virtual ~Evaluator();
    Evaluator(const Evaluator& other);
    Evaluator& operator=(const Evaluator& other);
    void init(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    virtual double computeError(DataSequence* X, Model* m) = 0;
    virtual double computeError(DataSet* X, Model* m);
    virtual void computeStateLabels(DataSequence* X, Model* m, iVector* vecStateLabels, dMatrix * prob = NULL);
    virtual int computeSequenceLabel(DataSequence* X, Model* m, dVector* prob, dMatrix* hprob = NULL);
	virtual void setMaxNumberThreads(int maxThreads);
	void setInferenceEngine(InferenceEngine* infEngine){pInfEngine = infEngine;}
    
  protected:
    InferenceEngine* pInfEngine;
    FeatureGenerator* pFeatureGen;
    void computeLabels(Beliefs& bel, iVector* vecStateLabels, dMatrix * prob = NULL);
    friend class OptimizerLBFGS;
	int nbThreadsMP;
};

class EvaluatorCRF:public Evaluator
{
  public:
    EvaluatorCRF();
    EvaluatorCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    ~EvaluatorCRF();
    double computeError(DataSequence* X, Model* m);
    virtual double computeError(DataSet* X, Model * m){
        return Evaluator::computeError(X,m);
    }
};

class EvaluatorHCRF:public Evaluator
{
  public:
    EvaluatorHCRF();
    EvaluatorHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    ~EvaluatorHCRF();
    double computeError(DataSequence* X, Model* m);
    virtual double computeError(DataSet* X, Model * m){
        return Evaluator::computeError(X,m);
    }
    int computeSequenceLabel(DataSequence* X, Model* m, dVector * prob, dMatrix* hprob = NULL);
};

class EvaluatorLDCRF:public Evaluator
{
  public:
    EvaluatorLDCRF();
    EvaluatorLDCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    ~EvaluatorLDCRF();
    virtual double computeError(DataSequence* X, Model* m);
    virtual double computeError(DataSet* X, Model * m){
        return Evaluator::computeError(X,m);
    }
    void computeStateLabels(DataSequence* X, Model* m, iVector* vecStateLabels, dMatrix * prob = NULL);
};
 
class EvaluatorMVHCRF:public Evaluator
{
  public:
    EvaluatorMVHCRF();
    EvaluatorMVHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    ~EvaluatorMVHCRF();
    double computeError(DataSequence* X, Model* m);
    virtual double computeError(DataSet* X, Model * m){
        return Evaluator::computeError(X,m);
    }

    int computeSequenceLabel(DataSequence* X, Model* m, dVector* prob, dMatrix* hprob = NULL);
};

class EvaluatorMVLDCRF:public Evaluator
{
  public:
    EvaluatorMVLDCRF();
    EvaluatorMVLDCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    ~EvaluatorMVLDCRF();
    double computeError(DataSequence* X, Model* m);
    virtual double computeError(DataSet* X, Model * m){
        return Evaluator::computeError(X,m);
    }

    void computeStateLabels(DataSequence* X, Model* m, iVector* vecStateLabels, dMatrix * prob = NULL);
};

class EvaluatorHSSHCRF:public Evaluator
{
  public:
    EvaluatorHSSHCRF();
    EvaluatorHSSHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    ~EvaluatorHSSHCRF();
    double computeError(DataSequence* X, Model* m);
    virtual double computeError(DataSet* X, Model * m){
        return Evaluator::computeError(X,m);
    }

    int computeSequenceLabel(DataSequence* X, Model* m, dVector* prob, dMatrix* hprob = NULL);
};

#endif 

