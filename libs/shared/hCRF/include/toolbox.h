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

#ifndef __TOOLBOX_H
#define __TOOLBOX_H

#include "Features.h"
#include "MultiviewFeatures.h"
#include "HierarchicalFeatures.h"

#include "dataset.h"
#include "model.h"
#include "optimizer.h"
#include "gradient.h"
#include "evaluator.h"
#include "inferenceengine.h"
#include "hcrfExcep.h"
#include <map>

#ifdef _OPENMP
#include <omp.h>
#endif

enum {
   INIT_RANDOM,
   INIT_PREDEFINED
};

enum{
   OPTIMIZER_LBFGS,
   OPTIMIZER_NRBM  
};
 

//Toolbox, used to make predictions, must be trained before anything.
class Toolbox
{
public:
	Toolbox(bool isContinuousModel = false);
	Toolbox(const Toolbox&);
	Toolbox& operator=(const Toolbox&){
	   throw std::logic_error("Toolbox should not be copied");
	};
	virtual ~Toolbox();

	virtual void train(DataSet& X, bool bInitWeights = true);
	virtual double test(DataSet& X, char* filenameOutput = NULL,char* filenameStats = NULL) = 0;
	virtual void validate(DataSet& dataTrain, DataSet& dataValidate,
		double& optimalRegularisation,char* filenameStats = NULL);   

	virtual void load(char* filenameModel, char* filenameFeatures);
	virtual void save(char* filenameModel, char* filenameFeatures);
	virtual double computeError(DataSet& X);

	virtual void setMaxNumberThreads(int maxThreads);

	double getRegularizationL1();
	double getRegularizationL2();
	void setRegularizationL1(double regFactorL1, eFeatureTypes typeFeature=allTypes);
	void setRegularizationL2(double regFactorL2, eFeatureTypes typeFeature=allTypes);

	int getMaxNbIteration();
	int getWeightInitType();
	void setMaxNbIteration(int maxit);
	void setWeightInitType(int initType);

	void setRandomSeed(long seed);
	long getRandomSeed();

	void setInitWeights(const dVector& w);
	dVector& getInitWeights();

	void setWeights(const dVector& w);

	int getDebugLevel();
	void setDebugLevel(int newDebugLevel);

	bool isContinuousModel() {return bContinuousModel;}

	featureVector* getAllFeatures(DataSet &X);
	FeatureGenerator* getFeatureGenerator();
	Optimizer* getOptimizer();
	Model* getModel();

	int getLastNbIterations() {if(pOptimizer) return pOptimizer->getLastNbIterations();}

	void setRangeWeights(double minRange, double maxRange);
	void setMinRangeWeights(double minRange);
	void setMaxRangeWeights(double maxRange);
	double getMinRangeWeights();
	double getMaxRangeWeights();

	void addFeatureFunction(int featureFunctionID, int iParam1 = 0, int iParam2 = 0);
	void setInferenceEngine(InferenceEngine* engine);
	void setOptimizer(int optimizerType);
	virtual void initToolbox();
	void initWeights(DataSet &X);
	void set_num_threads(int);
	virtual void initModel(DataSet &X) = 0;

protected:
	virtual void initWeightsRandom(); 

	bool bContinuousModel;
	int weightInitType;
	int nbThreadsMP;
	long seed;
	double minRangeWeights;
	double maxRangeWeights;
	dVector initW;

	Optimizer* pOptimizer;
	Gradient* pGradient;
	Evaluator* pEvaluator;
	Model* pModel;
	InferenceEngine* pInferenceEngine;
	FeatureGenerator* pFeatureGenerator;

};
 
class ToolboxCRF: public Toolbox
{
public:
	ToolboxCRF();
	virtual ~ToolboxCRF();

	virtual void initToolbox();
	virtual double test(DataSet& X, char* filenameOutput = NULL, char* filenameStats = NULL);

protected:
   virtual void initModel(DataSet &X);
};


class ToolboxHCRF: public Toolbox
{
public:
	ToolboxHCRF();
	ToolboxHCRF(int nbHiddenStates);
	virtual ~ToolboxHCRF();

	virtual double test(DataSet& X, char* filenameOutput = NULL, char* filenameStats = NULL);
	virtual void initToolbox();
	virtual void initModel(DataSet &X);

protected:
	int numberOfHiddenStates;
};

class ToolboxLDCRF : public Toolbox
{
public:
	ToolboxLDCRF();
	ToolboxLDCRF(int nbHiddenStatesPerLabel);
	virtual ~ToolboxLDCRF();

	virtual void initToolbox();
	virtual double test(DataSet& X, char* filenameOutput = NULL, char* filenameStats = NULL);
	virtual void initModel(DataSet &X);

protected:
	int numberOfHiddenStatesPerLabel; 

}; 

// Multi-view Latent Variable Models - Song et al. CVPR'12
class ToolboxMVHCRF: public ToolboxHCRF
{
public:
	ToolboxMVHCRF();

	ToolboxMVHCRF(
		eGraphTypes graphType,  
		int nbViews, 
		std::vector<int> nbHiddenStates, 
		std::vector<std::vector<int> > rawFeatureIndex);
	virtual ~ToolboxMVHCRF(); 
	
	virtual void initToolbox();

protected:
	virtual void initModel(DataSet& X);
	int m_nbViews; 
	int* m_nbHiddenStatesMultiView;
	std::vector<std::vector<int> > m_rawFeatureIndex;
	eGraphTypes m_graphType;
};

class ToolboxMVLDCRF: public ToolboxLDCRF
{
public:
	ToolboxMVLDCRF();
	ToolboxMVLDCRF(
		eGraphTypes graphType,
		int nbViews,
		std::vector<int> nbHiddenStates,
		std::vector<std::vector<int> > rawFeatureIndex);
	virtual ~ToolboxMVLDCRF();

	virtual void initToolbox();

protected:
   virtual void initModel(DataSet &X);
	int m_nbViews; 
	int* m_nbHiddenStatesMultiView;
	std::vector<std::vector<int> > m_rawFeatureIndex;
	eGraphTypes m_graphType;
};

// Hierarchical Sequence Summarization (HSS) Model -- Song et al. CVPR'13
class ToolboxHSSHCRF: public ToolboxHCRF
{
public:
	ToolboxHSSHCRF();

	//MATLAB will call this constructor.
	ToolboxHSSHCRF(int nbHiddenStates);

	virtual ~ToolboxHSSHCRF();
	virtual void train(DataSet& X, bool bInitWeights = true);
	virtual void initToolbox();

protected:
	int numberOfHiddenStates;
	virtual void initModel(DataSet &X);
	void buildDeepSequence(DataSet &X, int cur_layer);
};


#endif
