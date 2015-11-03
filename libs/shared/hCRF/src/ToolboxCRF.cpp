#include "toolbox.h"

using namespace std;

ToolboxCRF::ToolboxCRF()
: Toolbox(true)
{}

ToolboxCRF::~ToolboxCRF()
{}

void ToolboxCRF::initToolbox()
{
	Toolbox::initToolbox();
	if(!pFeatureGenerator) pFeatureGenerator = new FeatureGenerator();
	if(!pInferenceEngine)  setInferenceEngine(new InferenceEngineFB());
	if(!pOptimizer)        pOptimizer = new OptimizerLBFGS();
	if(!pGradient)         pGradient = new GradientCRF(pInferenceEngine, pFeatureGenerator);
	if(!pEvaluator)        pEvaluator = new EvaluatorCRF(pInferenceEngine, pFeatureGenerator);

	if(!pFeatureGenerator->getFeatureByBasicType(NODE_FEATURE))
		pFeatureGenerator->addFeature(new RawFeatures);

	if(!pFeatureGenerator->getFeatureByBasicType(EDGE_FEATURE))
		pFeatureGenerator->addFeature(new EdgeFeatures());
}

void ToolboxCRF::initModel(DataSet &X)
{
	// Find number of states and initialize Model
	pModel->setNumberOfStates(X.searchNumberOfStates());
	pModel->setNumberOfStateLabels(X.searchNumberOfStates());

	int numberOfRawFeaturesPerFrame = 0;
	if((*X.begin())->getPrecomputedFeatures() != NULL)
			numberOfRawFeaturesPerFrame += (*X.begin())->getPrecomputedFeatures()->getHeight();
	pModel->setNumberOfRawFeaturesPerFrame(numberOfRawFeaturesPerFrame);

	// Initialize feature generator
	pFeatureGenerator->initFeatures(X,*pModel);
}

double ToolboxCRF::test(DataSet& X, char* filenameOutput, char* filenameStats)
{
	double returnedF1value = 0.0;
	ofstream* fileOutput = NULL;
	if(filenameOutput)
	{
		fileOutput = new ofstream(filenameOutput);
		if (!fileOutput->is_open())
		{
			delete fileOutput;
			fileOutput = NULL;
		}
	}
	ostream* fileStats = NULL;
	if(filenameStats)
	{
		fileStats = new ofstream(filenameStats, ios_base::out | ios_base::app);
		if (!((ofstream*)fileStats)->is_open())
		{
			delete fileStats;
			fileStats = NULL;
		}
	}
	if(fileStats == NULL && pModel->getDebugLevel() >= 1)
		fileStats = &cout;

	DataSet::iterator it;
	int nbStateLabels = pModel->getNumberOfStates();
	iVector seqTruePos(nbStateLabels);
	iVector seqTotalPos(nbStateLabels);
	iVector seqTotalPosDetected(nbStateLabels);

	iVector truePos(nbStateLabels);
	iVector totalPos(nbStateLabels);
	iVector totalPosDetected(nbStateLabels);

	iVector tokenPerLabel(nbStateLabels);
	iVector tokenPerLabelDetected(nbStateLabels);

	//Mostly for MATLAB modules.
	initToolbox();
	initModel(X);
 
	for(it = X.begin(); it != X.end(); it ++)
	{  
		//  Compute detected labels
		dMatrix* posterior = new dMatrix;
		iVector* vecLabels = new iVector;
		pEvaluator->computeStateLabels(*it,pModel,vecLabels, posterior);

		Beliefs bel;
		pInferenceEngine->computeBeliefs(bel, pFeatureGenerator, *it, pModel, 0);

		(*it)->setEstimatedStateLabels(vecLabels);
		(*it)->setEstimatedStatePosterior(posterior);		
		
		if( fileOutput)
		{
			//matProbabilities->write(fileOutput);
			for(int i = 0; i < (*it)->length(); i++)
			{
				(*fileOutput) << (*it)->getStateLabels(i) << "\t" << vecLabels->getValue(i);
				for(int l = 0; l < nbStateLabels; l++)
					(*fileOutput) << "\t" << posterior->getValue(l,i);			
				(*fileOutput) << endl;
			}
			(*fileOutput) << endl;
		}
		//Count state labels for the sequence
		tokenPerLabel.set(0);
		tokenPerLabelDetected.set(0);
		for(int i = 0; i < (*it)->length(); i++)
		{
			tokenPerLabel[(*it)->getStateLabels(i)]++;
			tokenPerLabelDetected[vecLabels->getValue(i)]++;

			totalPos[(*it)->getStateLabels(i)]++;
			totalPosDetected[vecLabels->getValue(i)]++;

			if(vecLabels->getValue(i) == (*it)->getStateLabels(i))
				truePos[vecLabels->getValue(i)]++;
		}
		//Find max label for the sequence
		int maxLabel = 0;
		int maxLabelDetected = 0;
		for(int j = 1 ; j < nbStateLabels ; j++) 
		{
		    if(tokenPerLabel[maxLabel] < tokenPerLabel[j])
		    	maxLabel = j;
		    if(tokenPerLabelDetected[maxLabelDetected] < tokenPerLabelDetected[j])
		    	maxLabelDetected = j;
		}
		(*it)->setEstimatedSequenceLabel(maxLabelDetected);
		// Update total of positive detections
		seqTotalPos[maxLabel]++;
		seqTotalPosDetected[maxLabelDetected]++;
		if( maxLabel == maxLabelDetected)
			seqTruePos[maxLabel]++;
	}
	// Print results
	if(fileStats)
	{
		(*fileStats) << endl << "Calculations per samples:" << endl;
		(*fileStats) << "Label\tTrue+\tMarked+\tDetect+\tPrec.\tRecall\tF1" << endl;
	}
	double prec,recall;
	int SumTruePos = 0, SumTotalPos = 0, SumTotalPosDetected = 0;
	for(int i=0 ; i<nbStateLabels ; i++) 
	{
		SumTruePos += truePos[i]; SumTotalPos += totalPos[i]; SumTotalPosDetected += totalPosDetected[i];
		prec=(totalPos[i]==0)?0:((double)truePos[i])*100.0/((double)totalPos[i]);
		recall=(totalPosDetected[i]==0)?0:((double)truePos[i])*100.0/((double)totalPosDetected[i]);
		if(fileStats)
			(*fileStats) << i << ":\t" << truePos[i] << "\t" << totalPos[i] << "\t" << totalPosDetected[i] << "\t" << prec << "\t" << recall << "\t" << 2*prec*recall/(prec+recall) << endl;
	}
	prec=(SumTotalPos==0)?0:((double)SumTruePos)*100.0/((double)SumTotalPos);
	recall=(SumTotalPosDetected==0)?0:((double)SumTruePos)*100.0/((double)SumTotalPosDetected);
	if(fileStats)
	{
		(*fileStats) << "-----------------------------------------------------------------------" << endl;
		(*fileStats) << "Ov:\t" << SumTruePos << "\t" << SumTotalPos << "\t" << SumTotalPosDetected << "\t" << prec << "\t" << recall << "\t" << 2*prec*recall/(prec+recall) << endl;
	}
	returnedF1value = 2*prec*recall/(prec+recall);

	if(fileStats)
	{
		(*fileStats) << endl << "Calculations per sequences:" << endl;
		(*fileStats) << "Label\tTrue+\tMarked+\tDetect+\tPrec.\tRecall\tF1" << endl;
		SumTruePos = 0, SumTotalPos = 0, SumTotalPosDetected = 0;
		for(int i=0 ; i<nbStateLabels ; i++) 
		{
			SumTruePos += seqTruePos[i]; SumTotalPos += seqTotalPos[i]; SumTotalPosDetected += seqTotalPosDetected[i];
			prec=(seqTotalPos[i]==0)?0:((double)(seqTruePos[i]*100000/seqTotalPos[i]))/1000;
			recall=(seqTotalPosDetected[i]==0)?0:((double)(seqTruePos[i]*100000/seqTotalPosDetected[i]))/1000;
			(*fileStats) << i << ":\t" << seqTruePos[i] << "\t" << seqTotalPos[i] << "\t" << seqTotalPosDetected[i] << "\t" << prec << "\t" << recall << "\t" << 2*prec*recall/(prec+recall) << endl;
		}
		prec=(SumTotalPos==0)?0:((double)SumTruePos)*100.0/((double)SumTotalPos);
		recall=(SumTotalPosDetected==0)?0:((double)SumTruePos)*100.0/((double)SumTotalPosDetected);
		(*fileStats) << "-----------------------------------------------------------------------" << endl;
		(*fileStats) << "Ov:\t" << SumTruePos << "\t" << SumTotalPos << "\t" << SumTotalPosDetected << "\t" << prec << "\t" << recall << "\t" << 2*prec*recall/(prec+recall) << endl;
	}

	if( fileOutput )
	{
		fileOutput->close();
		delete fileOutput;
	}
	if(fileStats != &cout && fileStats != NULL)
	{
		((ofstream*)fileStats)->close();
		delete fileStats;
	}
	return returnedF1value;
}
