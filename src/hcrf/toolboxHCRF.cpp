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

#include "hcrf/toolbox.h"

ToolboxHCRF::ToolboxHCRF()
:Toolbox(false)
{
}

ToolboxHCRF::ToolboxHCRF(int nbHiddenStates)
:Toolbox(false),numberOfHiddenStates(nbHiddenStates)
{}

ToolboxHCRF::~ToolboxHCRF()
{}

void ToolboxHCRF::initToolbox()
{
  if(!pFeatureGenerator) pFeatureGenerator = new FeatureGenerator();
  if(!pInferenceEngine)  setInferenceEngine(new InferenceEngineFB());
  if(!pOptimizer)        pOptimizer = new OptimizerLBFGS();
  if(!pGradient)         pGradient = new GradientHCRF(pInferenceEngine, pFeatureGenerator);
  if(!pEvaluator)        pEvaluator = new EvaluatorHCRF(pInferenceEngine, pFeatureGenerator);
  
  if(!pFeatureGenerator->getFeatureByBasicType(NODE_FEATURE))
  {
    if( getModel()->getNbGates()>0 )
      pFeatureGenerator->addFeature(new GateNodeFeatures(getModel()->getNbGates()));
    else
      pFeatureGenerator->addFeature(new RawFeatures);
  }
  if (!pFeatureGenerator->getFeatureByBasicType(LABEL_EDGE_FEATURE))
    pFeatureGenerator->addFeature(new LabelEdgeFeatures());
  
  if(!pFeatureGenerator->getFeatureByBasicType(EDGE_FEATURE))
    pFeatureGenerator->addFeature(new EdgeFeatures());
}

void ToolboxHCRF::initModel(DataSet &X)
{
  // Find number of states and initialize Model
  pModel->setNumberOfStates(numberOfHiddenStates);
  pModel->setNumberOfSequenceLabels(X.searchNumberOfSequenceLabels());
  
  // Initialize feature generator
  pFeatureGenerator->initFeatures(X,*pModel);
}

double ToolboxHCRF::test(DataSet& X, char* filenameOutput, char* filenameStats)
{
  double returnedF1value = 0.0;
  std::ofstream* fileOutput = NULL;
  if(filenameOutput)
  {
    fileOutput = new std::ofstream(filenameOutput);
    if (!fileOutput->is_open())
    {
      delete fileOutput;
      fileOutput = NULL;
    }
  }
  std::ostream* fileStats = NULL;
  if(filenameStats)
  {
    fileStats = new std::ofstream(filenameStats, std::ios_base::out | std::ios_base::app);
    if (!((std::ofstream*)fileStats)->is_open())
    {
      delete fileStats;
      fileStats = NULL;
    }
  }
  if(fileStats == NULL && pModel->getDebugLevel() >= 1)
    fileStats = &std::cout;
  
  DataSet::iterator it;
  int nbSeqLabels = pModel->getNumberOfSequenceLabels();
  iVector seqTruePos(nbSeqLabels);
  iVector seqTotalPos(nbSeqLabels);
  iVector seqTotalPosDetected(nbSeqLabels);
  
  for(it = X.begin(); it != X.end(); it++)
  {
    //  Compute detected label
    dVector* posterior = new dVector;
    dMatrix* hposterior = new dMatrix;
    int y_star = pEvaluator->computeSequenceLabel(*it,pModel,posterior,hposterior);
    (*it)->setEstimatedSequenceLabel(y_star);
    (*it)->setEstimatedSequencePosterior(posterior);
    (*it)->setEstimatedHiddenStatePosterior(hposterior);
    
    // Read ground truth label
    int y_true = (*it)->getSequenceLabel();
    
    // optionally writes results in file
    if( fileOutput) {
      (*fileOutput) << y_star << " [";
      for( int i=0; i<nbSeqLabels; i++ )
        (*fileOutput) << posterior->getValue(i) << "\t";
      (*fileOutput) << "]\n";
    }
    
    // Update total of positive detections
    seqTotalPos[y_true]++;
    seqTotalPosDetected[y_star]++;
    if( y_true == y_star)
      seqTruePos[y_true]++;
  }
  
  // Print results
  if(fileStats)
  {
    (*fileStats) << std::endl << "Calculations per sequences:" << std::endl;
    (*fileStats) << "Label\tTrue+\tMarked+\tDetect+\tPrec.\tRecall\tF1" << std::endl;
  }
  double prec,recall;
  int SumTruePos = 0, SumTotalPos = 0, SumTotalPosDetected = 0;
  for(int i=0 ; i<nbSeqLabels ; i++)
  {
    SumTruePos += seqTruePos[i]; SumTotalPos += seqTotalPos[i]; SumTotalPosDetected += seqTotalPosDetected[i];
    prec=(seqTotalPos[i]==0)?0:((double)(seqTruePos[i]*100000/seqTotalPos[i]))/1000;
    recall=(seqTotalPosDetected[i]==0)?0:((double)(seqTruePos[i]*100000/seqTotalPosDetected[i]))/1000;
    if(fileStats)
      (*fileStats) << i << ":\t" << seqTruePos[i] << "\t" << seqTotalPos[i] << "\t" << seqTotalPosDetected[i] << "\t" << prec << "\t" << recall << "\t" << 2*prec*recall/(prec+recall) << std::endl;
  }
  prec=(SumTotalPos==0)?0:((double)SumTruePos)*100.0/((double)SumTotalPos);
  recall=(SumTotalPosDetected==0)?0:((double)SumTruePos)*100.0/((double)SumTotalPosDetected);
  if(fileStats)
  {
    (*fileStats) << "-----------------------------------------------------------------------" << std::endl;
    (*fileStats) << "Ov:\t" << SumTruePos << "\t" << SumTotalPos << "\t" << SumTotalPosDetected << "\t" << prec << "\t" << recall << "\t" << 2*prec*recall/(prec+recall) << std::endl;
  }
  returnedF1value = 2*prec*recall/(prec+recall);
  
  if( fileOutput )
  {
    fileOutput->close();
    delete fileOutput;
  }
  if(fileStats != &std::cout && fileStats != NULL)
  {
    ((std::ofstream*)fileStats)->close();
    delete fileStats;
  }
  return returnedF1value;
}
