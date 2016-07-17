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

#include <iostream>
#include <string>
#include <time.h>

#ifdef WIN32
#include <conio.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "hcrf/hCRF.h"

using namespace std;

// Single-view models
#define TOOLBOX_CRF 0
#define TOOLBOX_HCRF  1
#define TOOLBOX_LDCRF 2
// Multi-view models [Song et al., CVPR 2012]
#define TOOLBOX_MVHCRF  3
#define TOOLBOX_MVLDCRF 4
// Hierarchical Sequence Summarization Models [Song et al., CVPR 2013]
#define TOOLBOX_HSSHCRF 5
// One-Class Models [Song et al., IJCAI 2013]
#define TOOLBOX_OCCRF 6
#define TOOLBOX_OCHCRF 7

void usage (char **argv)
{
  cerr << "HCRF-light library ver 3.0"<< endl;
  cerr << "  Includes CRF, HCRF, LDCRF, MV-HCRF, MV-LDCRF, HSS-HCRF, OCCRF, OCHCRF"<< endl;
  cerr << endl;
  
  cerr << "options (case sensitive):" << endl;
  cerr << " -Fd\t Filename of the training data (def.= dataTrain.csv)" << endl;
  cerr << " -Fl\t Filename of the training labels (def.= labelsTrain.csv)" << endl;
  cerr << " -Fq\t Filename of the training sequence labels (def.= seqLabelsTrain.csv)" << endl;
  cerr << " -FD\t Filename of the testing data (def.= dataTest.csv)" << endl;
  cerr << " -FL\t Filename of the testing labels (def.= labelsTest.csv)" << endl;
  cerr << " -FQ\t Filename of the testing sequence labels (def.= seqLabelsTest.csv)" << endl;
  cerr << " -Fm\t Filename where the model is written (def.= model.txt)" << endl;
  cerr << " -Ff\t Filename where the features are written (def.= features.txt)" << endl;
  cerr << " -Fo\t Filename where the computed labels are written (def.= results.txt)" << endl;
  cerr << " -Fc\t Filename where the statistics are written (def.= stats.txt)" << endl;
  cerr << " -m\t Select the model type (crf, hcrf, ldcrf, mvhcrf, mvldcrf, hsshcrf, occrf, ochcrf) (def. =hcrf)" << endl;
  cerr << " -e\t (For multi-view models only) Select the graph type (linked, coupled, linkedcoupled) (def. = linked)" << endl;
  cerr << " -b\t Select the inference engine (fb, jt, lbp) (def. = fb)" << endl;
  //cerr << " -o\t Select optimizer: lbfgs or nrbm (def.= lbfgs)" << endl;
  cerr << " -v\t Number of views (def. = 1)" << endl;
  cerr << " -h\t Number of hidden states (def. = 4)" << endl;
  cerr << " -g\t Number of gates (def. = 0)" << endl;
  cerr << " -i\t Maximum number of optimization iterations (def. = 1000)" << endl;
  cerr << " -r\t rho: OCCRF probability margin threshold (def. = 0.5)" << endl;
  cerr << " -s\t Sigma2: L2-norm regularization factor (def. = 0.1)" << endl;
  cerr << " -z\t Random seed (def. = 123)" << endl;
  cerr << " -p\t Verbose level (def. = 1)" << endl;
  cerr << " -HL\t Number of feature layers (for HSS, def. = 4)" << endl;
  cerr << " -HT\t Threshold value for segmentation (for HSS, def. = 0.5)" << endl;
  cerr << endl;
  
  cerr << "Examples:" << endl;
  cerr << " HCRF with nbHiddenStates = 4" << endl;
  cerr << "   > hcrf-light -m hcrf -h 4 -Fd data.csv -Fq label.csv -FD data.csv -DQ label.csv" << endl;
  cerr << " HCNF with nbHiddenStates = 4, nbGates = 8" << endl;
  cerr << "   > hcrf-light -m hcrf -h 4 -g 8 -Fd data.csv -Fq label.csv -FD data.csv -DQ label.csv" << endl;
  cerr << " LDCRF with nbHiddenStates = 4" << endl;
  cerr << "   > hcrf-light -m ldcrf -h 4 -Fd data.csv -Fl label.csv -FD data.csv -DL label.csv" << endl;
  cerr << " HSS-HCRF with nbHiddenStates = 4, nbGates = 0, maxFeatureLayer = 4" << endl;
  cerr << "   > hcrf-light -m hsshcrf -h 4 -g 0 -HL 4 -HT 0.1 -Fd data.csv -Fq label.csv -FD data.csv -DQ label.csv" << endl;
  cerr << " HSS-HCNF with nbHiddenStates = 4, nbGates = 8, maxFeatureLayer = 4" << endl;
  cerr << "   > hcrf-light -m hsshcrf -h 4 -g 8 -HL 4 -HT 0.1 -Fd data.csv -Fq label.csv -FD data.csv -DQ label.csv" << endl;
  cerr << " OCHCRF with nbHiddenStates = 4, rho = 0.5" << endl;
  cerr << "   > hcrf-light -m ochcrf -h 4 -r 0.5 -Fd data.csv -Fq label.csv -FD data.csv -DQ label.csv" << endl;
  exit(1);
}

int main(int argc, char **argv)
{
  if( argc==1 )
    usage(argv);
  
  int toolboxType, opt, maxIt, debugLevel;
  int nbViews, nbHiddenStates, nbGates, maxFeatureLayer;
  double regFactorL2, segmentTau, rho;
  bool bWeightSequence;
  long seed;
  eGraphTypes graphType;
  
  // Default parameter values
  toolboxType     = TOOLBOX_HCRF;
  opt             = OPTIMIZER_LBFGS;
  maxIt           = 1000;
  debugLevel      = 1;
  nbViews         = 1;
  nbHiddenStates  = 4;
  nbGates         = 0;
  maxFeatureLayer = 1;
  regFactorL2     = 10;
  segmentTau      = 0.5;
  rho             = 0.5;
  graphType       = CHAIN;
  seed            = 6172632663;
  bWeightSequence = false;
		
  Toolbox* toolbox = NULL;
  InferenceEngine* engine = NULL;
  
  // Params for multi-view models
  std::vector<int> nbHiddenStatesMV;
  std::vector<std::vector<int> > rawFeatureIndex;
  
  string fDataTrain      = "dataTrain.csv";
  string fLabelsTrain    = "labelsTrain.csv" ;
  string fSeqLabelsTrain = "seqLabelsTrain.csv";
  string fDataTest       = "dataTest.csv";
  string fLabelsTest     = "labelsTest.csv";
  string fSeqLabelsTest  = "seqLabelsTest.csv";
  string fModel          = "model.txt";
  string fFeatures       = "features.txt";
  string fOutput         = "results.txt";
  string fStats          = "stats.txt";
  
  /* Read command-line arguments */
  for (int k=1; k<argc; k++)
  {
    if(argv[k][0] != '-')
      break;
    else if(argv[k][1] == 'm') {
      if(!strcmp(argv[k+1],"crf") || !strcmp(argv[k+1],"CRF"))
        toolboxType = TOOLBOX_CRF;
      else if(!strcmp(argv[k+1],"hcrf") || !strcmp(argv[k+1],"HCRF"))
        toolboxType = TOOLBOX_HCRF;
      else if(!strcmp(argv[k+1],"ldcrf") || !strcmp(argv[k+1],"LDCRF"))
        toolboxType = TOOLBOX_LDCRF;
      else if(!strcmp(argv[k+1],"mvhcrf") || !strcmp(argv[k+1],"MVHCRF"))
        toolboxType = TOOLBOX_MVHCRF;
      else if(!strcmp(argv[k+1],"mvldcrf") || !strcmp(argv[k+1],"MVLDCRF"))
        toolboxType = TOOLBOX_MVLDCRF;
      else if(!strcmp(argv[k+1],"hsshcrf") || !strcmp(argv[k+1],"HSSHCRF"))
        toolboxType = TOOLBOX_HSSHCRF;
      else if(!strcmp(argv[k+1],"occrf") || !strcmp(argv[k+1],"OCCRF"))
        toolboxType = TOOLBOX_OCCRF;
      else if(!strcmp(argv[k+1],"ochcrf") || !strcmp(argv[k+1],"OCHCRF"))
        toolboxType = TOOLBOX_OCHCRF;
      k++;
    }
    else if(argv[k][1] == 'e') {
      if(!strcmp(argv[k+1],"linked") || !strcmp(argv[k+1],"LINKED"))
        graphType = MV_GRAPH_LINKED;
      if(!strcmp(argv[k+1],"coupled") || !strcmp(argv[k+1],"COUPLED"))
        graphType = MV_GRAPH_COUPLED;
      if(!strcmp(argv[k+1],"linkedcoupled") || !strcmp(argv[k+1],"LINKEDCOUPLED"))
        graphType = MV_GRAPH_LINKED_COUPLED;
      k++;
    }
    else if(argv[k][1] == 'b') {
      if(!strcmp(argv[k+1],"fb") || !strcmp(argv[k+1],"FB"))
        engine = new InferenceEngineFB();
      else if(!strcmp(argv[k+1],"jt") || !strcmp(argv[k+1],"JT"))
        engine = new InferenceEngineJT();
      else if(!strcmp(argv[k+1],"lbp") || !strcmp(argv[k+1],"LBP"))
        engine = new InferenceEngineLoopyBP();
      k++;
    }
    else if(argv[k][1] == 'o') {
      if(!strcmp(argv[k+1],"lbfgs") || !strcmp(argv[k+1],"LBFGS"))
        opt = OPTIMIZER_LBFGS;
      else if(!strcmp(argv[k+1],"nrbm") || !strcmp(argv[k+1],"NRBM"))
        opt = OPTIMIZER_NRBM;
      k++;
    }
    else if(argv[k][1] == 'F') {
      if(argv[k][2] == 'd')      fDataTrain      = argv[++k];
      else if(argv[k][2] == 'l') fLabelsTrain    = argv[++k];
      else if(argv[k][2] == 'q') fSeqLabelsTrain = argv[++k];
      else if(argv[k][2] == 'D') fDataTest       = argv[++k];
      else if(argv[k][2] == 'L') fLabelsTest     = argv[++k];
      else if(argv[k][2] == 'Q') fSeqLabelsTest  = argv[++k];
      else if(argv[k][2] == 'm') fModel          = argv[++k];
      else if(argv[k][2] == 'f') fFeatures       = argv[++k];
      else if(argv[k][2] == 'o') fOutput         = argv[++k];
      else if(argv[k][2] == 'c') fStats          = argv[++k];
    }
    else if(argv[k][1] == 'p') debugLevel      = atoi(argv[++k]);
    else if(argv[k][1] == 'i') maxIt           = atoi(argv[++k]);
    else if(argv[k][1] == 'v') nbViews         = atoi(argv[++k]);
    else if(argv[k][1] == 'h') nbHiddenStates  = atoi(argv[++k]);
    else if(argv[k][1] == 'g') nbGates         = atoi(argv[++k]);
    else if(argv[k][1] == 'z') seed            = atol(argv[++k]);
    else if(argv[k][1] == 's') regFactorL2     = atof(argv[++k]);
    else if(argv[k][1] == 'r') rho             = atof(argv[++k]);
    else if(argv[k][1] == 'H') { // Hierarchical model params
      if(argv[k][2] == 'L')
        maxFeatureLayer = atoi(argv[++k]);
      else if(argv[k][2] == 'T')
        segmentTau = atof(argv[++k]);
    }
    else usage(argv);
  }
  
  if( toolboxType==TOOLBOX_MVHCRF || toolboxType==TOOLBOX_MVLDCRF ) {
    nbViews = 2;
    graphType = MV_GRAPH_LINKED;
    nbHiddenStatesMV.push_back(3);
    nbHiddenStatesMV.push_back(3);
    std::vector<int>  rfi_v1, rfi_v2;
    for(int i=0; i<12; i++)  rfi_v1.push_back(i);
    for(int i=12; i<20; i++)  rfi_v2.push_back(i);
    rawFeatureIndex.push_back(rfi_v1);
    rawFeatureIndex.push_back(rfi_v2);
  }
  
  if(toolboxType == TOOLBOX_CRF)
    toolbox = new ToolboxCRF();
  else if(toolboxType == TOOLBOX_HCRF)
    toolbox = new ToolboxHCRF(nbHiddenStates);
  else if(toolboxType == TOOLBOX_LDCRF)
    toolbox = new ToolboxLDCRF(nbHiddenStates);
  else if(toolboxType == TOOLBOX_HSSHCRF)
    toolbox = new ToolboxHSSHCRF(nbHiddenStates);
  else if(toolboxType == TOOLBOX_MVHCRF)
    toolbox = new ToolboxMVHCRF(graphType, nbViews, nbHiddenStatesMV, rawFeatureIndex);
  else if(toolboxType == TOOLBOX_MVLDCRF)
    toolbox = new ToolboxMVLDCRF(graphType, nbViews, nbHiddenStatesMV, rawFeatureIndex);
  else if(toolboxType == TOOLBOX_OCCRF)
    toolbox = new ToolboxOCCRF();
  else if(toolboxType == TOOLBOX_OCHCRF)
    toolbox = new ToolboxOCHCRF(nbHiddenStates);
  
  if( engine )
    toolbox->setInferenceEngine(engine);
  
  toolbox->setOptimizer(opt);
  toolbox->setDebugLevel(debugLevel);
  toolbox->setRandomSeed(seed);
  toolbox->getModel()->setRho(rho);
  toolbox->getModel()->setNbGates(nbGates);
  toolbox->getModel()->setWeightSequence(bWeightSequence);
  
  if(toolboxType == TOOLBOX_HSSHCRF) {
    toolbox->getModel()->setMaxFeatureLayer(maxFeatureLayer);
    toolbox->getModel()->setSegmentConst(segmentTau);
  }
  
  // Train
  {
    cout << "Reading training set..." << endl;
    DataSet data;
    const char* fileData = fDataTrain.empty()? 0 : fDataTrain.c_str();
    if( toolbox->isContinuousModel() )
      data.load(fileData,(char*)fLabelsTrain.c_str(),NULL,NULL,NULL);
    else
      data.load(fileData,NULL,(char*)fSeqLabelsTrain.c_str(),NULL,NULL);
    
    toolbox->setMaxNbIteration(maxIt);
    toolbox->setRegularizationL2(regFactorL2);
    toolbox->setRangeWeights(-1,1);
    toolbox->setWeightInitType(INIT_RANDOM);
    
    cout << "Starting training ..." << endl;
    toolbox->train(data,true);
    toolbox->save((char*)fModel.c_str(),(char*)fFeatures.c_str());
  }
  
  // Test
  {
    cout << "Reading testing set..." << endl;
    DataSet data;
    if( toolbox->isContinuousModel() )
      data.load((char*)fDataTest.c_str(),(char*)fLabelsTest.c_str());
    else
      data.load((char*)fDataTest.c_str(),NULL,(char*)fSeqLabelsTest.c_str());
    
    ofstream fileStats1 ((char*)fStats.c_str());
    if (fileStats1.is_open())
    {
      fileStats1 << endl << endl << "TESTING DATA SET" << endl << endl;
      fileStats1.close();
    }
    
    cout << "Starting testing ..." << endl;
    toolbox->load((char*)fModel.c_str(),(char*)fFeatures.c_str());
    toolbox->initToolbox();
    toolbox->test(data,(char*)fOutput.c_str(),(char*)fStats.c_str()); 
  }
  
  if(toolbox)
    delete toolbox;
  
  cout << "Press a key to continue..." << endl;
#ifdef WIN32
  _getch();
#endif
  
  return 0;
} 

