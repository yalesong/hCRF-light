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

#ifndef MODEL_H
#define MODEL_H

#define MAX_NUMBER_OF_LEVELS 10

//Standard Template Library includes
#include <vector>
#include <iostream>

//hCRF library includes
#include "hcrf/hcrfExcep.h"
#include "hcrf/dataset.h"

class FeatureGenerator;

enum eFeatureTypes
{
  allTypes = 0,
  edgeFeaturesOnly,
  nodeFeaturesOnly
};


// Type of graph topology
enum eGraphTypes {
  CHAIN,
  MV_GRAPH_LINKED,
  MV_GRAPH_COUPLED,
  MV_GRAPH_LINKED_COUPLED,
  MV_GRAPH_PREDEFINED,
  ADJMAT_PREDEFINED
};

enum{
  ALLSTATES,
  STATES_BASED_ON_LABELS,
  STATEMAT_PREDEFINED,
  STATEMAT_PROBABILISTIC
};

//-------------------------------------------------------------
// Model Class
//

class Model {
public:
  Model(int numberOfStates = 0, int numberOfSeqLabels = 0, int numberOfStateLabels = 0);
  ~Model();
  
  void setAdjacencyMatType(eGraphTypes atype, ...);
  eGraphTypes getAdjacencyMatType();
  
  int setStateMatType(int stype, ...);
  int getStateMatType();
  
  // adjacency and state matrix sizes are max-sizes, based on the
  // longest sequences seen thus far; use sequences length instead for
  // width and height of these matrices
  void getAdjacencyMatrix(uMatrix&, DataSequence* seq);
  iMatrix * getStateMatrix(DataSequence* seq);
  iVector * getStateMatrix(DataSequence* seq, int nodeIndex);
  
  void refreshWeights();
  void setWeights(const dVector& weights);
  dVector * getWeights(int seqLabel = -1);
  
  int getNumberOfStates() const;
  void setNumberOfStates(int numberOfStates);
  
  int getNumberOfStateLabels() const;
  void setNumberOfStateLabels(int numberOfStateLabels);
  
  int getNumberOfSequenceLabels() const;
  void setNumberOfSequenceLabels(int numberOfSequenceLabels);
  
  int getNumberOfRawFeaturesPerFrame();
  void setNumberOfRawFeaturesPerFrame(int numberOfRawFeaturesPerFrame);
  
  void setRegL1Sigma(double sigma, eFeatureTypes typeFeature = allTypes);
  void setRegL2Sigma(double sigma, eFeatureTypes typeFeature = allTypes);
  double getRegL1Sigma();
  double getRegL2Sigma();
  eFeatureTypes getRegL1FeatureTypes();
  eFeatureTypes getRegL2FeatureTypes();
  void setLambda(double l) {lambda=l;};
  double getLambda() {return lambda;};
  
  void setFeatureMask(iMatrix &ftrMask);
  iMatrix* getFeatureMask();
  int getNumberOfFeaturesPerLabel();
  
  iMatrix& getStatesPerLabel();
  iVector& getLabelPerState();
  int getDebugLevel();
  void setDebugLevel(int newDebugLevel);
  
  void load(const char* pFilename);
  void save(const char* pFilename) const;
  
  int read(std::istream* stream);
  int write(std::ostream* stream) const;
  
  uMatrix* getInternalAdjencyMatrix();
  iMatrix *getInternalStateMatrix();
  
  // Multi-View Support
  Model(int numberOfViews, int* numberOfStatesMultiView,
        int numberOfSeqLabels = 0, int numberOfStateLabels = 0);
  
  bool isMultiViewMode() const;
  
  void setNumberOfViews(int numberOfViews);
  int getNumberOfViews() const;
  
  void setCurrentView(int view) {currentView = view;};
  int getCurrentView() {return currentView;};
  
  void setNumberOfStatesMV(int* numberOfStatesMultiView);
  int getNumberOfStatesMV(int view) const;
  
  iMatrix& getStatesPerLabelMV(int view);
  iVector& getLabelPerStateMV(int view);
  
  void setRawFeatureIndexMV(std::vector<std::vector<int> > rawFeatureIndexPerView);
  std::vector<int> getRawFeatureIndexMV(int view) const;
  
  void getAdjacencyMatrixMV(iMatrix&, DataSequence* seq);
  
  //
  void setWeightSequence(bool bws) {bWeightSequence = bws;};
  bool isWeightSequence() {return bWeightSequence;};
  
  //
  void setMaxMargin(bool bMaxMargin) {bComputeMaxMargin = bMaxMargin;};
  bool isMaxMargin() {return bComputeMaxMargin;};
  
  void setNbGates(int n) {nbGates=n;};
  int getNbGates() {return nbGates;};
  
  // Hierarchical
  int getCurrentFeatureLayer() {return currentFeatureLayer;};
  int getMaxFeatureLayer() {return maxFeatureLayer;};
  double getSegmentConst() {return segment_c;};
  
  void setCurrentFeatureLayer(int layer) {currentFeatureLayer=layer;};
  void setMaxFeatureLayer(int layer) {maxFeatureLayer=layer;};
  void setSegmentConst(double c) {segment_c=c;};
  
  // One-class
  void setRho(double r) {rho = r;};
  double getRho() {return rho;};
  
private:
  int numberOfSequenceLabels;
  int numberOfStates;
  int numberOfStateLabels;
  int numberOfFeaturesPerLabel;
  int numberOfRawFeaturesPerFrame;
  
  dVector weights;
  std::vector<dVector> weights_y;
  
  double regL1Sigma;
  double regL2Sigma;
  double rho;
  double lambda;
  eFeatureTypes regL1FeatureType;
  eFeatureTypes regL2FeatureType;
  
  int debugLevel;
  int stateMatType;
  
  eGraphTypes adjMatType;
  uMatrix adjMat;
  iMatrix stateMat, featureMask;
  iMatrix statesPerLabel;
  iVector stateVec, labelPerState;
  
  int loadAdjacencyMatrix(const char *pFilename);
  int loadStateMatrix(const char *pFilename);
  
  void makeChain(uMatrix& m, int n);
  void predefAdjMat(uMatrix& m, int n);
  iMatrix * makeFullStateMat(int n);
  iMatrix * makeLabelsBasedStateMat(DataSequence* seq);
  iMatrix * predefStateMat(int n);
  void updateStatesPerLabel();
  
  // Distribution-sensitive prior
  bool bWeightSequence;
  
  // Max margin
  bool bComputeMaxMargin;
  
  // Neural Layer
  int nbGates;
  
  // Multi-View Support
  int numberOfViews;
  int currentView;
  int* numberOfStatesMV;
  std::vector<std::vector<int> > rawFeatureIndex;
  std::vector<iVector> labelPerStateMV;
  std::vector<iMatrix> statesPerLabelMV;
  
  // Creates adjMat for multiview chains. Contains unique edgeID.
  void makeChainMV(iMatrix& m, int seqLength);
  void updateStatesPerLabelMV();
  
  // Hierarchical
  int currentFeatureLayer;
  int maxFeatureLayer;
  double segment_c;
};

// stream io routines
std::istream& operator >>(std::istream& in, Model& m);
std::ostream& operator <<(std::ostream& out, const Model& m);

#endif
