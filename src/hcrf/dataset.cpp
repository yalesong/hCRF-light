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

#include "hcrf/dataset.h"

using namespace std;
using namespace hCRF;

DataSequence::DataSequence()
{
  init();
}

DataSequence::DataSequence(const DataSequence& other)
{
  init();
  operator=(other);
}

DataSequence& DataSequence::operator=(const DataSequence&)
{
  throw HcrfNotImplemented("Copying DataSequence is not supported") ;
}

DataSequence::DataSequence(dMatrix* precomputedFeatures, iVector* stateLabels_, int seqLabel)
{
  init();
  precompFeatures = precomputedFeatures;
  stateLabels = stateLabels_;
  sequenceLabel = seqLabel;
}

void DataSequence::init()
{
  sequenceLabel = -1;
  weightSequence = 1.0;
  stateLabels = 0;
  statesPerNode = 0;
  adjMat = 0;
  adjMatMV = 0;
  precompFeatures = 0;
  precompFeaturesSparse = 0;
  estimatedSequenceLabel = -1;
  estimatedStateLabels = 0;
  estimatedSequencePosterior = 0;
  estimatedStatePosterior = 0;
  estimatedHiddenStatePosterior = 0;
  gateMat = new dMatrix();
  deepSeqGroupLabels.clear();
}

DataSequence::~DataSequence()
{
  if(adjMat != NULL)
  {
    delete adjMat;
    adjMat = NULL;
  }
  if(adjMatMV != NULL)
  {
    delete adjMatMV;
    adjMatMV = NULL;
  }
  if(stateLabels != NULL)
  {
    delete stateLabels;
    stateLabels = NULL;
  }
  if(statesPerNode != NULL)
  {
    delete statesPerNode;
    statesPerNode = NULL;
  }
  if (precompFeatures != NULL)
  {
    delete precompFeatures;
    precompFeatures = NULL;
  }
  if (estimatedStateLabels != NULL)
  {
    delete estimatedStateLabels;
    estimatedStateLabels = NULL;
  }
  if (estimatedSequencePosterior != NULL)
  {
    delete estimatedSequencePosterior;
    estimatedSequencePosterior = NULL;
  }
  if (estimatedStatePosterior != NULL)
  {
    delete estimatedStatePosterior;
    estimatedStatePosterior = NULL;
  }
  if (estimatedHiddenStatePosterior != NULL)
  {
    delete estimatedHiddenStatePosterior;
    estimatedHiddenStatePosterior = NULL;
  }
  if (gateMat != NULL)
  {
    delete gateMat;
    gateMat = NULL;
  }
}

//*
// Public Methods
//*

int DataSequence::load(istream* isData, istream* isLabels, istream* isAdjMat,
                       istream* isStatesPerNodes, istream* isDataSparse)
{
  if(isData == NULL && isLabels == NULL && isAdjMat == NULL && isStatesPerNodes == NULL)
    return 1;
  
  if(isData)
  {
    dMatrix* pNewMat = new dMatrix;
    if(pNewMat->read(isData)==0)
      precompFeatures = pNewMat;
    else
    {
      delete pNewMat;
      return 1;
    }
  }
  if(isLabels)
  {
    iVector* pNewVec = new iVector;
    if(pNewVec->read(isLabels)==0)
      stateLabels = pNewVec;
    else
    {
      delete pNewVec;
      return 1;
    }
  }
  if(isAdjMat)
  {
    uMatrix* pNewMat = new uMatrix;
    if(pNewMat->read(isAdjMat)==0)
      adjMat = pNewMat;
    else
    {
      delete pNewMat;
      return 1;
    }
  }
  if(isStatesPerNodes)
  {
    iMatrix* pNewMat = new iMatrix;
    if(pNewMat->read(isStatesPerNodes)==0)
      statesPerNode = pNewMat;
    else
    {
      delete pNewMat;
      return 1;
    }
  }
  if(isDataSparse)
  {
    dMatrixSparse* pNewMat = new dMatrixSparse;
    if(pNewMat->read(isDataSparse)==0)
      precompFeaturesSparse = pNewMat;
    else
    {
      delete pNewMat;
      return 1;
    }
  }
  
  return 0;
}


int	DataSequence::length() const
{
  if (precompFeatures != NULL)
    return precompFeatures->getWidth();
  if (precompFeaturesSparse != NULL)
    return (int)precompFeaturesSparse->getWidth();
  else
    return 0;
}


void DataSequence::equal(const DataSequence&)
{
  throw HcrfNotImplemented("Comparing DataSequence is not supported") ;
}

void DataSequence::setSequenceLabel(int seqLabel)
{
  sequenceLabel = seqLabel;
}

int DataSequence::getSequenceLabel() const
{
  return sequenceLabel;
}

void DataSequence::setStateLabels(iVector *v)
{
  if(stateLabels != NULL)
  {
    delete stateLabels;
    stateLabels = NULL;
  }
  stateLabels = v;
}

int DataSequence::getStateLabels(int nodeIndex) const
{
  return (*stateLabels)[nodeIndex];
}


iVector* DataSequence::getStateLabels() const
{
  return stateLabels;
}

void DataSequence::setAdjacencyMatrix(uMatrix* m)
{
  if(adjMat != NULL)
  {
    delete adjMat;
    adjMat = NULL;
  }
  adjMat = m;
}

void DataSequence::getAdjacencyMatrix(uMatrix& outMat) const
{
  if (adjMat == NULL) {
    outMat.resize(0,0);
  } else{
    outMat = *adjMat;
  }
}

void DataSequence::setAdjacencyMatrixMV(iMatrix* m)
{
  if(adjMatMV != NULL)
  {
    delete adjMatMV;
    adjMat = NULL;
  }
  adjMatMV = m;
}

void DataSequence::getAdjacencyMatrixMV(iMatrix& outMat) const
{
  if (adjMatMV == NULL) {
    outMat.resize(0,0);
  } else{
    outMat = *adjMatMV;
  }
}

void DataSequence::setPrecomputedFeatures(dMatrix* m)
{
  if(precompFeatures != NULL)
  {
    delete precompFeatures;
    precompFeatures = NULL;
  }
  precompFeatures = m;
}

dMatrix* DataSequence::getPrecomputedFeatures() const
{
  return precompFeatures;
}

void DataSequence::setPrecomputedFeaturesSparse(dMatrixSparse* m)
{
  if(precompFeaturesSparse != NULL)
  {
    delete precompFeaturesSparse;
    precompFeaturesSparse = NULL;
  }
  precompFeaturesSparse = m;
}

dMatrixSparse* DataSequence::getPrecomputedFeaturesSparse() const
{
  return precompFeaturesSparse;
}

// Takes ownership of spn
void DataSequence::setStatesPerNode(iMatrix* spn)
{
  if(statesPerNode != NULL)
  {
    delete statesPerNode;
    statesPerNode = NULL;
  }
  statesPerNode = spn;
}

iMatrix* DataSequence::getStatesPerNode() const
{
  return statesPerNode;
}

void DataSequence::setEstimatedStateLabels(iVector *v)
{
  if(estimatedStateLabels != NULL)
  {
    delete estimatedStateLabels;
    estimatedStateLabels = NULL;
  }
  estimatedStateLabels = v;
}

iVector* DataSequence::getEstimatedStateLabels() const
{
  return estimatedStateLabels;
}

void DataSequence::setEstimatedSequenceLabel(int seqLabel)
{
  estimatedSequenceLabel = seqLabel;
}

int DataSequence::getEstimatedSequenceLabel() const
{
  return estimatedSequenceLabel;
}

void DataSequence::setEstimatedSequencePosterior(dVector *m)
{
  if(estimatedSequencePosterior != NULL)
  {
    delete estimatedSequencePosterior;
    estimatedSequencePosterior = NULL;
  }
  estimatedSequencePosterior= m;
}

dVector* DataSequence::getEstimatedSequencePosterior() const
{
  return estimatedSequencePosterior;
}

void DataSequence::setEstimatedStatePosterior(dMatrix *m)
{
  if(estimatedStatePosterior != NULL)
  {
    delete estimatedStatePosterior;
    estimatedStatePosterior = NULL;
  }
  estimatedStatePosterior= m;
}

dMatrix* DataSequence::getEstimatedStatePosterior() const
{
  return estimatedStatePosterior;
}

void DataSequence::setEstimatedHiddenStatePosterior(dMatrix *m)
{
  if(estimatedHiddenStatePosterior != NULL)
  {
    delete estimatedHiddenStatePosterior;
    estimatedHiddenStatePosterior = NULL;
  }
  estimatedHiddenStatePosterior = m;
}

dMatrix* DataSequence::getEstimatedHiddenStatePosterior() const
{
  return estimatedHiddenStatePosterior;
}

void DataSequence::setWeightSequence(double w)
{
  weightSequence = w;
}

double DataSequence::getWeightSequence() const
{
  return weightSequence;
}

dMatrix* DataSequence::getGateMatrix()
{
  if(gateMat == NULL)
    gateMat = new dMatrix();
  return gateMat;
}


//-------------------------------------------------------------
// DataSet Class
//-------------------------------------------------------------

//*
// Constructors and Deconstructor
//*

DataSet::DataSet()
: container(std::vector<DataSequence*>())
{
  //does nothing
  numSamplesPerClass = new iVector();
}

DataSet::DataSet(const char *fileData, const char *fileStateLabels,
                 const char *fileSeqLabels, const char * fileAdjMat ,
                 const char * fileStatesPerNodes,const char * fileDataSparse)
: container(std::vector<DataSequence*>())
{
  numSamplesPerClass = new iVector();
  load(fileData, fileStateLabels, fileSeqLabels, fileAdjMat ,
       fileStatesPerNodes,fileDataSparse);
}

DataSet::~DataSet()
{
  clearSequence();
  if( numSamplesPerClass != NULL ) {
    delete numSamplesPerClass;
    numSamplesPerClass = NULL;
  }
}

//*
// Public Methods
//*

void DataSet::clearSequence()
{
  for(vector<DataSequence*>::iterator itSeq = container.begin();
      itSeq != container.end(); itSeq++)
  {
    delete (*itSeq);
    (*itSeq) = NULL;
  }
  container.clear();
}

int DataSet::load(const char *fileData, const char *fileStateLabels,
                  const char *fileSeqLabels, const char * fileAdjMat,
                  const char * fileStatesPerNodes,const char * fileDataSparse)
{
  istream* isData = NULL;
  istream* isDataSparse = NULL;
  istream* isStateLabels = NULL;
  istream* isSeqLabels = NULL;
  istream* isAdjMat = NULL;
  istream* isStatesPerNodes = NULL;
  
  if(fileData != NULL)
  {
    isData = new ifstream(fileData);
    if(!((ifstream*)isData)->is_open())
    {
      cerr << "Can't find data file: " << fileData << endl;
      delete isData;
      isData = NULL;
      throw BadFileName("Can't find data files");
    }
  }
  
  if(fileStateLabels != NULL)
  {
    isStateLabels = new ifstream(fileStateLabels);
    if(!((ifstream*)isStateLabels)->is_open())
    {
      cerr << "Can't find state labels file: " << fileStateLabels << endl;
      delete isStateLabels;
      isStateLabels = NULL;
      throw BadFileName("Can't find state labels file");
    }
  }
  if(fileSeqLabels != NULL)
  {
    isSeqLabels = new ifstream(fileSeqLabels);
    if(!((ifstream*)isSeqLabels)->is_open())
    {
      cerr << "Can't find sequence labels file: " << fileSeqLabels << endl;
      delete isSeqLabels;
      isSeqLabels = NULL;
    }
  }
  if(fileAdjMat != NULL)
  {
    isAdjMat = new ifstream(fileAdjMat);
    if(!((ifstream*)isAdjMat)->is_open())
    {
      cerr << "Can't find adjency matrices file: " << fileAdjMat << endl;
      delete isAdjMat;
      isAdjMat = NULL;
    }
  }
  if(fileStatesPerNodes != NULL)
  {
    isStatesPerNodes = new ifstream(fileStatesPerNodes);
    if(!((ifstream*)isStatesPerNodes)->is_open())
    {
      cerr << "Can't find states per nodes file: " << fileStatesPerNodes << endl;
      delete isStatesPerNodes;
      isStatesPerNodes = NULL;
    }
  }
  
  if(fileDataSparse != NULL)
  {
    isDataSparse = new ifstream(fileDataSparse);
    if(!((ifstream*)isDataSparse)->is_open())
    {
      cerr << "Can't find sparse data file: " << fileDataSparse << endl;
      delete isDataSparse;
      isDataSparse = NULL;
      throw BadFileName("Can't find sparse data files");
    }
  }
  
  DataSequence* seq = new DataSequence;
  int seqLabel;
  
  while(seq->load(isData,isStateLabels,isAdjMat,isStatesPerNodes,isDataSparse) == 0)
  {
    if(isSeqLabels)
    {
      *isSeqLabels >> seqLabel;
      seq->setSequenceLabel(seqLabel);
    }
    container.insert(container.end(),seq);
    seq = new DataSequence;
  }
  delete seq;
  
  if(isSeqLabels)
    searchNumberOfSamplesPerClass();
  
  if(isData)
    delete isData;
  if(isStateLabels)
    delete isStateLabels;
  if(isSeqLabels)
    delete isSeqLabels;
  if(isAdjMat)
    delete isAdjMat;
  if(isStatesPerNodes)
    delete isStatesPerNodes;
  if(isDataSparse)
    delete isDataSparse;
  
  return 0;
}


int DataSet::searchNumberOfStates()
{
  int MaxLabel = -1;
  
  for(vector<DataSequence*>::iterator itSeq = container.begin(); itSeq != container.end(); itSeq++) {
    if((*itSeq)->getStateLabels()) {
      int seqMaxLabel = (*itSeq)->getStateLabels()->getMaxValue();
      if(seqMaxLabel > MaxLabel)
        MaxLabel = seqMaxLabel;
    }
  }
  return MaxLabel + 1;
}

int DataSet::searchNumberOfSequenceLabels()
{
  int MaxLabel = -1;
  
  for(vector<DataSequence*>::iterator itSeq = container.begin(); itSeq != container.end(); itSeq++) {
    if((*itSeq)->getSequenceLabel() > MaxLabel)
      MaxLabel = (*itSeq)->getSequenceLabel();
  }
  return MaxLabel + 1;
}

int DataSet::getNumberofRawFeatures()
{
  if(size() > 0 && (*(container.begin()))->getPrecomputedFeatures() != NULL)
    return (*(container.begin()))->getPrecomputedFeatures()->getHeight();
  else
    return 0;
}

void DataSet::searchNumberOfSamplesPerClass()
{
  int nbClass = searchNumberOfSequenceLabels();
  numSamplesPerClass = new iVector(nbClass);
  
  int yi;
  std::vector<DataSequence*>::iterator itSeq;
  for(itSeq = container.begin(); itSeq != container.end(); itSeq++)
  {
    yi = (*itSeq)->getSequenceLabel();
    numSamplesPerClass->setValue(yi, numSamplesPerClass->getValue(yi)+1);
  }
  
  double nominator = (double)container.size() / (double)nbClass;
  for(itSeq = container.begin(); itSeq != container.end(); itSeq++)
  {
    yi = (*itSeq)->getSequenceLabel();
    (*itSeq)->setWeightSequence(nominator/(double)numSamplesPerClass->getValue(yi));
  }
  printf("numSamplesPerClass: [ ");
  for(int i=0; i<numSamplesPerClass->getLength(); i++)
    printf("%d ", numSamplesPerClass->getValue(i));
  printf("]\n");
}

int DataSet::getNumberOfSamplesPerClass(int y)
{
  return (y<numSamplesPerClass->getLength())
		? numSamplesPerClass->getValue(y) : -1;
}


std::ostream& operator <<(std::ostream& out, const DataSequence& seq)
{
  for(int i = 0; i < seq.length(); i++) {
    if(seq.getPrecomputedFeatures()) {
      out << "f(:," << i << ") = [";
      for(int j=0; j < seq.getPrecomputedFeatures()->getHeight(); j++)
        out << (*seq.getPrecomputedFeatures())(j,i) << " ";
      out << endl;
    }
    if(seq.getStateLabels())
      out << "y(" << i << ") = " << (*seq.getStateLabels())[i] << endl;
  }
  return out;
}

std::ostream& operator <<(std::ostream& out, const DataSet& data)
{
  for(size_t indexSeq = 0; indexSeq < data.size(); indexSeq++) {
    out << "Sequence " << indexSeq << endl;
    out << *(data.at(indexSeq));
  }
  return out;
}

