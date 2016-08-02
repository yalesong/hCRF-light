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


#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "hcrf/featuregenerator.h"

featureVector::featureVector(): realSize(8), capacity(8), pFeatures(NULL)
{
  pFeatures = new feature[capacity];
  memset(pFeatures, 0, capacity*sizeof(feature));
}

featureVector::featureVector(const featureVector& source):
realSize(source.realSize), capacity(source.capacity), pFeatures(NULL)
{
  pFeatures = new feature[capacity];
  memcpy(pFeatures, source.pFeatures, capacity*sizeof(feature));
}

featureVector& featureVector::operator = (const featureVector& source)
{
  if(pFeatures) {
    delete[] pFeatures;
  }
  realSize = source.realSize;
  capacity = source.capacity;
  pFeatures = new feature[capacity];
  memcpy(pFeatures, source.pFeatures, capacity*sizeof(feature));
  return *this;
}

featureVector::~featureVector()
{
  if(pFeatures) {
    delete[] pFeatures;
    pFeatures = NULL;
  }
}

void featureVector::resize(int newSize)
/** This function is used to resize the vector. To step,
 first we check if need to grow the vector (doubling size each time).
 Next we initilise the new space.
 **/
{
  // Easy case: We dont need to allocate any new memory.
  if (newSize <= capacity)
  {
    realSize = newSize;
    return;
  }
  //While the capacity is too small (*2 is cheap)
  while (newSize > capacity)
    capacity *=2;
  feature* tmpNewPointer = new feature[capacity];
  memcpy(tmpNewPointer,pFeatures,realSize*sizeof(feature));
  //We also initialise to zero the extra memory
  memset(tmpNewPointer+realSize, 0, (capacity-realSize)*sizeof(feature));
  delete[] pFeatures;
  pFeatures = tmpNewPointer;
  realSize = newSize;
}

int featureVector::size()
{
  return realSize;
}

void featureVector::clear()
{
  resize(0);
}

feature* featureVector::getPtr()
{
  return pFeatures;
}

feature* featureVector::addElement()
{
  resize(realSize+1);
  return pFeatures+realSize-1;
}


FeatureType::FeatureType():layer(0), idOffset(0), nbFeatures(0),
idOffsetPerLabel(0), nbFeaturesPerLabel(0), strFeatureTypeName(), featureTypeId(0)
{}
FeatureType::FeatureType(int l):layer(l), idOffset(0), nbFeatures(0),
idOffsetPerLabel(0), nbFeaturesPerLabel(0), strFeatureTypeName(), featureTypeId(0)
{}

FeatureType::~FeatureType()
{}

void FeatureType::setIdOffset(int offset, int seqLabel)
{
  if(seqLabel == -1 || idOffsetPerLabel.getLength() == 0)
    idOffset = offset;
  else
    idOffsetPerLabel[seqLabel] = offset;
}

void FeatureType::init(const DataSet&, const Model& m)
{
  if (m.getNumberOfSequenceLabels()>0) {
    idOffsetPerLabel.create(m.getNumberOfSequenceLabels());
    nbFeaturesPerLabel.create(m.getNumberOfSequenceLabels());
  }
}

void FeatureType::computeFeatureMask(iMatrix& matFeatureMask, const Model& m)
{
  int firstOffset = idOffset;
  int lastOffset = idOffset + nbFeatures;
  int nbLabels = m.getNumberOfSequenceLabels();
  
  for(int i = firstOffset; i < lastOffset; i++)
    for(int j = 0; j < nbLabels; j++)
      matFeatureMask(i,j) = 1;
}


int FeatureType::getNumberOfFeatures(int seqLabel)
{
  if (seqLabel == -1 || nbFeaturesPerLabel.getLength() == 0)
    return nbFeatures;
  else
    return nbFeaturesPerLabel[seqLabel];
}

bool FeatureType::isEdgeFeatureType()
{
  return false;
}

iVector& FeatureType::getOffsetPerLabel()
{
  return idOffsetPerLabel;
}

iVector& FeatureType::getNbFeaturePerLabel()
{
  return nbFeaturesPerLabel;
}

char* FeatureType::getFeatureTypeName()
{
  return (char*)strFeatureTypeName.c_str();
}

int FeatureType::getFeatureTypeId()
{
  return featureTypeId;
}

void FeatureType::setNumberOfFeatures(int nFeatures)
{
  nbFeatures = nFeatures;
}

void FeatureType::setOffsetPerLabel(const iVector& newOffsetPerLabel)
{
  idOffsetPerLabel = newOffsetPerLabel;
}

void FeatureType::setNbFeaturePerLabel(const iVector& newNbFeaturePerLabel)
{
  nbFeaturesPerLabel = newNbFeaturePerLabel;
}

void FeatureType::read(std::istream& is)
{
  is >> idOffset >> nbFeatures;
  idOffsetPerLabel.read(&is);
  nbFeaturesPerLabel.read(&is);
}

void FeatureType::write(std::ostream& os)
{
  os << idOffset << " " << nbFeatures << std::endl;
  idOffsetPerLabel.write(&os);
  nbFeaturesPerLabel.write(&os);
}

FeatureGenerator::FeatureGenerator():listFeatureTypes(), vecFeatures()
{
  nbThreadsMP = 1;
}

FeatureGenerator::~FeatureGenerator()
{
  clearFeatureList();
}

void FeatureGenerator::setMaxNumberThreads(int maxThreads)
{
  if (nbThreadsMP < maxThreads)
  {
    nbThreadsMP = maxThreads;
  }
}

void FeatureGenerator::addFeature(FeatureType* featureGen, bool insertAtEnd)
{
  if(featureGen->isEdgeFeatureType() || insertAtEnd)
    listFeatureTypes.insert(listFeatureTypes.end(),featureGen);
  else
    listFeatureTypes.insert(listFeatureTypes.begin(),featureGen);
}

void FeatureGenerator::initFeatures(const DataSet& dataset, Model& m)
{
  int y, nbLabels, offset=0;
  int cur_layer, prev_layer=0; // for HSS models
  nbLabels = m.getNumberOfSequenceLabels();
  
  iVector offsetPerLabel;
  if (nbLabels>0)
    offsetPerLabel.create(nbLabels);
  
  std::list<FeatureType*>::iterator itFeature;
  for(itFeature = listFeatureTypes.begin(); itFeature != listFeatureTypes.end(); itFeature++)
  {
    // Added for HSS: If layer is changed, reset offsetPerLabel
    cur_layer = (*itFeature)->getLayer();
    if( cur_layer>prev_layer ) {
      for(y = 0; y < nbLabels; y++)
        offsetPerLabel[y] = offset;
      prev_layer = cur_layer;
    }
    
    // Initialize feature
    (*itFeature)->init(dataset, m);
    
    // Set offsets for the current feature
    (*itFeature)->setIdOffset(offset);
    for(y = 0; y < nbLabels; y++)
      (*itFeature)->setIdOffset(offsetPerLabel[y],y);
    
    // Increment offsets for the next feature
    offset += (*itFeature)->getNumberOfFeatures();
    for(y = 0; y < nbLabels; y++)
      offsetPerLabel[y] += (*itFeature)->getNumberOfFeatures(y);
  }
  if(nbLabels > 0) {
    iMatrix matFeatureMask(nbLabels, offset);
    for( itFeature = listFeatureTypes.begin(); itFeature != listFeatureTypes.end(); itFeature++)
      (*itFeature)->computeFeatureMask(matFeatureMask,m);
    m.setFeatureMask(matFeatureMask);
  }
}

void FeatureGenerator::clearFeatureList()
{
  for(std::list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
      itFeature != listFeatureTypes.end(); itFeature++)
  {
    delete *itFeature;
    *itFeature = NULL;
  }
  listFeatureTypes.clear();
}

FeatureType* FeatureGenerator::getFeatureById(int id)
{
  FeatureType* result = NULL;
  for( std::list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
      itFeature != listFeatureTypes.end(); itFeature++) {
    if ((*itFeature)->getFeatureTypeId() == id)
    {
      result = (*itFeature);
      break;
    }
  }
  return result;
}

FeatureType* FeatureGenerator::getFeatureByIdAndLayer(int id,int layer)
{
  FeatureType* result = NULL;
  for( std::list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
      itFeature != listFeatureTypes.end(); itFeature++) {
    if ((*itFeature)->getFeatureTypeId()==id && (*itFeature)->getLayer()==layer)
    {
      result = (*itFeature);
      break;
    }
  }
  return result;
}

FeatureType* FeatureGenerator::getFeatureByBasicType(BasicFeatureType type)
{
  FeatureType* result = NULL;
  for( std::list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
      itFeature != listFeatureTypes.end(); itFeature++) {
    if ((*itFeature)->getBasicFeatureType() == type)
    {
      result = (*itFeature);
      break;
    }
  }
  return result;
}

void FeatureGenerator::getFeatures(featureVector& vecFeatures, DataSequence* X,
                                   Model* m, int nodeIndex, int prevNodeIndex, int seqLabel)
{
  vecFeatures.clear();
  for( std::list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
      itFeature != listFeatureTypes.end(); itFeature++) {
    (*itFeature)->getFeatures(vecFeatures, X, m, nodeIndex, prevNodeIndex, seqLabel);
  }
}

featureVector* FeatureGenerator::getFeatures(DataSequence* X, Model* m,
                                             int nodeIndex, int prevNodeIndex, int seqLabel)
{
  vecFeatures.clear();
  for( std::list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
      itFeature != listFeatureTypes.end(); itFeature++) {
    (*itFeature)->getFeatures(vecFeatures, X, m, nodeIndex, prevNodeIndex, seqLabel);
  }
  return &vecFeatures;
}


// Multiview Support
featureVector* FeatureGenerator::getAllFeatures(Model* m, int nbRawFeatures)
{
  vecFeatures.clear();
  for( std::list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
      itFeature != listFeatureTypes.end(); itFeature++)
  {
    (*itFeature)->getAllFeatures(vecFeatures, m, nbRawFeatures);
  }
  return &vecFeatures;
}

int FeatureGenerator::getNumberOfFeatures(eFeatureTypes typeFeature,
                                          int seqLabel)
{
  int totalFeatures = 0;
  for( std::list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
      itFeature != listFeatureTypes.end(); itFeature++)
  {
    if( (typeFeature == allTypes) ||
       (typeFeature == edgeFeaturesOnly &&
        (*itFeature)->isEdgeFeatureType() ) ||
       (typeFeature == nodeFeaturesOnly &&
        !(*itFeature)->isEdgeFeatureType()))
    {
      totalFeatures += (*itFeature)->getNumberOfFeatures(seqLabel);
    }
  }
  return totalFeatures;
  
}

double FeatureGenerator::evaluateLabels(DataSequence* X, Model* m, int seqLabel)
{
  dVector *w = m->getWeights();
  double phi = 0;
  int n = X->length(), i;
  feature* pFeature;
  iVector *s = X->getStateLabels();
  
  featureVector vecFeatures;
  for(i = 0; i<n; i++) {
    getFeatures(vecFeatures, X, m, i, -1,seqLabel);
    pFeature = vecFeatures.getPtr();
    for(int j = 0; j < vecFeatures.size(); j++,pFeature++){
      if(pFeature->nodeState==s->getValue(i))
        phi += w->getValue(pFeature->globalId) * pFeature->value;
    }
  }
  //add edge features
  uMatrix adjMat;
  m->getAdjacencyMatrix(adjMat, X);
  int row,col;
  for(col=0; col<n; col++) { // current node index
    for(row=0; row<=col; row++) { // previous node index
      if(adjMat.getValue(row,col)==0) {
        continue;
      }
      getFeatures(vecFeatures, X, m, col, row,seqLabel);
      pFeature = vecFeatures.getPtr();
      for(int j = 0; j < vecFeatures.size(); j++, pFeature++) {
        if(pFeature->nodeState==s->getValue(col) && pFeature->prevNodeState==s->getValue(row)) {
          phi += w->getValue(pFeature->globalId) * pFeature->value;
        }
      }
    }
  }
  return phi;
}

std::list<FeatureType*>& FeatureGenerator::getListFeatureTypes()
{
  return listFeatureTypes;
}


void FeatureGenerator::load(char* pFilename)
{
  std::ifstream fileInput(pFilename);
  
  if (!fileInput.is_open())
  {
    std::cerr << "Can't find features definition file: " << pFilename << std::endl;
    return;
  }
  
  for( std::list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
      itFeature != listFeatureTypes.end(); itFeature++)
  {
    (*itFeature)->read(fileInput);
  }
  
  fileInput.close();
}

void FeatureGenerator::save(char* pFilename)
{
  std::ofstream fileOutput(pFilename);
  
  if (!fileOutput.is_open())
    return;
  
  for(std:: list<FeatureType*>::iterator itFeature = listFeatureTypes.begin();
      itFeature != listFeatureTypes.end(); itFeature++)
  {
    (*itFeature)->write(fileOutput);
  }
  fileOutput.close();
}

std::ostream& operator <<(std::ostream& out, const feature& f)
{
  out << "f[" << f.id <<"] : (i=" << f.prevNodeIndex << ", j=" << f.nodeIndex ;
  out << ", yi=" << f.prevNodeState << ", yj=" << f.nodeState << ", Y=" ;
  out << f.sequenceLabel <<") = " << f.value << std::endl;
  return out;
}

std::ostream& operator <<(std::ostream& out, featureVector& v)
{
  feature* data = v.getPtr();
  for (int i = 0; i< v.size();i++) {
    out<<(data[i]);
  }
  return out;
}

