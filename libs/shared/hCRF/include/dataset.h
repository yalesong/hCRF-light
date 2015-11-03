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


#ifndef DATASET_H
#define DATASET_H

//Standard Template Library includes
#include <vector>
#include <list>
#include <fstream>

//library includes
#include "matrix.h"
#include "hcrfExcep.h"
 

//-------------------------------------------------------------
// DataSequence Class
//



class DataSequence {
public:
    DataSequence();
    DataSequence(const DataSequence& other);
    DataSequence& operator=(const DataSequence&);
    virtual ~DataSequence();
    DataSequence(dMatrix* precomputedFeatures, iVector* stateLabels, int sequenceLabel);

    int load(std::istream* isData, std::istream* isLabels,
             std::istream* isAdjMat, std::istream* isStatesPerNodes, 
			 std::istream* isDataSparse);
    void equal(const DataSequence & seq);

    int	length() const;

    void setStateLabels(iVector *v);
    iVector* getStateLabels() const;
    int getStateLabels(int nodeIndex) const;

    void setAdjacencyMatrix(uMatrix *m);
    void getAdjacencyMatrix(uMatrix& m) const;
    void setAdjacencyMatrixMV(iMatrix *m);
    void getAdjacencyMatrixMV(iMatrix& m) const;

    void setPrecomputedFeatures(dMatrix *m);
    dMatrix* getPrecomputedFeatures() const;
	
	void setPrecomputedFeaturesSparse(dMatrixSparse *m);
	dMatrixSparse* getPrecomputedFeaturesSparse() const;

    void setStatesPerNode(iMatrix* spn);
    iMatrix* getStatesPerNode() const;

    void setSequenceLabel(int seqLabel);
    int getSequenceLabel() const;

	// For CRF-based
    void setEstimatedStateLabels(iVector *v);
    iVector* getEstimatedStateLabels() const;
	
	// For CRF-based, m is of size |Y|-by-|T_i|
    void setEstimatedStatePosterior(dMatrix *m);
    dMatrix* getEstimatedStatePosterior() const;

	// For HCRF-based
    void setEstimatedSequenceLabel(int seqLabel);
    int getEstimatedSequenceLabel() const;
	
	// For HCRF-based, m is of size |Y|-by-1 
    void setEstimatedSequencePosterior(dVector *m);
    dVector* getEstimatedSequencePosterior() const;

	// For HCRF-based, m is of size |H|-by-|T_i|
	void setEstimatedHiddenStatePosterior(dMatrix *m);
	dMatrix* getEstimatedHiddenStatePosterior() const;

    void   setWeightSequence(double w);
    double getWeightSequence() const;
	
	dMatrix* getGateMatrix();
	
	// Hierarchical
	std::vector<std::vector<std::vector<int> > >* getDeepSeqGroupLabels() {return &deepSeqGroupLabels;};
	
  protected:
    void init();

    int      sequenceLabel;
    double	 weightSequence;
    iVector* stateLabels; //
    iMatrix* statesPerNode; //
    uMatrix* adjMat; //
	iMatrix* adjMatMV;
    dMatrix* precompFeatures; //
    dMatrixSparse* precompFeaturesSparse; //

    int		 estimatedSequenceLabel;
    iVector* estimatedStateLabels;
    dVector* estimatedSequencePosterior;
    dMatrix* estimatedStatePosterior;
	dMatrix* estimatedHiddenStatePosterior;

	dMatrix* gateMat;

	// Hierarchical
	std::vector<std::vector<std::vector<int> > > deepSeqGroupLabels;
};




//-------------------------------------------------------------
// DataSet Class
//

class DataSet
{
  public:
    DataSet();
    ~DataSet();
    DataSet(const char *fileData, const char *fileStateLabels = NULL,
            const char *fileSeqLabels = NULL, const char * fileAdjMat = NULL,
            const char * fileStatesPerNodes = NULL, const char * fileDataSparse = NULL);

    int load(const char *fileData, const char *fileStateLabels = NULL,
             const char *fileSeqLabels = NULL, const char * fileAdjMat = NULL,
             const char * fileStatesPerNodes = NULL, const char * fileDataSparse = NULL);
    void clearSequence();
	
    int searchNumberOfStates();
    int searchNumberOfSequenceLabels();
    int getNumberofRawFeatures();
		int getNumberOfSamplesPerClass(int y);
    
	void insert(std::vector<DataSequence*>::iterator iter, DataSequence* d){
		container.insert(iter, d);
	}
	
	DataSequence* at (size_t i) const{
       return container.at(i);
    }
    size_t size() const{
       return container.size();
    }
    typedef std::vector<DataSequence*>::iterator iterator;
    iterator begin(){
        return container.begin();
    }
    iterator end(){
        return container.end();
    }
    typedef std::vector<DataSequence*>::const_iterator const_iterator;
    const_iterator begin() const{
        return container.begin();
    }
    const_iterator end() const{
        return container.end();
    }
  private:
    std::vector<DataSequence*> container;
	iVector* numSamplesPerClass;
	void searchNumberOfSamplesPerClass();
};

std::ostream& operator <<(std::ostream& out, const DataSequence& seq);
std::ostream& operator <<(std::ostream& out, const DataSet& data);

#endif
