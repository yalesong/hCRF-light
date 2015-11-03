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


#include "inferenceengine.h"
#ifdef _OPENMP
#include <omp.h>
#endif  

  
InferenceEngineLoopyBP::InferenceEngineLoopyBP(int max_iter, double error_threshold)
: InferenceEngine(), m_max_iter(max_iter), m_min_threshold(error_threshold) 
{}

InferenceEngineLoopyBP::~InferenceEngineLoopyBP()
{}

void InferenceEngineLoopyBP::computeBeliefs(Beliefs &beliefs, FeatureGenerator *fGen, 
	DataSequence *X, Model *m, int bComputePartition, int seqLabel, bool bUseStatePerNodes)
{  
	// Get adjacency matrix; values indicate edge index (use upper triangle only)
	iMatrix adjMat;	 
	if( m->isMultiViewMode() )
		m->getAdjacencyMatrixMV(adjMat, X);
	else {
		// Quick and dirty, but I don't want to change Model::makeChain()
		int edgeID = 1;
		adjMat.create(X->length(),X->length());
		for(int r=1; r<adjMat.getHeight(); r++) {
			adjMat(r,r-1) = edgeID;
			adjMat(r-1,r) = edgeID;
			edgeID++;
		}
	} 
	int adjMatMax = adjMat.getMaxValue();
 
	// 1. Initialize beliefs 
	initializeBeliefs(beliefs, X, m, adjMat); 

	// 2. Initialize messages
    std::vector<dVector> messages, prev_messages;
	initializeMessages(messages, X, m, adjMat, adjMatMax, false); 
	for(unsigned int i=0; i<messages.size(); i++) {
		dVector v; prev_messages.push_back(v);
	}

	// 3. Initialize potentials 
	Beliefs potentials;
	initializePotentials(potentials, fGen, X, m, adjMat, seqLabel, bUseStatePerNodes); 

	// 4. Update loopy belief network	
	int T = X->length(); 
	int V = m->getNumberOfViews(); 
	int nbNodes = V*T;
 
	int *messageUpdateOrder = new int[nbNodes];
	
	for(int iter=0; iter<m_max_iter; iter++)
	{
		// Get message update order
		getRandomUpdateOrder(messageUpdateOrder, X, m); 
		
		// Update messages
		for( int i=0; i<nbNodes; i++ ) {
			int xi = messageUpdateOrder[i];  
			for( int xj=0; xj<nbNodes; xj++ ) {
				if( !adjMat(xi,xj) ) continue; 
				sendMessage(xi,xj,nbNodes,potentials,messages,adjMat,adjMatMax,m->isMaxMargin());
			}		 
		}
	
		// Convergence check
		if( iter>0 ) {
			double error = 0;
			for(unsigned int i=0; i<messages.size(); i++) 
				for(int j=0; j<messages[i].getLength(); j++) 
					error += fabs(messages[i][j] - prev_messages[i][j]);
			if( error < m_min_threshold ) break;			
		}
		
		// Copy messages
		for(unsigned int i=0; i<messages.size(); i++) 
			prev_messages[i] = messages[i];	
	} 	 

	// Compute beliefs & compute partition
	updateBeliefs(nbNodes, beliefs, potentials, messages, adjMat, adjMatMax);  

	// 5. Clean up and Return 
	if( messageUpdateOrder ) { 
		delete[] messageUpdateOrder; messageUpdateOrder = 0;
	} 
}

double InferenceEngineLoopyBP::computePartition(FeatureGenerator *fGen, DataSequence *X, 
	Model *m, int seqLabel, bool bUseStatePerNodes)
{
	Beliefs beliefs;
	computeBeliefs(beliefs, fGen, X, m, false, seqLabel, bUseStatePerNodes); 

	return beliefs.partition;
}

 
///////////////////////////////////////////////////////////////////////////
// PRIVATE
//

void InferenceEngineLoopyBP::initializeBeliefs(Beliefs& b, DataSequence* X, Model* m, iMatrix adjMat)
{	
	if( !m->isMultiViewMode() ) {
		int T = X->length();
		// Initialize singleton potentials
		if( b.belStates.size() != T ) {
			b.belStates.resize(T);
			for(int xi=0; xi<T; xi++)
				b.belStates[xi].create(m->getNumberOfStates());
		}

		// Initialized pairwise potentials
		if( b.belEdges.size() != T-1 ) {
			b.belEdges.resize(T-1);
			for(int xi=0; xi<T-1; xi++)
				b.belEdges[xi].create(m->getNumberOfStates(),m->getNumberOfStates());
		}
	}
	else {
		int T = X->length(); 
		int V = m->getNumberOfViews(); 
		int nbNodes = V*T;  

		// Initialize singleton potentials
		if( b.belStates.size() != nbNodes ) {
			b.belStates.resize(nbNodes); 
			for(int xi=0; xi<nbNodes; xi++)
				b.belStates[xi].create(m->getNumberOfStatesMV(xi/T));  
		}

		// Initialize pairwise potentials.  
		if( b.belEdges.size() != adjMat.getMaxValue() ) {
			b.belEdges.resize(adjMat.getMaxValue());	
			for(int xi=0; xi<nbNodes; xi++) { 
				for(int xj=xi+1; xj<nbNodes; xj++) {
					if( !adjMat(xi,xj) ) continue;
					b.belEdges[adjMat(xi,xj)-1].create(
						m->getNumberOfStatesMV(xj/T),
						m->getNumberOfStatesMV(xi/T)); 
				}
			}
		}  
	}
}

void InferenceEngineLoopyBP::initializeMessages(std::vector<dVector>& messages, 
	DataSequence* X, Model* m, iMatrix adjMat, int adjMatMax, bool randomInit)
{	
	if( !m->isMultiViewMode() ) {
		int T = X->length();
		int nbMsg = 2*(T-1);

		// Initialize messages associated with pairwise potentials
		if( messages.size() != nbMsg  ) {
			messages.resize(nbMsg);
			for(int i=0; i<nbMsg; i++)
				messages[i].create(m->getNumberOfStates());
		}
	}
	else {		
		int T = X->length(); 
		int V = m->getNumberOfViews();
		int nbNodes = V*T; 
	 
		// Initialize messages associated with pairwise potentials.  
		if( messages.size() != 2*adjMatMax ) {
			messages.resize(2*adjMatMax);   
			for(int xi=0; xi<nbNodes; xi++) {
				for(int xj=xi+1; xj<nbNodes; xj++) {
					if( !adjMat(xi,xj) ) continue;
					// First half contains forward-msgs, second half contains backward-msgs
					messages[adjMat(xi,xj)-1].create(m->getNumberOfStatesMV(xj/T)); // m_ij(xj)	
					messages[adjMatMax+adjMat(xj,xi)-1].create(m->getNumberOfStatesMV(xi/T)); // m_ji(xi) 
			}	} 
		} 
	}
}


void InferenceEngineLoopyBP::initializePotentials(Beliefs& potentials, FeatureGenerator* fGen, 
	DataSequence *X, Model* m, iMatrix adjMat, int seqLabel, bool bUseStatePerNodes)
{
	initializeBeliefs(potentials, X, m, adjMat);
	
	int V = m->getNumberOfViews();
	int T = X->length();  
	int nbNodes = V*T;

	featureVector vecFeatures;

	feature *f;
	dVector *lambda = m->getWeights(); 	
	double val;
 
	// Singleton potentials
	for(int xi=0; xi<nbNodes; xi++) {
		fGen->getFeatures(vecFeatures,X,m,xi,-1,seqLabel);
		f = vecFeatures.getPtr();
		for(int k=0; k<vecFeatures.size(); k++, f++) {
			val = (*lambda)[f->globalId] * f->value;
			potentials.belStates[xi][f->nodeState] += val; 
		}  
	} 

	// Pairwise potentials. Values are symmetrical: pot(xi,xj)==pot(xj,xi).
	for(int xi=0; xi<nbNodes; xi++) {
		for(int xj=xi+1; xj<nbNodes; xj++) {
			if( !adjMat(xi,xj) ) continue;
			fGen->getFeatures(vecFeatures,X,m,xj,xi,seqLabel);
			f = vecFeatures.getPtr();
			for(int k=0; k<vecFeatures.size(); k++, f++) { 		
				val = (*lambda)[f->globalId] * f->value;
				potentials.belEdges[adjMat(xi,xj)-1](f->prevNodeState, f->nodeState) += val;
			}  
		}
	}    

	// For MV-LDCRF
	if( !bUseStatePerNodes ) return;
	if( !m->isMultiViewMode() ) {
		iMatrix* pStatesPerNode = m->getStateMatrix(X);
		for(int xi=0; xi<nbNodes; xi++) 
			for(int h=0; h<m->getNumberOfStates(); h++)
				if( pStatesPerNode->getValue(h,xi)==0 )
					potentials.belStates[xi][h] = -INF_VALUE;
	}
	else {
		std::vector<iMatrix> statesPerNodeMV;	
		for(int v=0; v<m->getNumberOfViews(); v++) {
			iMatrix spn(m->getNumberOfStatesMV(v),X->length());
			for(int t=0; t<X->length(); t++) {
				for(int h=0; h<m->getNumberOfStatesMV(v); h++) {
					spn(t,h) = m->getStatesPerLabelMV(v)(h,X->getStateLabels()->getValue(t));
				}
			}
			statesPerNodeMV.push_back(spn); 
		}
		for(int xi=0; xi<nbNodes; xi++) {		
			int v = xi/T; int t = xi%T; 
			for(int h=0; h<m->getNumberOfStatesMV(v); h++)
				if(statesPerNodeMV[v](t,h)==0) 
					potentials.belStates[xi][h] = -INF_VALUE;  
		}
	}
}  


// m_ij(x_j) = sum_xi {potential(i)*potential(i,j)*prod_{u \in N(i)\j} {m_ui(xi)}}
void InferenceEngineLoopyBP::sendMessage(int xi, int xj, int nbNodes, const Beliefs potentials, 
	std::vector<dVector>& messages, iMatrix adjMat, int adjMatMax, bool bMaxProd)
{   	 
	int max_hi=-1; // for Viterbi decoding

	// potential(i)
	dVector Vi(potentials.belStates[xi]); 

	// potential(i,j) 
	dMatrix Mij(potentials.belEdges[adjMat(xi,xj)-1]); 
	if( xi>xj ) Mij.transpose(); 
	
	// prod_{u \in N(i)\j} {m_ui(xi)}}
	int msg_idx; 
	for( int xu=0; xu<nbNodes; xu++ ) {
		if( !adjMat(xu,xi) || xu==xj ) continue; 
		msg_idx = (xu>xi) ? adjMatMax+adjMat(xu,xi)-1 : adjMat(xu,xi)-1;
		Vi.add(messages[msg_idx]);
	}

	// m_ij(xj) = Vi \dot Mij
	msg_idx = (xi>xj) ? adjMatMax+adjMat(xi,xj)-1 : adjMat(xi,xj)-1; 
	if( bMaxProd )
		max_hi = logMultiplyMaxProd(Vi, Mij, messages[msg_idx]);
	else
		logMultiply(Vi, Mij, messages[msg_idx]); 
	
	// Normalize messages to avoid numerical over/under-flow
	// Make \sum_{xj} m_ij(xj)=1. Other methods could also be used.	 
	double min = messages[msg_idx].min();
	if( min < 0 ) messages[msg_idx].add(-min);
	messages[msg_idx].multiply(1/messages[msg_idx].sum()); 
} 
  

// \sum_{xj} m_ij(xj) = 1
void InferenceEngineLoopyBP::normalizeMessage(int xi,
	std::vector<dVector>& messages, iMatrix adjMat, int adjMatMax) 
{ 
	bool shift = true;
	double sum = 0;
	// 1. Shift and compute sum
	for(int xj=0; xj<adjMat.getWidth(); xj++) {
		if( !adjMat(xi,xj) ) continue;
		int msg_idx = (xi>xj) ? adjMatMax+adjMat(xi,xj)-1 : adjMat(xi,xj)-1;
		if( shift ) {
			double min = messages[msg_idx].min();
			if( min < 0 ) messages[msg_idx].add(-min);
		}
		sum += messages[msg_idx].sum();			
	}

	// 2. \sum_{xj} m_ij(xj) = 1
	for(int xj=0; xj<adjMat.getWidth(); xj++) {
		if( !adjMat(xi,xj) ) continue;
		int msg_idx = (xi>xj) ? adjMatMax+adjMat(xi,xj)-1 : adjMat(xi,xj)-1;
		messages[msg_idx].multiply(1/sum);
		//printf("[ "); for(int i=0 ;i<messages[msg_idx].getLength(); i++) printf("%f ", messages[msg_idx][i]); printf("]\n");
	}
}  


// b_i(xi)     = potential(i) * prod_{u \in N(i)}{m_ui}
// b_ij(xi,xj) = potential(i) * potential(j) * potential(i,j)
//			   * prod_{u \in N(i)\j}{m_ui} * prod_{u \in N(j)\i}{m_uj}
void InferenceEngineLoopyBP::updateBeliefs(int nbNodes, Beliefs& beliefs, 
	const Beliefs potentials, const std::vector<dVector> messages, 
	iMatrix adjMat, int adjMatMax) 
{ 
	int msg_idx;
	dVector Vi, Vj;
	dMatrix Mij;  

	// b_i(xi) = potential(i) * prod_{u \in N(i)}{m_ui}
	for(int xi=0; xi<nbNodes; xi++) {
		beliefs.belStates[xi].set(potentials.belStates[xi]); 
		for(int xu=0; xu<nbNodes; xu++) {
			if( !adjMat(xu,xi) ) continue;
			msg_idx = (xu>xi) ? adjMatMax+adjMat(xu,xi)-1 : adjMat(xu,xi)-1;
			beliefs.belStates[xi].add(messages[msg_idx]);
		}	  
	}

	// b_ij(xi,xj) = potential(i) * potential(j) * potential(i,j)
	//			   * prod_{u \in N(i)\j}{m_ui} * prod_{u \in N(j)\i}{m_uj}	 
	for(int xi=0; xi<nbNodes; xi++) {
		for(int xj=xi+1; xj<nbNodes; xj++ ) { // xj starts from xi+1 because b_ij==b_ji
			if( !adjMat(xi,xj) ) continue;

			// potential(i) * prod_{u \in N(i)\j){m_ui}
			Vi.set(potentials.belStates[xi]);
			for(int xu=0; xu<nbNodes; xu++ ) {
				if( !adjMat(xu,xi) || xu==xj ) continue;
				msg_idx = (xu>xi) ? adjMatMax+adjMat(xu,xi)-1 : adjMat(xu,xi)-1; 
				Vi.add(messages[msg_idx]);
			}

			// potential(j) * prod_{u \in N(j)\i){m_uj}
			Vj.set(potentials.belStates[xj]);
			for(int xu=0; xu<nbNodes; xu++ ) {
				if( !adjMat(xu,xj) || xu==xi ) continue;
				msg_idx = (xu>xj) ? adjMatMax+adjMat(xu,xj)-1 : adjMat(xu,xj)-1; 
				Vj.add(messages[msg_idx]);
			}			

			// potential(i,j) * Vi*Vj*Mij
			Mij.create( Vj.getLength(), Vi.getLength() );
			logMultiply(Vi,Vj,Mij);
			Mij.add(potentials.belEdges[adjMat(xi,xj)-1]);
			beliefs.belEdges[adjMat(xi,xj)-1].set(Mij);
		}
	}   

	// Normalize beliefs and compute partition
	double logZ = 0;
	beliefs.partition = 0;
	for(unsigned int i=0; i<beliefs.belStates.size(); i++) {
		logZ = beliefs.belStates[i].logSumExp();
		beliefs.belStates[i].add(-logZ);
		beliefs.belStates[i].eltExp();
		beliefs.partition += logZ;
	}
	for(unsigned int i=0; i<beliefs.belEdges.size(); i++) {
		logZ = beliefs.belEdges[i].logSumExp();
		beliefs.belEdges[i].add(-logZ);
		beliefs.belEdges[i].eltExp();
		beliefs.partition += logZ;
	}
}
  

void InferenceEngineLoopyBP::getSequentialUpdateOrder(int* list, DataSequence* X, Model* m)
{
	int V = m->getNumberOfViews();
	int T = X->length(); 

	// sequential in time. 
	// e.g., (t=0,v=0),(t=0,v=1),(t=1,v=0),(t=1,v=1),...
	int cnt=0;
	for( int t=0; t<T; t++ ) { for( int v=0; v<V; v++ ) { 
		int i = t*V + v;
		list[cnt++] = i;
	} } 
}


void InferenceEngineLoopyBP::getRandomUpdateOrder(int* list, DataSequence* X, Model* m)
{
	int V = m->getNumberOfViews();
	int T = X->length(); 
	int nbNodes = V*T;

	// create an ordered list 
	for(int i=0; i<nbNodes; i++)
		list[i] = i;

	// swap the i-th element with a randomly chosen element
	for(int i=0; i<nbNodes; i++) {
		int swap_idx = rand()%(V*T);
		int tmp = list[i];
		list[i] = list[swap_idx];
		list[swap_idx] = tmp;
	}	
}
