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
 
InferenceEngineBP::InferenceEngineBP(bool isp): InferenceEngine(), isSumProduct(isp)
{}

InferenceEngineBP::~InferenceEngineBP()
{}

void InferenceEngineBP::computeBeliefs(Beliefs &beliefs, FeatureGenerator *fGen, 
	DataSequence *X, Model *m, int bComputePartition, int seqLabel, bool bUseStatePerNodes)
{  
	// Variable definition  
	int xi, xj, nbNodes, seqLength;
	std::map<int,BPNode*> nodes; // tree graph
	std::map<int,BPNode*>::iterator itm;
	std::list<BPNode*>::iterator itl;
	BPNode* root;
	iMatrix adjMat;
	iVector nbStates;
	dVector*  phi_i  = 0; // singleton potentials
	dMatrix** phi_ij = 0; // pairwise potentials
	dVector** msg    = 0; // messages

	if( m->isMultiViewMode() )
		m->getAdjacencyMatrixMV(adjMat, X);
	else {
		uMatrix uAdjMat;
		m->getAdjacencyMatrix(uAdjMat, X);
		adjMat.resize(uAdjMat.getWidth(),uAdjMat.getHeight());
		for(xi=0; xi<uAdjMat.getHeight(); xi++)
			for(xj=0; xj<uAdjMat.getWidth(); xj++)
				adjMat(xi,xj) = uAdjMat(xi,xj);
	}

	nbNodes = adjMat.getHeight(); 
	seqLength = X->length();

	// Create a vector that contains nbStates
	nbStates.create(nbNodes);
	for(xi=0; xi<nbNodes; xi++)
		nbStates[xi] = (m->isMultiViewMode()) 
			? m->getNumberOfStatesMV(xi/seqLength) : m->getNumberOfStates();

	// Create BPGraph from adjMat
	for(xi=0; xi<nbNodes; xi++) {		
		BPNode* v = new BPNode(xi, nbStates[xi]);
		nodes.insert( std::pair<int,BPNode*>(xi,v) );
	}
	for(xi=0; xi<nbNodes; xi++) {
		for(xj=xi+1; xj<nbNodes; xj++) {
			if( !adjMat(xi,xj) ) continue;
			nodes[xi]->addNeighbor(nodes[xj]);
			nodes[xj]->addNeighbor(nodes[xi]);
		}
	}

	// Initialize  
	initMessages(msg, X, m, adjMat, nbStates);
	initBeliefs(beliefs, X, m, adjMat, nbStates);
	initPotentials(phi_i, phi_ij, fGen, X, m, adjMat, nbStates, seqLabel);
	
	// Message update
	root = nodes[0]; // any node can be the root node
	{
		for(itl=root->neighbors.begin(); itl!=root->neighbors.end(); itl++)
			collect(root, *itl, phi_i, phi_ij, msg);
		for(itl=root->neighbors.begin(); itl!=root->neighbors.end(); itl++)
			distribute(root, *itl, phi_i, phi_ij, msg);
	}
	updateBeliefs(beliefs, phi_i, phi_ij, msg, X, m, adjMat);

	// Clean up
	for(xi=0; xi<nbNodes; xi++) { 		
		delete[] msg[xi]; msg[xi] = 0; 
		delete[] phi_ij[xi]; phi_ij[xi] = 0;
	}
	delete[] msg; msg=0;
	delete[] phi_i; phi_i = 0;
	delete[] phi_ij;  phi_ij  = 0; 

	for(itm=nodes.begin(); itm!=nodes.end(); itm++) 
		delete (*itm).second; 
	nodes.clear();   
}

double InferenceEngineBP::computePartition(FeatureGenerator *fGen, DataSequence *X, 
	Model *m, int seqLabel, bool bUseStatePerNodes)
{
	Beliefs beliefs;
	computeBeliefs(beliefs, fGen, X, m, false, seqLabel, bUseStatePerNodes); 
	return beliefs.partition;
}

/////////////////////////////////////////////////////////////////////////////////
// Private
void InferenceEngineBP::collect(BPNode* xi, BPNode* xj, dVector* phi_i, dMatrix** phi_ij, dVector** m)
{
	std::list<BPNode*>::iterator it;
	for(it=xj->neighbors.begin(); it!=xj->neighbors.end(); it++) {
		if( xi->equal(*it) ) continue;
		collect(xj,*it,phi_i,phi_ij,m);
	}
	sendMessage(xj,xi,phi_i,phi_ij,m);		
}

void InferenceEngineBP::distribute(BPNode* xi, BPNode* xj, dVector* phi_i, dMatrix** phi_ij, dVector** m)
{
	std::list<BPNode*>::iterator it;
	sendMessage(xi,xj,phi_i,phi_ij,m);
	for(it=xj->neighbors.begin(); it!=xj->neighbors.end(); it++) {
		if( xi->equal(*it) ) continue;
		distribute(xj,*it,phi_i,phi_ij,m);
	}
}

// m_ij(x_j) = sum_xi {phi(i)*phi(i,j)*prod_{u \in N(i)\j} {m_uj(xi)}}
// m_ij(x_j) = max_xi {phi(i)*phi(i,j)*prod_{u \in N(i)\j} {m_uj(xi)}}
void InferenceEngineBP::sendMessage(BPNode* xi, BPNode* xj, dVector* phi_i, dMatrix** phi_ij, dVector** msg)
{    
	// potential(i) -> Vi
	dVector Vi(phi_i[xi->id]);
	
	// potential(i,j) -> Mij
	dMatrix Mij;
	if( xi->id < xj->id )
		Mij.set( phi_ij[xi->id][xj->id] );
	else {
		Mij.set( phi_ij[xj->id][xi->id] );
		Mij.transpose();
	}

	// prod_{u \in N(i)\j} {m_ui(xi)}	-> Vi
	std::list<BPNode*>::iterator it;
	for(it=xi->neighbors.begin(); it!=xi->neighbors.end(); it++) {
		if( xj->equal(*it) ) continue;
		Vi.add( msg[(*it)->id][xi->id] );
	}

	if( isSumProduct )
		logMultiply( Vi, Mij, msg[xi->id][xj->id] ); 
	else
		logMultiplyMaxProd( Vi, Mij, msg[xi->id][xj->id] ); 
}  
	
void InferenceEngineBP::initMessages(dVector**& msg, DataSequence* X, Model* m, iMatrix adjMat, iVector nbStates)
{
	int xi, xj, nbNodes, seqLength;
	nbNodes   = adjMat.getHeight();
	seqLength = X->length();

	msg = new dVector*[nbNodes];
	for(xi=0; xi<nbNodes; xi++) {
		msg[xi] = new dVector[nbNodes];
		for(xj=0; xj<nbNodes; xj++) 
			if( adjMat(xi,xj) ) {
				msg[xi][xj].create(nbStates[xj]);
			}
	}
}

void InferenceEngineBP::initBeliefs(Beliefs& b, DataSequence* X, Model* m, iMatrix adjMat, iVector nbStates)
{
	int xi, xj, nbNodes, nbEdges, seqLength;
	nbNodes   = adjMat.getHeight();
	nbEdges   = nbNodes - 1;
	seqLength = X->length();

	b.belStates.resize(nbNodes);
	for(xi=0; xi<nbNodes; xi++) { 
		b.belStates[xi].create(nbStates[xi]);
	}

	b.belEdges.resize(nbEdges);	
	for(xi=0; xi<nbNodes; xi++) {
		for(xj=xi+1; xj<nbNodes; xj++) {
			if( !adjMat(xi,xj) ) continue;
			b.belEdges[adjMat(xi,xj)-1].create(nbStates[xj],nbStates[xi]);			
		}
	}
}

void InferenceEngineBP::initPotentials(dVector*& phi_i, dMatrix**& phi_ij, 
	FeatureGenerator* fGen, DataSequence* X, Model* m, iMatrix adjMat, iVector nbStates, int seqLabel)
{
	int k, xi, xj, nbNodes, seqLength;
	nbNodes   = adjMat.getHeight();
	seqLength = X->length();

	// init singleton potentials
	phi_i = new dVector[nbNodes];
	for(xi=0; xi<nbNodes; xi++) {
		phi_i[xi].create(nbStates[xi]);
	}

	// init pairwise potentials
	phi_ij = new dMatrix*[nbNodes];
	for(xi=0; xi<nbNodes; xi++) {
		phi_ij[xi] = new dMatrix[nbNodes];
		for(xj=xi+1; xj<nbNodes; xj++) // ALWAYS (xj>xi)
			if( adjMat(xi,xj) )
				phi_ij[xi][xj].create(nbStates[xj],nbStates[xi]);
	} 
 
	//
	// Assign evidence to potentials
	feature *f;
	featureVector vecFeatures;

	dVector *lambda = m->getWeights(); 	
	double val;

	// singleton potentials
	for(xi=0; xi<nbNodes; xi++) {
		fGen->getFeatures(vecFeatures,X,m,xi,-1,seqLabel);
		f = vecFeatures.getPtr();
		for(k=0; k<vecFeatures.size(); k++, f++) {
			val = (*lambda)[f->globalId] * f->value;
			phi_i[xi].addValue(f->nodeState,val);
		}
	}	
	// pairwise potentials
	for(xi=0; xi<nbNodes; xi++) {
		for(xj=xi+1; xj<nbNodes; xj++) {
			if( !adjMat(xi,xj) ) continue;
			fGen->getFeatures(vecFeatures,X,m,xj,xi,seqLabel);
			f = vecFeatures.getPtr();
			for(k=0; k<vecFeatures.size(); k++, f++) {
				val = (*lambda)[f->globalId] * f->value;
				phi_ij[xi][xj].addValue(f->prevNodeState,f->nodeState,val);
			}
		}
	}

}
 
// b_i(xi) = potential(i) * prod_{u \in N(i)}{m_ui}
// b_ij(xi,xj) = potential(i) * potential(j) * potential(i,j)
//			     * prod_{u \in N(i)\j}{m_ui} * prod_{u \in N(j)\i}{m_uj}
void InferenceEngineBP::updateBeliefs(Beliefs& b, dVector* phi_i, dMatrix** phi_ij, 
	dVector** msg, DataSequence* X, Model* m, iMatrix adjMat)
{
	int xi, xj, xu, nbNodes, seqLength;
	dVector Vi, Vj;
	dMatrix Mij;
	
	nbNodes = adjMat.getHeight();
	seqLength = X->length();

	// b_i(xi) = potential(i) * prod_{u \in N(i)}{m_ui}
	for(xi=0; xi<nbNodes; xi++) {
		b.belStates[xi].set( phi_i[xi] ); 
		for(int xu=0; xu<nbNodes; xu++) {
			if( !adjMat(xu,xi) ) continue;
			b.belStates[xi].add( msg[xu][xi] );
		}	  
	}

	// b_ij(xi,xj) = potential(i) * potential(j) * potential(i,j)
	//			   * prod_{u \in N(i)\j}{m_ui} * prod_{u \in N(j)\i}{m_uj}	 
	for(xi=0; xi<nbNodes; xi++) {
		for(xj=xi+1; xj<nbNodes; xj++) { // xj starts from xi+1 because b_ij==b_ji
			if( !adjMat(xi,xj) ) continue;

			// potential(i) * prod_{u \in N(i)\j){m_ui}
			Vi.set( phi_i[xi] );
			for(xu=0; xu<nbNodes; xu++ ) {
				if( !adjMat(xu,xi) || xu==xj ) continue;
				Vi.add( msg[xu][xi] );
			}

			// potential(j) * prod_{u \in N(j)\i){m_uj}
			Vj.set( phi_i[xj] );
			for(xu=0; xu<nbNodes; xu++ ) {
				if( !adjMat(xu,xj) || xu==xi ) continue;
				Vj.add( msg[xu][xj] );
			}

			// (Vi*Vj*Mij) * potential(i,j)
			Mij.create(Vj.getLength(), Vi.getLength());
			logMultiply(Vi, Vj, Mij);
			Mij.add( phi_ij[xi][xj] );

			if( m->isMultiViewMode() )
				b.belEdges[adjMat(xi,xj)-1].set(Mij);
			else
				b.belEdges[xi].set(Mij);
		}
	}   

	// Normalize beliefs and compute partition
	unsigned int i;
	double logZ = 0;
	// marginals are consistent across nodes
	logZ = b.belStates[0].logSumExp(); 
	for(i=0; i<b.belStates.size(); i++) {
		b.belStates[i].add(-logZ);
		b.belStates[i].eltExp();
	}
	for(i=0; i<b.belEdges.size(); i++) {
		b.belEdges[i].add(-logZ);
		b.belEdges[i].eltExp();
	} 
	b.partition = logZ;
}


void InferenceEngineBP::printBeliefs(Beliefs beliefs)
{
	for(int i=0; i<(int)beliefs.belStates.size(); i++) {
		for(int r=0; r<beliefs.belStates[i].getLength(); r++ )
			printf("%f\n", beliefs.belStates[i][r]);
		printf("------------------------------------------------------------\n");
	}

	for(int i=0; i<(int)beliefs.belEdges.size(); i++) {
		for(int r=0; r<beliefs.belEdges[i].getHeight(); r++) {
			for(int c=0; c<beliefs.belEdges[i].getWidth(); c++)
				printf("%f\t", beliefs.belEdges[i](r,c));
			printf("\n");
		}		
		printf("------------------------------------------------------------\n");
	}
}
  
///////////////////////////////////////////////////////////////////////////
// Implementation of the class BPNode
//  
BPNode::BPNode(int node_id, int node_size): id(node_id), size(node_size) {}
BPNode::~BPNode() {}
void BPNode::addNeighbor(BPNode* v) { neighbors.push_back(v); }
bool BPNode::equal(BPNode* v) { return id==v->id; }
void BPNode::print()
{
	std::list<BPNode*>::iterator it;
	printf("[%d] {", id); 
	for(it=neighbors.begin(); it!=neighbors.end(); it++) printf("%d ", (*it)->id); printf("}\n");
}