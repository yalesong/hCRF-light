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


#ifdef _OPENMP
#include <omp.h>
#endif

#include "hcrf/inferenceengine.h"

///////////////////////////////////////////////////////////////////////////
// CONSTRUCTOR / DESTRUCTOR
//
InferenceEngineJT::InferenceEngineJT(): InferenceEngine() {}
InferenceEngineJT::~InferenceEngineJT() {}


///////////////////////////////////////////////////////////////////////////
// PUBLIC
//
void InferenceEngineJT::computeBeliefs(Beliefs &beliefs,FeatureGenerator *fGen,
                                       DataSequence *X, Model *m, int bComputePartition, int seqLabel, bool bUseStatePerNodes)
{
  iMatrix adjMat, JTadjMat;
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
		
  // Create a vector that contains nbStates
  int nbNodes = adjMat.getHeight();
  int seqLength = X->length();
  iVector nbStates(nbNodes);
  for(int xi=0; xi<nbNodes; xi++)
    nbStates[xi] = (m->isMultiViewMode())
    ? m->getNumberOfStatesMV(xi/seqLength) : m->getNumberOfStates();
  
  // 1. Construct a junction tree
  std::vector<Clique*> vecCliques;
  std::vector<Separator*> vecSeparators;
  constructTriangulatedGraph(vecCliques,m,X,adjMat,nbStates);
  buildJunctionTree(vecCliques,vecSeparators,m,X,JTadjMat,nbStates);
  removeRedundantNodes(vecCliques,vecSeparators,m,X,JTadjMat);
  
  // 2. For computational efficiency, find the index mapping between nodes and cliques
  iVector vecNodeToClique(adjMat.getHeight());
  iMatrix matNodeToClique(adjMat.getWidth(),adjMat.getHeight());
  findNodeToCliqueIndexMap(m,X,vecCliques,vecNodeToClique,matNodeToClique,adjMat);
  
  // 3. Assign potentials to cliques
  initializePontentials(vecCliques,vecSeparators,fGen,m,X,seqLabel,adjMat,
                        vecNodeToClique,matNodeToClique,bUseStatePerNodes);
  
  // 4. Update junction tree
  iMatrix untouched;
  untouched.set(JTadjMat); collectEvidence(vecCliques,0,untouched,m->isMaxMargin());
  untouched.set(JTadjMat); distributeEvidence(vecCliques,0,untouched,m->isMaxMargin());
  //checkConsistency(vecCliques,vecSeparators,m->isMaxMargin()); getchar();
  
  // 5. Compute beliefs
  initializeBeliefs(beliefs,X,m,adjMat,nbStates);
  updateBeliefs(vecCliques,vecSeparators,beliefs,m,X,adjMat,nbStates,
                vecNodeToClique,matNodeToClique,m->isMaxMargin());
  
  // 6. Done. Clean up variables
  for(int i=0; i<(int)vecCliques.size(); i++) delete vecCliques.at(i); vecCliques.clear();
  for(int i=0; i<(int)vecSeparators.size(); i++) delete vecSeparators.at(i); vecSeparators.clear();
}

double InferenceEngineJT::computePartition(FeatureGenerator *fGen, DataSequence *X,
                                           Model *m, int seqLabel, bool bUseStatePerNodes)
{
  Beliefs beliefs;
  computeBeliefs(beliefs, fGen, X, m, false, seqLabel, bUseStatePerNodes);
  return beliefs.partition;
}



///////////////////////////////////////////////////////////////////////////
// PRIVATE
//

void InferenceEngineJT::constructTriangulatedGraph(
                                                   std::vector<Clique*> &vecCliques, Model* m, DataSequence* X, iMatrix adjMat, iVector nbStates)
{
  int node=0;
  int num_states=0;
  
  std::list<int>::iterator ni, nj;
  while( adjMat.getMaxValue() )
  {
    // Get the next node to eliminate in the triangulation process
    // (heuristic: fewest edges)
    node = getNextElimination(adjMat);
    num_states = nbStates[node];
    
    // Create a new clique to be added
    Clique *clique = new Clique();
    clique->vars.push_back(node);
    clique->cardinalities.push_back(num_states);
    
    // Add all the neighboring nodes to this cluster node
    for(int c=0; c<adjMat.getWidth(); c++) {
      if( adjMat(node,c) ) {
        num_states = nbStates[c];
        clique->vars.push_back(c);
        clique->cardinalities.push_back(num_states);
        adjMat(node,c) = adjMat(c,node) = 0;
      }	}
    
    // Sort vars AND cardinalities in the order of vars
    clique->sort();
    
    // Triangulate neighboring edges
    for(ni=clique->vars.begin(); ni!=clique->vars.end(); ni++) {
      if( (*ni)==node ) continue;
      for(nj=clique->vars.begin(); nj!=clique->vars.end(); nj++) {
        if( (*ni)==(*nj) || (*nj)==node ) continue;
        adjMat((*ni),(*nj)) = adjMat((*nj),(*ni)) = 1;
      }	}
    vecCliques.push_back(clique);
  }
  // Sort vecCliques according to their front-node number
  std::sort( vecCliques.begin(), vecCliques.end(), &InferenceEngineJT::CliqueSortPredicate );
}

void InferenceEngineJT::buildJunctionTree(
                                          std::vector<Clique*> &vecCliques, std::vector<Separator*> &vecSeparators,
                                          Model* m, DataSequence* X, iMatrix& JTadjMat, iVector nbStates)
{
  if( JTadjMat.getWidth() != (int) vecCliques.size() )
    JTadjMat.resize((int)vecCliques.size(), (int)vecCliques.size());
  
  std::list<int>::iterator ni, nj;
  
  // [1] Extract edge weights
  iMatrix edgeWeights((int)vecCliques.size(), (int)vecCliques.size());
  for(int i=0; i<(int)vecCliques.size(); i++) {
    for(int j=i+1; j<(int)vecCliques.size(); j++) {
      int seperatorSize = 0;
      for(ni=vecCliques[i]->vars.begin(); ni!=vecCliques[i]->vars.end(); ni++)
        for(nj=vecCliques[j]->vars.begin(); nj!=vecCliques[j]->vars.end(); nj++)
          seperatorSize += (*ni)==(*nj);
      edgeWeights(i,j) = edgeWeights(j,i) = seperatorSize;
    }	}
  
  // [2] Kruskal's maximum spanning tree algorithm
  // For each vertex in G, define an elementary cluster C(v) <- {v}
  std::map<int, std::list<int> > clusters;
  for(int i=0; i<(int)vecCliques.size(); i++) {
    std::list<int> cluster;
    cluster.push_back(i);
    clusters.insert( std::pair<int,std::list<int> >(i,cluster) );
  }
  
  int max_w, max_ei, max_ej;
  while(edgeWeights.getMaxValue()!=0) {
    // Choose an edge with the maximum weight
    max_w = max_ei = max_ej = -1;
    for(int r=0; r<edgeWeights.getHeight(); r++)
      for(int c=r+1; c<edgeWeights.getWidth(); c++)
        if( edgeWeights(r,c) > max_w ) {
          max_ei = r; max_ej = c;
          max_w = edgeWeights(r,c);
        }
    edgeWeights(max_ei,max_ej) = edgeWeights(max_ej,max_ei) = 0;
    
    // If C(max_ei) != C(max_ej), add the edge and combine the two clusters
    std::list<int> *val_ei = &(clusters.find(max_ei)->second);
    std::list<int> *val_ej = &(clusters.find(max_ej)->second);
    
    if( (*val_ei)!=(*val_ej) ) {
      JTadjMat(max_ei,max_ej) = JTadjMat(max_ej,max_ei) = 1;
      val_ei->merge(*val_ej);
      for(ni=val_ei->begin(); ni!=val_ei->end(); ni++)
        if( (*ni)!=max_ei ) clusters.find(*ni)->second = *val_ei;
    }
  }
  
  // [3] Extract separators
  for(int r=0; r<JTadjMat.getHeight(); r++) {
    for(int c=r+1; c<JTadjMat.getWidth(); c++) {
      if( JTadjMat(r,c) ) {
        Separator* s = new Separator(vecCliques[r],vecCliques[c]);
        s->vars = getIntersect(vecCliques[r],vecCliques[c]);
        for(ni=s->vars.begin(); ni!=s->vars.end(); ni++) {
          int num_states = nbStates[*ni];
          s->cardinalities.push_back(num_states);
        }
        s->sort();
        vecCliques[r]->add_neighbor(s, vecCliques[c]);
        vecCliques[c]->add_neighbor(s, vecCliques[r]);
        vecSeparators.push_back(s);
      }	}	}
}

void InferenceEngineJT::removeRedundantNodes(
                                             std::vector<Clique*> &vecCliques, std::vector<Separator*> &vecSeparators,
                                             Model* model, DataSequence* X, iMatrix& JTadjMat)
{
  // After a junction tree is built, it often contains redundant nodes
  // where a clique and one of its separators have identical sets of vars.
  // In such cases, the separator can be deleted, and neighboring two
  // cliques can be merged.
  Clique *A, *B; Separator *S;
  std::vector<Clique*>::iterator itc;
  std::vector<Separator*>::iterator its;
  std::map<Clique*,Separator*>::iterator itcs;
  int idx_tmp;
  
  for(itc=vecCliques.begin(); itc!=vecCliques.end(); itc++) {
    A = *itc;
    for(itcs=A->neighbors.begin(); itcs!=A->neighbors.end(); )  {
      B = (*itcs).first; S = (*itcs).second;
      if( !B->equals(S) ) { ++itcs; continue; }
      // Redundant node found!
      // \forall (B',S')\in B.neighbors(), make S' point to A
      Clique* B2; Separator* S2;
      std::map<Clique*,Separator*>::iterator itcs2;
      for(itcs2=B->neighbors.begin(); itcs2!=B->neighbors.end(); itcs2++) {
        B2 = (*itcs2).first; S2 = (*itcs2).second;
        if( B2->equals(A) ) continue;
        if( S2->clique_A->equals(B) )
          S2->clique_A = A;
        else if( S2->clique_B->equals(B) )
          S2->clique_B = A;
        A->add_neighbor(S2,B2);
      }
      idx_tmp = findCliqueOffset(vecCliques,B);
      delete B; vecCliques.erase( vecCliques.begin()+idx_tmp );
      idx_tmp = findSeparatorOffset(vecSeparators,S);
      delete S; vecSeparators.erase( vecSeparators.begin()+idx_tmp );
      A->neighbors.erase(itcs++);
    }
  }
  
  // Update JTadjMat
  if( JTadjMat.getWidth() != (int) vecCliques.size() )
    JTadjMat.resize((int)vecCliques.size(), (int)vecCliques.size());
  JTadjMat.set(0);
  for(its=vecSeparators.begin(); its!=vecSeparators.end(); its++) {
    int idx_A = findCliqueOffset(vecCliques, (*its)->clique_A);
    int idx_B = findCliqueOffset(vecCliques, (*its)->clique_B);
    JTadjMat(idx_A,idx_B) = JTadjMat(idx_B,idx_A) = 1;
  }
}

int InferenceEngineJT::findCliqueOffset(
                                        std::vector<Clique*> vecCliques, Clique *clique)
{
  int i; for(i=0; i<(int)vecCliques.size(); i++)
    if( vecCliques[i]->equals(clique) ) break;
  return i;
}

int InferenceEngineJT::findSeparatorOffset(
                                           std::vector<Separator*> vecSeparators, Separator *separator)
{
  int i; for(i=0; i<(int)vecSeparators.size(); i++)
    if( vecSeparators[i]->equals(separator) ) break;
  return i;
}


void InferenceEngineJT::findNodeToCliqueIndexMap(Model* m, DataSequence* X,
                                                 std::vector<Clique*> vecCliques, iVector& vec, iMatrix& mat, iMatrix adjMat)
{
  int T = X->length();
  int V = m->getNumberOfViews();
  int nbNodes = V*T;
  // Find vecNodeToCliqueIndex
  for(int xi=0; xi<nbNodes; xi++) {
    for(int i=0; i<(int)vecCliques.size(); i++) {
      if( vecCliques[i]->contains(xi) ) {
        vec[xi] = i;
        break;
      }	}	}
  
  // Find matNodeToCliqueIndex
  for(int xi=0; xi<nbNodes; xi++) {
    for(int xj=xi+1; xj<nbNodes; xj++) {
      if( !adjMat(xi,xj) ) continue;
      for(int i=0; i<(int)vecCliques.size(); i++) {
        if( vecCliques[i]->contains(xi,xj) ) {
          mat(xi,xj)=mat(xj,xi) = i;
          break;
        }	}	}	}
}


void InferenceEngineJT::initializePontentials(
                                              std::vector<Clique*> &vecCliques, std::vector<Separator*> &vecSeparators,
                                              FeatureGenerator* fGen, Model* m, DataSequence* X, int seqLabel,
                                              iMatrix adjMat, iVector vecN2C, iMatrix matN2C, bool bUseStatePerNodes)
{
  std::vector<Clique*>::iterator itc;
  for(itc=vecCliques.begin(); itc!=vecCliques.end(); itc++)
    (*itc)->initialize();
  std::vector<Separator*>::iterator its;
  for(its=vecSeparators.begin(); its!=vecSeparators.end(); its++)
    (*its)->initialize();
  
  int T = X->length();
  int V = m->getNumberOfViews();
  int nbNodes = V*T;
  
  featureVector vecFeatures;
  
  feature *f;
  const dVector *lambda = m->getWeights();
  
  // Singleton potentials
  for(int xi=0; xi<nbNodes; xi++) {
    fGen->getFeatures(vecFeatures,X,m,xi,-1,seqLabel);
    f = vecFeatures.getPtr();
    for(int k=0; k<vecFeatures.size(); k++, f++) {
      vecCliques[vecN2C[xi]]->assign_potential(
                                               xi,f->nodeState,(*lambda)[f->globalId]*f->value);
    }
  }
  
  // Pairwise potentials
  for(int xi=0; xi<nbNodes; xi++) {
    for(int xj=xi+1; xj<nbNodes; xj++) {
      if( !adjMat(xi,xj) ) continue;
      fGen->getFeatures(vecFeatures,X,m,xj,xi,seqLabel);
      f = vecFeatures.getPtr();
      for(int k=0; k<vecFeatures.size(); k++, f++) {
        vecCliques[matN2C(xi,xj)]->assign_potential(
                                                    xi,xj,f->prevNodeState,f->nodeState,(*lambda)[f->globalId]*f->value);
      }	}  }
  
  if( !bUseStatePerNodes ) return;
  
  // Below: for MV-LDCRF
  if( !m->isMultiViewMode() ) {
    iMatrix* pStatesPerNode = m->getStateMatrix(X);
    for(int xi=0; xi<nbNodes; xi++) {
      for(int h=0; h<m->getNumberOfStates(); h++)
        if( pStatesPerNode->getValue(h,xi)==0 )
          vecCliques[vecN2C[xi]]->assign_potential(xi,h,-INF_VALUE);
    }
    
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
          vecCliques[vecN2C[xi]]->assign_potential(xi,h,-INF_VALUE);
    }
  }
}

void InferenceEngineJT::collectEvidence(
                                        std::vector<Clique*> vecCliques, int node_idx, iMatrix& untouched, bool bMax)
{
  for(int i=0; i<untouched.getWidth(); i++) {
    if( untouched(node_idx,i)>0 ) {
      untouched(node_idx,i) = untouched(i,node_idx) = 0;
      collectEvidence(vecCliques, i, untouched, bMax);
      update(vecCliques[i], vecCliques[node_idx], bMax);
    }
  }
}
void InferenceEngineJT::distributeEvidence(
                                           std::vector<Clique*> vecCliques, int node_idx, iMatrix& untouched, bool bMax)
{
  for(int i=0; i<untouched.getWidth(); i++) {
    if( untouched(node_idx,i)>0 ) {
      untouched(node_idx,i) = untouched(i,node_idx) = 0;
      update(vecCliques[node_idx], vecCliques[i], bMax);
      distributeEvidence(vecCliques, i, untouched, bMax);
    }
  }
}

void InferenceEngineJT::update(Clique* A, Clique* B, bool bMax)
{
  Separator* S = A->neighbors.find(B)->second;
  S->update_ratio(A->marginalize(S->vars,bMax));
  B->scale(S->vars, S->ratio);;
}



void InferenceEngineJT::initializeBeliefs(
                                          Beliefs& b, DataSequence* X, Model* m, iMatrix adjMat, iVector nbStates)
{
  int T = X->length();
  int V = m->getNumberOfViews();
  int nbNodes = V*T;
  
  // Initialize singleton potentials
  if( b.belStates.size() != nbNodes ) {
    b.belStates.resize(nbNodes);
    for(int xi=0; xi<nbNodes; xi++) {
      b.belStates[xi].create(nbStates[xi]);
    }
  }
  
  // Initialize pairwise potentials.
  if( b.belEdges.size() != adjMat.getMaxValue() ) {
    b.belEdges.resize(adjMat.getMaxValue());
    for(int xi=0; xi<nbNodes; xi++) {
      for(int xj=xi+1; xj<nbNodes; xj++) {
        if( !adjMat(xi,xj) ) continue;
        b.belEdges[adjMat(xi,xj)-1].create(nbStates[xj],nbStates[xi]);
      }
    }
  }
}

void InferenceEngineJT::updateBeliefs(
                                      std::vector<Clique*> vecCliques, std::vector<Separator*> vecSeparators,
                                      Beliefs &bel, Model *m, DataSequence *X, iMatrix adjMat, iVector nbStates,
                                      iVector vecN2C, iMatrix matN2C, bool bMaxProduct)
{
  int V = m->getNumberOfViews();
  int T = X->length();
  int nbNodes = V*T;
  
  // Singleton beliefs
  for(int xi=0; xi<nbNodes; xi++) {
    bel.belStates[xi].set(vecCliques[vecN2C[xi]]->marginalize(xi,bMaxProduct));
    double logZ = bel.belStates[xi].logSumExp();
    bel.belStates[xi].add(-logZ);
    bel.belStates[xi].eltExp();
  }
  
  // Pairwise beliefs
  for(int xi=0; xi<adjMat.getHeight(); xi++) {
    for(int xj=xi+1; xj<adjMat.getWidth(); xj++) {
      if( !adjMat(xi,xj) ) continue;
      dVector b = vecCliques[matN2C(xi,xj)]->marginalize(xi,xj,bMaxProduct);
      int hi = nbStates[xi];
      int hj = nbStates[xj];
      for(int j=0; j<hj; j++) for(int i=0; i<hi; i++)
        bel.belEdges[adjMat(xi,xj)-1](i,j) = b[j*hi+i]; // BEWARE THE ORDER OF INDICES!
      double logZ = bel.belEdges[adjMat(xi,xj)-1].logSumExp();
      bel.belEdges[adjMat(xi,xj)-1].add(-logZ);
      bel.belEdges[adjMat(xi,xj)-1].eltExp();
    }	}
  
  // Compute partition
  std::vector<Clique*>::iterator itc;
  std::vector<Separator*>::iterator its;
  for(itc=vecCliques.begin(); itc!=vecCliques.end(); itc++)
    bel.partition += (*itc)->potentials.logSumExp();
  for(its=vecSeparators.begin(); its!=vecSeparators.end(); its++)
    bel.partition -= (*its)->potentials.logSumExp();
  
}

int InferenceEngineJT::getNextElimination(iMatrix adjMat)
{
  int min_node_idx = 0;
  int min_neighbor_cnt = 100000;
  int neighbor_cnt = 0;
  
  for(int r=0; r<adjMat.getHeight(); r++) {
    neighbor_cnt = 0;
    for(int c=0; c<adjMat.getWidth(); c++)
      if( adjMat(r,c) ) neighbor_cnt++; // Note: entries are not always 1's
    if( neighbor_cnt==0 ) continue;
    if( neighbor_cnt < min_neighbor_cnt ) {
      min_node_idx = r;
      min_neighbor_cnt = neighbor_cnt ;
    }
  }
  return min_node_idx;
}

std::list<int> InferenceEngineJT::getIntersect(Clique* a, Clique* b)
{
  std::list<int> intersect;
  std::list<int>::iterator ni, nj;
  for(ni=a->vars.begin(); ni!=a->vars.end(); ni++) {
    for(nj=b->vars.begin(); nj!=b->vars.end(); nj++) {
      if( (*ni)==(*nj) ) {
        intersect.push_back((*ni));
      }	}	}
  return intersect;
}

void InferenceEngineJT::printJunctionTree(
                                          std::vector<Clique*> vecCliques, std::vector<Separator*> vecSeparators)
{
  std::vector<Separator*>::iterator its;
  for(its=vecSeparators.begin(); its!=vecSeparators.end(); its++) {
    printf("A"); (*its)->clique_A->print_vars();
    printf(" - S"); (*its)->print_vars();
    printf(" - B"); (*its)->clique_B->print_vars();
    printf("\n");
  }
  printf("vecCliques.size() = %d, vecSeparators.size() = %d\n",
         (int)vecCliques.size(), (int)vecSeparators.size());
}

void InferenceEngineJT::checkConsistency(
                                         std::vector<Clique*> vecCliques, std::vector<Separator*> vecSeparators, bool bMax)
{
  double logZ;
  Clique *A, *B;
  Separator *S;
  std::vector<Clique*>::iterator it;
  std::map<Clique*,Separator*>::iterator itm;
  for(it=vecCliques.begin(); it!=vecCliques.end(); it++) {
    A = (*it);
    for(itm=A->neighbors.begin(); itm!=A->neighbors.end(); itm++) {
      B = (*itm).first;
      S = (*itm).second;
      dVector src = A->marginalize(S->vars,bMax);
      dVector dst = B->marginalize(S->vars,bMax);
      logZ = src.logSumExp(); src.add(-logZ); src.eltExp();
      logZ = dst.logSumExp(); dst.add(-logZ); dst.eltExp();
      printf("dif: [ "); for(int i=0; i<dst.getLength(); i++) printf("%.4f ", src[i]-dst[i]); printf("]\n");
    }	}
}






///////////////////////////////////////////////////////////////////////////
// Implementation of the class JTNode, Clique, Separator
//
JTNode::JTNode():num_states(0),enum_states(0){};

JTNode::~JTNode()
{
  if( num_states ) {
    delete[] num_states;
    num_states = 0;
  }
  if( enum_states ) {
    for(int i=0; i<total_num_states; i++) {
      delete[] enum_states[i];
      enum_states[i] = 0;
    }
    delete[] enum_states;
    enum_states = 0;
  }
}

void JTNode::initialize(double val)
{
  total_num_states = 1;
  num_states = new int[(int)vars.size()];
  
  int i=0;
  std::list<int>::iterator it;
  for(it=cardinalities.begin(); it!=cardinalities.end(); it++, i++) {
    num_states[i] = (*it);
    total_num_states *= (*it);
  }
  potentials.resize(1,total_num_states,val); // col vector
  ratio.resize(1,total_num_states,val); // this is only for Separator
  
  // for easy marginalization
  enum_states = new int*[total_num_states];
  for(int i=0; i<total_num_states; i++)
    enum_states[i] = new int[(int)vars.size()];
  
  for(int j=0; j<(int)vars.size(); j++) {
    int z = 1;
    for(int k=0; k<j; k++)
      z *= num_states[k];
    for(int i=0; i<total_num_states; i++)
      enum_states[i][j] = (i/z)%num_states[j];
  }
}

void JTNode::assign_potential(int var, int state, double val)
{
  int idx, i; idx = -1;
  std::list<int>::iterator it;
  for(i=0, it=vars.begin(); it!=vars.end(); i++, it++)
    if( (*it)==var ) idx = i;
  
  for(int i=0; i<total_num_states; i++)
    if( enum_states[i][idx]==state )
      potentials[i] += val;
}

void JTNode::assign_potential(int var_a, int var_b, int state_a, int state_b, double val)
{
  int idx_a, idx_b, i;
  idx_a = idx_b = -1;
  std::list<int>::iterator it;
  for(i=0, it=vars.begin(); it!=vars.end(); i++, it++) {
    if( (*it)==var_a ) idx_a = i;
    if( (*it)==var_b ) idx_b = i;
  }
  
  for(int i=0; i<total_num_states; i++)
    if( enum_states[i][idx_a]==state_a && enum_states[i][idx_b]==state_b )
      potentials[i] += val;
}



dVector JTNode::marginalize(int var, bool bMax)
{
  std::list<int> v; v.push_back(var);
  return marginalize(v,bMax);
}

dVector JTNode::marginalize(int var_a, int var_b, bool bMax)
{
  std::list<int> v; v.push_back(var_a); v.push_back(var_b);
  return marginalize(v,bMax);
}

dVector JTNode::marginalize(std::list<int> sum_to, bool bMax)
{
  if( vars==sum_to )
    return potentials;
  
  int i=0;
  int var_size = (int)vars.size();
  int* mask = new int[var_size]; // if( mask[i] ) sum-up the i-th var
  std::list<int>::iterator ita, itb;
  for(int i=0; i<var_size; i++)
    mask[i] = 1;
  for(ita=vars.begin(); ita!=vars.end(); ita++, i++)
    for(itb=sum_to.begin(); itb!=sum_to.end(); itb++)
      if( (*ita)==(*itb) ) mask[i] = 0;
  
  // create a dVector with the size of given vars in {v}
  int sz_inner, sz_outer;
  sz_inner = sz_outer = 1;
  for(int i=0; i<var_size; i++) {
    if( mask[i] ) sz_inner *= num_states[i];
    else sz_outer *= num_states[i];
  }
  std::vector<dVector> tmp_potential(sz_outer);
  for(int i=0; i<sz_outer; i++)
    tmp_potential[i].resize(1,sz_inner);
  dVector new_potential(sz_outer);
  
  // marginalize
  for(int i=0; i<total_num_states; i++) {
    int idx_outer, idx_inner, offset_outer, offset_inner;
    idx_outer = idx_inner = 0;
    for(int j=0; j<var_size; j++) {
      offset_outer = offset_inner = 1;
      if( mask[j] ) { // this is the next variable we want to store to
        for(int k=0; k<j; k++)
          if( mask[k] ) offset_inner *= num_states[k];
        idx_inner += enum_states[i][j] * offset_inner;
      }
      else { // this is the next variable we want to sum over
        for(int k=0; k<j; k++)
          if( !mask[k] ) offset_outer *= num_states[k];
        idx_outer += enum_states[i][j] * offset_outer;
      }
    }
    tmp_potential[idx_outer][idx_inner] = potentials[i];
  }
  int max_idx; double max_val;
  for(int i=0; i<sz_outer; i++) {
    if( bMax ) {
      max_idx=0; max_val=tmp_potential[i][0];
      for(int j=1; j<tmp_potential[i].getLength(); j++) {
        if( tmp_potential[i][j]>max_val ) {
          max_idx = j; max_val = tmp_potential[i][j];
        }
      }
      new_potential[i] = tmp_potential[i][max_idx];
    }
    else
      new_potential[i] = tmp_potential[i].logSumExp();
  }
  
  delete[] mask; mask = 0;
  return new_potential;
}



void JTNode::scale(std::list<int> v, dVector ratio)
{
  // if( mask[i] ) sum-up the i-th var
  int i=0;
  std::list<int>::iterator ita, itb;
  int* mask = new int[(int)vars.size()];
  for(unsigned int i=0; i<vars.size(); i++) mask[i] = 1;
  for(ita=vars.begin(); ita!=vars.end(); ita++, i++)
    for(itb=v.begin(); itb!=v.end(); itb++)
      if( (*ita)==(*itb) ) mask[i] = 0;
  
  for(int i=0; i<total_num_states; i++) {
    int src_idx = 0;
    int offset = 1;
    for(int j=0; j<(int)vars.size(); j++) {
      offset = 1;
      if( !mask[j] ) {// this is the next variable we want to sum over
        for(int k=0; k<j; k++)
          if( !mask[k] ) offset *= num_states[k];
        src_idx += enum_states[i][j] * offset;
      }
    }
    potentials[i] += ratio[src_idx];
  }
  
  delete[] mask; mask = 0;
}


void JTNode::print_vars()
{
  std::list<int>::iterator ita;
  printf("[ "); 
  for(ita=vars.begin(); ita!=vars.end(); ita++)
    printf("%d ", (*ita)); 
  printf("]");
}
void JTNode::print_potentials()
{ 
  printf("[ ");
  for(int i=0; i<total_num_states; i++)
    printf("%.2f ", potentials[i]);
  printf("]");
}
bool JTNode::equals(JTNode* node)
{
  if( vars.size() != node->vars.size() )
    return false;
  if( total_num_states != node->total_num_states )
    return false;
  
  std::list<int>::iterator it_a, it_b;
  for(it_a=vars.begin(); it_a!=vars.end(); it_a++) {
    bool found = false;
    for(it_b=node->vars.begin(); it_b!=node->vars.end(); it_b++) {
      if( (*it_a)==(*it_b) ) {
        found = true;
        continue;
      }
    }
    if(!found) return false;
  }
  return true;
}
bool JTNode::contains(int var)
{ 
  std::list<int>::iterator it;
  for(it=vars.begin(); it!=vars.end(); it++)
    if( (*it)==var ) return true;
  return false;
}
bool JTNode::contains(int var_a, int var_b)
{
  bool found_a, found_b;
  found_a = found_b = false;
  std::list<int>::iterator it;
  for(it=vars.begin(); it!=vars.end(); it++) {
    if( (*it)==var_a ) found_a = true;
    if( (*it)==var_b ) found_b = true;
  }
  return found_a && found_b;
}

void JTNode::sort()
{
  int i,j;
  int* order = new int[(int)vars.size()];	
  std::list<int>::iterator ita, itb; 
  std::list<int> vars_original = vars;
  vars.sort(); 
  
  for(i=0, ita=vars.begin(); ita!=vars.end(); i++, ita++)
    for(j=0, itb=vars_original.begin(); itb!=vars_original.end(); j++, itb++)
      if( (*ita)==(*itb) ) order[i] = j;
  
  std::list<int> cardinality_sorted;
  for(i=0; i<(int)vars.size(); i++) {
    ita = cardinalities.begin();
    std::advance(ita,order[i]);
    cardinality_sorted.push_back(*ita);
  }
  cardinalities = cardinality_sorted;
  
  delete[] order; order = 0;
}

void Clique::add_neighbor(Separator *S, Clique *C)
{
  neighbors.insert(std::pair<Clique*,Separator*>(C,S));
}

void Separator::update_ratio(dVector new_potentials)
{
  ratio.set(potentials);
  ratio.negate();
  ratio.add(new_potentials);
  potentials.set(new_potentials);
}








