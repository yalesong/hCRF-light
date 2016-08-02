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

#ifndef INFERENCEENGINE_H
#define INFERENCEENGINE_H

//Standard Template Library includes
#include <vector>
#include <list>

#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <float.h>
#include <map>
#include <algorithm>

//hCRF Library includes
#include "hcrf/hcrfExcep.h"
#include "hcrf/featuregenerator.h"
#include "hcrf/matrix.h"

//#define INF_VALUE -DBL_MAX
#define INF_VALUE 1e100

#if defined(__VISUALC__)||defined(__BORLAND__)
#define wxFinite(n) _finite(n)
#elseif defined(__GNUC__)
#define wxFinite(n) finite(n)
#else
#define wxFinite(n) ((n) == (n))
#endif

struct Beliefs {
public:
  std::vector<dVector> belStates;
  std::vector<dMatrix> belEdges;
  double partition;
  Beliefs():belStates(), belEdges(), partition(0.0) {};
};

class InferenceEngine
{
public:
  //Constructor/Destructor
  InferenceEngine();
  virtual ~InferenceEngine();
  
  virtual void computeBeliefs(Beliefs& bel, FeatureGenerator* fGen,
                              DataSequence* X, Model* m,
                              int bComputePartition,int seqLabel=-1,
                              bool bUseStatePerNodes = false)=0;
  virtual double computePartition(FeatureGenerator* fGen,DataSequence* X,
                                  Model* m,int seqLabel=-1,
                                  bool bUseStatePerNodes = false)=0;
  virtual void setMaxNumberThreads(int maxThreads);
  
protected:
  // Private function that are used as utility function for several
  // beliefs propagations algorithms.
  void computeLogMi(FeatureGenerator* fGen, Model* model, DataSequence* X,
                    int i, int seqLabel, dMatrix& Mi_YY, dVector& Ri_Y,
                    bool takeExp, bool bUseStatePerNodes) ;
  void LogMultiply(dMatrix& Potentials,dVector& Beli, dVector& LogAB);
  
  void logMultiply(dVector src_Vi, dMatrix src_Mij, dVector& dst_Vj);
  void logMultiply(dVector src_Vi, dVector src_Vj, dMatrix& dst_Mij);
  int logMultiplyMaxProd(dVector src_Vi, dMatrix src_Mij, dVector& dst_Vj);
  
  int nbThreadsMP;
  
};

class InferenceEngineFB:public InferenceEngine
{
public:
  void computeBeliefs(Beliefs& bel, FeatureGenerator* fGen,DataSequence* X,
                      Model* crf, int bComputePartition,int seqLabel=-1,
                      bool bUseStatePerNodes = false);
  double computePartition(FeatureGenerator* fGen, DataSequence* X,
                          Model* crf, int seqLabel=-1,
                          bool bUseStatePerNodes = false);
private:
  void computeBeliefsLog(Beliefs& bel, FeatureGenerator* fGen,
                         DataSequence* X, Model* model,
                         int bComputePartition,int seqLabel,
                         bool bUseStatePerNodes);
  void computeBeliefsLinear(Beliefs& bel, FeatureGenerator* fGen,
                            DataSequence* X, Model* model,
                            int bComputePartition,int seqLabel,
                            bool bUseStatePerNodes);
};


class BPNode
{
public:
  BPNode(int node_id, int node_size);
  ~BPNode();
  void addNeighbor(BPNode* v);
  bool equal(BPNode* v);
  void print();
  
  int id;
  int size;
  std::list<BPNode*> neighbors;
};

class InferenceEngineBP: public InferenceEngine
{
public:
  InferenceEngineBP(bool isSumProduct = true);
  ~InferenceEngineBP();
  
  void computeBeliefs(Beliefs& beliefs, FeatureGenerator* fGen, DataSequence* X,
                      Model* model, int bComputePartition, int seqLabel=-1,
                      bool bUseStatePerNodes=false);
  
  double computePartition(FeatureGenerator* fGen, DataSequence* X, Model* model,
                          int seqLabel=-1, bool bUseStatePerNodes=false);
  
private:
  bool isSumProduct;
  
  void collect(BPNode* xi, BPNode* xj, dVector* phi_i, dMatrix** phi_ij, dVector** msg);
  void distribute(BPNode* xi, BPNode* xj, dVector* phi_i, dMatrix** phi_ij, dVector** msg);
  void sendMessage(BPNode* xi, BPNode* xj, dVector* phi_i, dMatrix** phi_ij, dVector** msg);
  void initMessages(dVector**& msg, DataSequence* X, Model* model, iMatrix adjMat, iVector nbStates);
  void initBeliefs(Beliefs& beliefs, DataSequence* X, Model* model, iMatrix adjMat, iVector nbStates);
  void initPotentials(dVector*& phi_i, dMatrix**& phi_ij,
                      FeatureGenerator* fGen, DataSequence* X, Model* model,
                      iMatrix adjMat, iVector nbStates, int seqLabel);
  void updateBeliefs(Beliefs& beliefs,
                     dVector* phi_i, dMatrix** phi_ij, dVector** msg,
                     DataSequence* X, Model* model, iMatrix adjMat);
  void printBeliefs(Beliefs beliefs);
};


class InferenceEngineLoopyBP: public InferenceEngine
{
public:
  InferenceEngineLoopyBP(int max_iter=15, double error_threshold=0.0001);
  ~InferenceEngineLoopyBP();
  
  void computeBeliefs(Beliefs& beliefs, FeatureGenerator* fGen, DataSequence* X,
                      Model* model, int bComputePartition, int seqLabel=-1,
                      bool bUseStatePerNodes=false);
  
  double computePartition(FeatureGenerator* fGen, DataSequence* X, Model* model,
                          int seqLabel=-1, bool bUseStatePerNodes=false);
  
private:
  // Convergence criterion
  int m_max_iter;
  double m_min_threshold;
  
  void initializeBeliefs(Beliefs& beliefs, DataSequence* X, Model* m, iMatrix adjMat);
  void initializeMessages(std::vector<dVector>& messages, DataSequence* X, Model* m, iMatrix adjMat, int adjMatMax, bool randomInit = false);
  void initializePotentials(Beliefs& potentials, FeatureGenerator* fGen, DataSequence *X,
                            Model* m, iMatrix adjMat, int seqLabel, bool bUseStatePerNodes=false);
  
  void normalizeMessage(int xi, std::vector<dVector>& messages, iMatrix adjMat, int adjMatMax);
  
  // m_ij(xj) = sum_xi {potential(i)*potential(i,j)*prod_{u \in N(i)\j} {m_ui(xi)}}
  // In case of maxProduct, returns an index of max_xi (for Viterbi decoding)
  void sendMessage(int xi, int xj, int numOfNodes,
                   const Beliefs potentials,
                   std::vector<dVector>& messages,
                   iMatrix adjMat,
                   int adjMatMax,
                   bool bMaxProd);
  
  // b_i(xi) = potential(i) * prod_{u \in N(i)}{m_ui}
  // b_ij(xi,xj) = potential(i) * potential(j) * potential(i,j)
  //			     * prod_{u \in N(i)\j}{m_ui} * prod_{u \in N(j)\i}{m_uj}
  void updateBeliefs(int numOfNodes,
                     Beliefs& beliefs,
                     const Beliefs potentials,
                     const std::vector<dVector> messages,
                     iMatrix adjMat,
                     int adjMatMax );
  
  //
  // Helper functions
  void getSequentialUpdateOrder(int* order, DataSequence* X, Model* m);
  void getRandomUpdateOrder(int* order, DataSequence* X, Model* m);
};


class JTNode
{
public:
  JTNode();
  ~JTNode();
  //
  void initialize(double initial_value=0.0);
  
  //
  dVector marginalize(int var, bool bMax);
  dVector marginalize(int var_a, int var_b, bool bMax);
  dVector marginalize(std::list<int> sum_to, bool bMax);
  
  //
  // Assign a singleton potential
  void assign_potential(int var, int state, double val);
  //
  // Assign a pairwise potential
  void assign_potential(int var_a, int var_b, int state_a, int state_b, double val);
  //
  void scale(std::list<int> vars, dVector ratio);
  
  // HELPER FUNCTIONS
  //
  void print_vars();
  void print_potentials();
  
  //
  bool equals(JTNode* node);
  bool contains(int var);
  bool contains(int var_a, int var_b);
  
  void sort();
  
  std::list<int> vars;
  std::list<int> cardinalities;
  
  int total_num_states;
  int* num_states;
  int** enum_states;
  dVector potentials;
  dVector ratio; // for Separator node
};

class Clique;
class Separator;

class Clique: public JTNode
{
public:
  Clique():JTNode() {};
  void add_neighbor(Separator* S, Clique* C);
  std::map<Clique*, Separator*> neighbors;
};

class Separator: public JTNode
{
public:
  Separator(Clique* node_A, Clique* node_B):
  JTNode(), clique_A(node_A), clique_B(node_B) {};
  void update_ratio(dVector new_potentials);
  Clique *clique_A, *clique_B;
};

class InferenceEngineJT: public InferenceEngine
{
public:
  InferenceEngineJT();
  ~InferenceEngineJT();
  
  void computeBeliefs(Beliefs& beliefs, FeatureGenerator* fGen, DataSequence* X,
                      Model* model, int bComputePartition, int seqLabel=-1,
                      bool bUseStatePerNodes=false);
  
  double computePartition(FeatureGenerator* fGen, DataSequence* X, Model* model,
                          int seqLabel=-1, bool bUseStatePerNodes=false);
  
private:
  void constructTriangulatedGraph(
                                  std::vector<Clique*> &vecCliques,
                                  Model* model,
                                  DataSequence* X,
                                  iMatrix adjMat,
                                  iVector nbStates);
  
  void buildJunctionTree(
                         std::vector<Clique*> &vecCliques,
                         std::vector<Separator*> &vecSeparators,
                         Model* model,
                         DataSequence* X,
                         iMatrix& JTadjMat,
                         iVector nbStates);
  
  void removeRedundantNodes(
                            std::vector<Clique*> &vecCliques,
                            std::vector<Separator*> &vecSeparators,
                            Model* model,
                            DataSequence* X,
                            iMatrix& JTadjMat);
  
  // Helper functions for removeRedundantNodes()
  int findCliqueOffset(std::vector<Clique*> vecCliques, Clique* clique);
  int findSeparatorOffset(std::vector<Separator*> vecSeparator, Separator* separator);
  
  void findNodeToCliqueIndexMap(
                                Model* model,
                                DataSequence* X,
                                std::vector<Clique*> vecCliques,
                                iVector& vecNodeToClique,
                                iMatrix& matNodeToClique,
                                iMatrix adjMat);
  
  void initializePontentials(
                             std::vector<Clique*> &vecCliques,
                             std::vector<Separator*> &vecSeparators,
                             FeatureGenerator* fGen,
                             Model* model,
                             DataSequence* X,
                             int seqLabel,
                             iMatrix adjMat,
                             iVector vecNodeToClique,
                             iMatrix matNodeToClique,
                             bool bUseStatePerNodes);
  
  void initializeBeliefs(
                         Beliefs& beliefs,
                         DataSequence* X,
                         Model* m,
                         iMatrix adjMat,
                         iVector nbStates);
  
  void updateBeliefs(
                     std::vector<Clique*> vecCliques,
                     std::vector<Separator*> vecSeparators,
                     Beliefs& beliefs,
                     Model* model,
                     DataSequence* X,
                     iMatrix adjMat,
                     iVector nbStates,
                     iVector vecNodeToClique,
                     iMatrix matNodeToClique,
                     bool bMaxProduct);
  
  void collectEvidence(
                       std::vector<Clique*> vecCliques,
                       int node_idx,
                       iMatrix& untouched_nodes,
                       bool bMaxProduct);
  
  void distributeEvidence(
                          std::vector<Clique*> vecCliques,
                          int node_idx,
                          iMatrix& untouched_nodes,
                          bool bMaxProduct);
  
  void update(
              Clique* src_clique,
              Clique* dst_clique,
              bool bMaxProduct);
  
  void printJunctionTree(
                         std::vector<Clique*> vecCliques,
                         std::vector<Separator*> vecSeparators);
  
  void checkConsistency(
                        std::vector<Clique*> vecCliques,
                        std::vector<Separator*> vecSeparators,
                        bool bMaxProduct);
  
  int getNextElimination(iMatrix adjMat);
  std::list<int> getIntersect(Clique* a, Clique* b);
  
  //
  // Predicate for sorting JTCliques using std::sort(...)
  static bool CliqueSortPredicate(Clique* a, Clique* b) {
    return (a->vars.front() < b->vars.front());
  };
};

#endif //INFERENCEENGINE_H
