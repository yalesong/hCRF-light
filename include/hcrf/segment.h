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

#ifndef SEGMENT_H
#define SEGMENT_H

#include <algorithm>
#include <cmath>
#include <vector>

#include "hcrf/matrix.h"
#include "hcrf/inferenceengine.h"

template <class T> inline T square(const T &x) { return x*x; };

typedef struct edge {
  int a, b;
  double w;
  edge(int i, int j, double weight): a(i), b(j), w(weight){}
} edge;

typedef struct {
  int p;
  int size;
  std::list<int> group;
} uni_seg;

class universe {
public:
  universe(int elements);
  ~universe();
  int find(int x);
  void join(int x, int y);
  int size(int x) const {return elts[x].size; }
  int num_sets() const {return num;}
  void get_segments(int elt_sz, int* elt_labels, int* seg_sizes, int** seg_elts);
  void get_segments(int elt_sz, std::list<int> &elt_labels,
                    std::list<int> &seg_sizes, std::vector<std::list<int> > &seg_elts);
private:
  std::vector<uni_seg> elts;
  int num;
};

class segment {
public:
  // Segment based on the observation
  void segment_sequence(const dMatrix src_M, dMatrix& dst_M, double c, int *num_ccs);
  
  // Segment based on the energy p(y,h|x)
  void segment_sequence(std::vector<Beliefs> energy, double c, int *num_ccs, int**elt_labels=0);
  void segment_sequence(const dMatrix src_M, dMatrix& dst_M,
                        std::vector<Beliefs> energy, double c, int *num_ccs, int**elt_labels=0);
  void segment_sequence(const dMatrix src_M, dMatrix& dst_M,
                        Beliefs energy, double c, int *num_ccs, int **seg_sizes=0);
  
  void segment_graph(universe &u, int num_vertices,
                     int num_edges, std::vector<edge> edges, double c);
  
private:
  void getLogE(std::vector<Beliefs> E, std::vector<Beliefs> &logE);
  
  // Distant functions for observation values
  double diff_unary(const dMatrix M, int c1, int c2);
  
  // Distant functions for energy values
  double diff_unary(Beliefs E, int c1, int c2, int dimH);
  double diff_unary(std::vector<Beliefs> E, int c1, int c2, int dimH, int dimY);
  double diff_min_unary(std::vector<Beliefs> E, int c1, int c2, int dimH, int dimY);
  double diff_mean_compat(std::vector<Beliefs> E, int c1, int c2, int dimH, int dimY);
  double mutual_info(std::vector<Beliefs> E, std::vector<Beliefs> logE, int c1, int c2);
  double multivar_mutual_info(std::vector<Beliefs> E, std::vector<Beliefs> logE, int nbVars,
                              int* indices, int dimH, int dimY);
};


#endif
