/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA

- 03/08/2012 Modified by Yale Song (yalesong@csail.mit.edu) 
*/  

#include "segment.h"

// threshold function
#define THRESHOLD(size, c) (c/size) 

bool operator<(const edge &a, const edge &b) {return a.w < b.w;}
bool operator>(const edge &a, const edge &b) {return a.w > b.w;}
double mymax(const double a, const double b) { if (a>b) return a; else return b; }

////////////////////////////////////////////////////////////////
// (Dis)similarity measure between observations  
 
double segment::diff_unary(const dMatrix M, int c1, int c2)
{
	double val = 0;
	for( int r=0; r<M.getHeight(); r++ ) 
		val += fabs( M(r,c1) - M(r,c2) );
	return val;
}

double segment::diff_unary(Beliefs E, int c1, int c2, int dimH)
{
	int i;
	double val = 0;
	for( i=0; i<dimH; i++ )
		val += fabs(E.belStates[c1][i] - E.belStates[c2][i]);
	return val;
}

double segment::diff_unary(std::vector<Beliefs> E, int c1, int c2, int dimH, int dimY)
{
	int i, y;
	dVector val_y(dimY);
	for( y=0; y<dimY; y++ ) {
		for( i=0; i<dimH; i++ )
			val_y[y] += fabs(E[y].belStates[c1][i] - E[y].belStates[c2][i]);
	}
	return val_y.sum();
}

double segment::diff_min_unary(std::vector<Beliefs> E, int c1, int c2, int dimH, int dimY)
{
	int i, y, max_idx1, max_idx2;
	double max_val1, max_val2;
	dVector val_y(dimY);
	for( y=0; y<dimY; y++ ) {
		max_idx1 = max_idx2 = -1; 
		max_val1 = max_val2 = -DBL_MAX;
		for( i=0; i<dimH; i++ ) {
			if( E[y].belStates[c1][i] > max_val1 ) {
				max_idx1 = i; 
				max_val1 = E[y].belStates[c1][i];
			}
			if( E[y].belStates[c2][i] > max_val2 ) {
				max_idx2 = i;
				max_val2 = E[y].belStates[c2][i];
			}
		}
		val_y[y] += mymax(E[y].belStates[c1][max_idx2]-max_val1, E[y].belStates[c2][max_idx1]-max_val2);
	}
	return val_y.sum();
}
double segment::diff_mean_compat(std::vector<Beliefs> E, int c1, int c2, int dimH, int dimY)
{
	int i, j, y;
	double agree_val, disagree_val;
	dVector val_y(dimY);
	for( y=0; y<dimY; y++ ) {
		agree_val = disagree_val = 0;
		for( i=0; i<dimH; i++ ) for( j=0; j<dimH; j++ )
			if( i==j )
				agree_val += E[y].belEdges[c1](i,j);
			else
				disagree_val += E[y].belEdges[c1](i,j);
		val_y[y] = agree_val/dimH - disagree_val/(dimH*(dimH-1));
	}
	return val_y.sum();
}

double segment::mutual_info(std::vector<Beliefs> E, std::vector<Beliefs> logE, int c1, int c2)
{
	// \sum_{xi} \sum_{xj} {p(xi,xj) * (log p(xi,xj) - log p(xi) - log p(xj)}
	int i, j, y, dimH, dimY;
	double partition_sum, val;

	partition_sum = val = 0;
	dimH = E[0].belStates[0].getLength();
	dimY = (int)E.size();	
	dVector val_y(dimY);

	for( y=0; y<dimY; y++ ) for( i=0; i<dimH; i++ ) for( j=0; j<dimH; j++ ) 
		val_y[y] += E[y].belEdges[c1](i,j) * (logE[y].belEdges[c1](i,j) 
			- logE[y].belStates[c1][i] - logE[y].belStates[c2][j]);
		
	// Multiply by p(y|x)
	for( y=0; y<dimY; y++ )
		partition_sum += E[y].partition;
	for( y=0; y<dimY; y++ ) 
		val += 100*(E[y].partition / partition_sum ) * val_y[y];
/*	
	printf("c1=%d, c2=%d\n", c1, c2);
	for( y=0; y<dimY; y++ ) {
		printf("y=%d \n", y);
		for( i=0; i<dimH; i++ ) { for( j=0; j<dimH; j++ ) printf("%.4f ", E[y].belEdges[c1](i,j)); printf("\n"); }
		printf("[%d]: ", c1); for( i=0; i<dimH; i++ )  printf("%.4f ", E[y].belStates[c1][i]); printf("\n");
		printf("[%d]: ", c2); for( i=0; i<dimH; i++ )  printf("%.4f ", E[y].belStates[c2][i]); printf("\n");
		printf(" => val = %f * %f\n", 100*E[y].partition/partition_sum ), val_y[y]);
		printf("\n-------------------\n");
		if( 100*(logE[y].partition/partition_sum )>30 ) getchar();
	}
	printf("mutual info = %f\n", val); 
*/
	//return ( val>0.0 ) ? val : 0;
	return (val!=0) ? 1/val : 0;
}  

double segment::multivar_mutual_info(
	std::vector<Beliefs> E, std::vector<Beliefs> logE, int nbVars, int* indices, int dimH, int dimY)
{
	int i, j, y, nbEnumStates;
	int *base, **enumStates;
	double partition_sum, val, joint, log_joint, log_marginal;
	dVector val_y(dimY);

	base = new int[nbVars];  base[0] = 1;
	for( i=0; i<nbVars; i++ )
		base[i] = (int)pow((double)dimH,i);

	nbEnumStates = (int)pow((double)dimH,nbVars);
	enumStates = new int*[nbEnumStates];
	for( i=0; i<nbEnumStates; i++ ) {
		enumStates[i] = new int[nbVars];
		enumStates[i][0] = i / base[nbVars-1];
		for( j=1; j<nbVars; j++ ) 
			enumStates[i][j] = (i%base[nbVars-j])/base[nbVars-j-1];		
	}
	
	// Add up mutual information
	partition_sum = val = 0;
	for( y=0; y<dimY; y++ ) {
		for( i=0; i<nbEnumStates; i++ ) {
			joint = 1; log_joint = 0; log_marginal = 0;
			for( j=0; j<nbVars-1; j++ ) {
				joint *= E[y].belEdges[indices[j]](enumStates[i][j],enumStates[i][j+1]);
				log_joint += logE[y].belEdges[indices[j]](enumStates[i][j],enumStates[i][j+1]);
			}
			for( j=0; j<nbVars; j++ )
				log_marginal += logE[y].belStates[indices[j]][enumStates[i][j]];
			val_y[y] += joint*(log_joint-log_marginal);			
		}
	}
	
	// Multiply by p(y|x)
	for( y=0; y<dimY; y++ )
		partition_sum += E[y].partition;
	for( y=0; y<dimY; y++ ) 
		val += 100*(E[y].partition / partition_sum ) * val_y[y];
/*
	printf("multivar_mutual_info [ ");for(i=0; i<nbVars; i++) printf("%d ", indices[i]); printf("]\n");
	for(y=0; y<dimY; y++) {
		for(i=0; i<nbVars; i++) {
			printf("y=%d, [%d]: ", y, indices[i]);
			for(j=0; j<dimH; j++) {
				printf("%.4f ", E[y].belStates[indices[i]][j]);
			}
			printf("\n");
		}
	}
	printf("multivariate mutual information = %f", val); getchar();
*/
	
	// Clean up and return
	delete[] base; base=0;
	for( i=0; i<nbEnumStates; i++ ) {
		delete[] enumStates[i]; enumStates[i]=0;
	}
	delete[] enumStates; enumStates=0;

	return val;
}

void segment::getLogE(std::vector<Beliefs> E, std::vector<Beliefs> &logE)
{
	int i, j, y, dimH, dimY;
	dimH = E[0].belStates[0].getLength();
	dimY = (int)E.size();

	std::vector<Beliefs>::iterator it;
	std::vector<dVector>::iterator it_v;
	std::vector<dMatrix>::iterator it_m;

	for( it=E.begin(); it!=E.end(); it++ ) 
	{
		Beliefs b = *it;
		for( it_v=b.belStates.begin(); it_v!=b.belStates.end(); it_v++ )
			for( i=0; i<dimH; i++ )
				(*it_v)[i] = log((*it_v)[i]);
		for( it_m=b.belEdges.begin(); it_m!=b.belEdges.end(); it_m++ )
			for( i=0; i<dimH; i++ ) for( j=0; j<dimH; j++ )
				(*it_m)(i,j) = log((*it_m)(i,j));
		logE.push_back(b);		
	}

	for( y=0; y<dimY; y++ )
		logE[y].partition = log(E[y].partition);	

}


/* Segment a sequence
 *
 * Returns a sequence representing the segmentation
 *
 * M: sequence to segment
 * c: constant for threshold function
 * min_size: minimum component size (enforced by post-processing stage)
 * num_ccs: number of connected components in the segmentation
 */
void segment::segment_sequence(const dMatrix src_M, dMatrix& dst_M, double c, int *num_ccs)
{
	int i, j, t, seqLength;
	seqLength = src_M.getWidth(); 
	
	// Build a graph
	std::vector<edge> edges;
	for( t=0; t<seqLength-1; t++ ) {
		edge e(t,t+1,diff_unary(src_M,t,t+1));
		edges.push_back(e);
	} 

	// segment
	universe u(seqLength); // make a disjoint-set forest 
	segment_graph(u, seqLength, seqLength-1, edges, c);
	*num_ccs = u.num_sets(); 
	
	int *elt_labels, *seg_sizes, **seg_elts;
	elt_labels	= new int[seqLength]; 
	seg_sizes	= new int[u.num_sets()];
	seg_elts	= new int*[u.num_sets()];
	u.get_segments(seqLength, elt_labels, seg_sizes, seg_elts);
	
//	for(i=0; i<u.num_sets(); i++) printf("[%d]", seg_sizes[i]); printf("\n");  
 
	// Generate observations for the new layer
	dst_M.resize(u.num_sets(), src_M.getHeight());
	for( i=0; i<src_M.getHeight(); i++ ) for( j=0; j<src_M.getWidth(); j++ ) 
		dst_M.addValue(i,elt_labels[j],src_M.getValue(i,j));
	for( i=0; i<dst_M.getHeight(); i++ ) for( j=0; j<dst_M.getWidth(); j++ ) 
		dst_M(i,j) = dst_M(i,j) / seg_sizes[j];
	
	// Clean up and return
	delete[] elt_labels; elt_labels = 0; 
	delete[] seg_sizes; seg_sizes = 0;  
	for(i=0; i<u.num_sets(); i++) {
		delete[] seg_elts[i];
		seg_elts[i] = 0;
	}
	delete[] seg_elts; seg_elts = 0; 
}


/* Segment a sequence
 *
 * Returns a sequence representing the segmentation
 *
 * M: sequence to segment
 * c: constant for threshold function
 * min_size: minimum component size (enforced by post-processing stage)
 * num_ccs: number of connected components in the segmentation
 */
void segment::segment_sequence(std::vector<Beliefs> E, double c, int *num_ccs, int**labels)
{
	int i, t, seqLength, dimH, dimY;
	dimY = (int)E.size();
	dimH = (int)E[0].belStates[0].getLength();
	seqLength = (int)E[0].belStates.size();

	std::vector<Beliefs> logE; 
	getLogE(E, logE);

	// Build a graph
	std::vector<edge> edges;
	for( t=0; t<seqLength-1; t++ ) {
		edge e(t,t+1,diff_unary(E,t,t+1,dimH,dimY));
		edges.push_back(e);
	} 

	// segment
	universe u(seqLength); // make a disjoint-set forest 
	segment_graph(u, seqLength, seqLength-1, edges, c);
	*num_ccs = u.num_sets(); 
	
	int *elt_labels, *seg_sizes, **seg_elts;
	elt_labels	= new int[seqLength]; 
	seg_sizes	= new int[u.num_sets()];
	seg_elts	= new int*[u.num_sets()];
	u.get_segments(seqLength, elt_labels, seg_sizes, seg_elts);

	// Optionally, save elt_labels to labels
	if( labels!=0 ) 
		memcpy(*labels, elt_labels, sizeof(int)*seqLength);
	
	// Clean up and return
	delete[] elt_labels; elt_labels = 0; 
	delete[] seg_sizes; seg_sizes = 0;  
	for(i=0; i<u.num_sets(); i++) {
		delete[] seg_elts[i];
		seg_elts[i] = 0;
	}
	delete[] seg_elts; seg_elts = 0; 
}

void segment::segment_sequence(const dMatrix src_M, dMatrix& dst_M, 
	std::vector<Beliefs> E, double c, int *num_ccs, int**labels)
{
	int i, j, t, seqLength, dimH, dimY;
	seqLength = src_M.getWidth(); 
	dimH = E[0].belStates[0].getLength();
	dimY = (int)E.size();
	std::vector<Beliefs> logE; getLogE(E, logE);

	// Build a graph
	std::vector<edge> edges;
	for( t=0; t<seqLength-1; t++ ) {
		edge e(t,t+1,diff_unary(E,t,t+1,dimH,dimY));
		edges.push_back(e);
	} 

	// segment
	universe u(seqLength); // make a disjoint-set forest 
	segment_graph(u, seqLength, seqLength-1, edges, c);
	*num_ccs = u.num_sets(); 
	
	int *elt_labels, *seg_sizes, **seg_elts;
	elt_labels	= new int[seqLength]; 
	seg_sizes	= new int[u.num_sets()];
	seg_elts	= new int*[u.num_sets()];
	u.get_segments(seqLength, elt_labels, seg_sizes, seg_elts);

	// Optionally, save elt_labels to labels
	if( labels!=0 ) 
		memcpy(*labels, elt_labels, sizeof(int)*seqLength);
	
	// Generate observations for the new layer
	dst_M.resize(u.num_sets(), src_M.getHeight());
	for( i=0; i<src_M.getHeight(); i++ ) for( j=0; j<src_M.getWidth(); j++ ) 
		dst_M.addValue(i,elt_labels[j],src_M.getValue(i,j));
	for( i=0; i<dst_M.getHeight(); i++ ) for( j=0; j<dst_M.getWidth(); j++ ) 
		dst_M(i,j) = dst_M(i,j) / seg_sizes[j];

	// Clean up and return
	delete[] elt_labels; elt_labels = 0; 
	delete[] seg_sizes; seg_sizes = 0;  
	for(i=0; i<u.num_sets(); i++) {
		delete[] seg_elts[i];
		seg_elts[i] = 0;
	}
	delete[] seg_elts; seg_elts = 0; 
}


void segment::segment_sequence(const dMatrix src_M, dMatrix& dst_M, 
	Beliefs E, double c, int *num_ccs, int **sizes)
{
	int i, j, t, seqLength, dimH;
	seqLength = src_M.getWidth(); 
	dimH = E.belStates[0].getLength();

	// Build a graph
	std::vector<edge> edges;
	for( t=0; t<seqLength-1; t++ ) {
		edge e(t,t+1,diff_unary(E,t,t+1,dimH));
		edges.push_back(e);
	} 

	// segment
	universe u(seqLength); // make a disjoint-set forest 
	segment_graph(u, seqLength, seqLength-1, edges, c);
	*num_ccs = u.num_sets(); 
	
	int *elt_labels, *seg_sizes, **seg_elts;
	elt_labels	= new int[seqLength]; 
	seg_sizes	= new int[u.num_sets()];
	seg_elts	= new int*[u.num_sets()];
	u.get_segments(seqLength, elt_labels, seg_sizes, seg_elts);

	for(i=0; i<u.num_sets(); i++) printf("[%d]", seg_sizes[i]); printf("\n");   //getchar();
	
	if( sizes ) {
		delete[] *sizes; *sizes = new int[u.num_sets()];
		memcpy(*sizes,seg_sizes,u.num_sets()*sizeof(int));
	}

	// Generate observations for the new layer
	dst_M.resize(u.num_sets(), src_M.getHeight());
	for( i=0; i<src_M.getHeight(); i++ ) for( j=0; j<src_M.getWidth(); j++ ) 
		dst_M.addValue(i,elt_labels[j],src_M.getValue(i,j));
	for( i=0; i<dst_M.getHeight(); i++ ) for( j=0; j<dst_M.getWidth(); j++ ) 
		dst_M(i,j) = dst_M(i,j) / seg_sizes[j];
	
	// Clean up and return
	delete[] elt_labels; elt_labels = 0; 
	delete[] seg_sizes; seg_sizes = 0;  
	for(i=0; i<u.num_sets(); i++) {
		delete[] seg_elts[i];
		seg_elts[i] = 0;
	}
	delete[] seg_elts; seg_elts = 0; 
}


/*
 * Segment a graph
 *
 * Returns a disjoint-set forest representing the segmentation.
 *
 * num_vertices: number of vertices in graph.
 * num_edges: number of edges in graph
 * edges: array of edges.
 * c: constant for treshold function.
 */
void segment::segment_graph(universe &u, int num_vertices, 
						 int num_edges, std::vector<edge> edges, double c) 
{
	int i, a, b;

	// sort edges by weight
	std::sort(edges.begin(), edges.begin()+num_edges);

	// init thresholds
	std::vector<double> threshold;
	for (i = 0; i < num_vertices; i++)
		threshold.push_back( THRESHOLD(1,c) );

	// for each edge, in non-decreasing weight order...
	for (i = 0; i < num_edges; i++) {
		edge *pedge = &edges[i];
		// components conected by this edge
		a = u.find(pedge->a);
		b = u.find(pedge->b);
		if( a==b ) continue;		
		if ((pedge->w <= threshold[a]) && (pedge->w <= threshold[b])) { 
			u.join(a, b);
			a = u.find(a);
			threshold[a] = pedge->w + THRESHOLD(u.size(a), c);
		}	
	}
}


////////////////////////////////////////////////////////////////
// universe: a set of segments
universe::universe(int elements) {
	num = elements;
	for(int i=0; i<elements; i++) {
		uni_seg elt;
		elt.p = i;
		elt.size = 1;
		elt.group.push_back(i);
		elts.push_back(elt);
	}
}

universe::~universe() {
}

int universe::find(int x) {
	int y = x;
	while( y!=elts[y].p )
		y = elts[y].p;
	elts[x].p = y;
	return y;
}

void universe::join(int x, int y) {
	std::list<int>::iterator it;
	if( x<y ) {
		elts[y].p = x;
		for(it=elts[y].group.begin(); it!=elts[y].group.end(); it++)
			elts[*it].p = x;
	} else {
		elts[x].p = y;
		for(it=elts[x].group.begin(); it!=elts[x].group.end(); it++)
			elts[*it].p = y;
	}
	elts[x].size += elts[y].size;
	elts[y].size = elts[x].size;
	elts[x].group.merge(elts[y].group);
	elts[y].group = elts[x].group;
	num--; 
} 

void universe::get_segments(
	int elt_sz, int* elt_labels, int* seg_sizes, int** seg_elts)
{
	int i, j, k, label;
	i=j=k=label=elt_labels[0] = 0;
	for( i=1; i<elt_sz; i++ )
	{
		if( elts[i].p != elts[i-1].p ) {
			seg_sizes[label] = elts[i-1].size;
			seg_elts[label]  = new int[elts[i-1].size];
			for( j=0; j<elts[i-1].size; j++ )
				seg_elts[label][j] = k++;
			label++;
		}
		elt_labels[i] = label;
	}
	seg_sizes[label] = elts[elt_sz-1].size;
	seg_elts[label] = new int[elts[elt_sz-1].p];
	for( j=0; j<elts[elt_sz-1].p; j++ )
		seg_elts[label][j] = k++;
}

void universe::get_segments(int elt_sz, std::list<int> &elt_labels, 
		std::list<int> &seg_sizes, std::vector<std::list<int> > &seg_elts)
{
	int i, j, k, label;
	i=j=k=label=0;
	elt_labels.push_back(0);
	for( i=1; i<elt_sz; i++ )
	{
		if( elts[i].p != elts[i-1].p ) {
			seg_sizes.push_back(elts[i-1].size);
			std::list<int> lst; seg_elts.push_back(lst);
			for( j=0; j<elts[i-1].size; j++ )
				seg_elts[label].push_back(k++);
			label++;
		}
		elt_labels.push_back(label);
	}
	seg_sizes.push_back(elts[elt_sz-1].size);
	std::list<int> lst; seg_elts.push_back(lst);
	for( j=0; j<elts[elt_sz-1].size; j++ )
		seg_elts[label].push_back(k++);
}