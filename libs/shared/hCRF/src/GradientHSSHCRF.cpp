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


#include "gradient.h"
#include "Features.h"
#include "HierarchicalFeatures.h"

GradientHSSHCRF::GradientHSSHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen) 
: Gradient(infEngine, featureGen)
{}

double GradientHSSHCRF::computeGradient(dVector& vecGradient, Model* m, DataSequence* X)
{
	if(m->getCurrentFeatureLayer() >= (int)X->getDeepSeqGroupLabels()->size())
		return 0.0;

	return computeGradientMLE(vecGradient,m,X);
}

double GradientHSSHCRF::computeGradientMLE(dVector& vecGradient, Model* m, DataSequence* X)
{
	int k, xi, y, nbSeqLabels, nbFeatures, offset, seqLength;
	int gidx, nbGates; // Neural layer params
	double val, f_val, lZx;

	nbSeqLabels  = m->getNumberOfSequenceLabels();
	nbFeatures   = pFeatureGen->getNumberOfFeatures();
	nbGates      = m->getNbGates();
	offset       = m->getCurrentFeatureLayer() * X->length();
	seqLength    = (int) X->getDeepSeqGroupLabels()->at(m->getCurrentFeatureLayer()).size();

	std::vector<Beliefs> condBeliefs(nbSeqLabels);
	dMatrix condEValues(nbFeatures, nbSeqLabels);
	dVector Partition(nbSeqLabels);

	////////////////////////////////////////////////////////////////////////////////////
	// Step 1 : Run Inference in each network to compute marginals conditioned on Y
	for(y=0; y<nbSeqLabels; y++) {
		pInfEngine->computeBeliefs(condBeliefs[y], pFeatureGen, X, m, true, y);
		Partition[y] = condBeliefs[y].partition;
	}
	lZx = Partition.logSumExp();
	f_val = lZx - Partition[X->getSequenceLabel()];

	////////////////////////////////////////////////////////////////////////////////////
	// Step 2 : Compute expected values for node/edge features conditioned on Y
	feature* f;
	featureVector vecFeatures;

	// Neural gate features
	dVector* W = m->getWeights();
	dVector gateProbWeightSum;
	HSSGateNodeFeatures* HSSgateF;

	// Compute gradient for singleton features
	if( nbGates==0 )
	{
		for(y=0; y<nbSeqLabels; y++) { 
			// Loop over nodes to compute features and update the gradient
			for(xi=0; xi<seqLength; xi++) {
				pFeatureGen->getFeatures(vecFeatures,X,m,xi+offset,-1,y);
				f = vecFeatures.getPtr();						
				for(k=0; k<vecFeatures.size(); k++, f++) 	
				{
					//if(m->getCurrentFeatureLayer()>0) {printf("[s] typeID=%d,gid=%d,id=%d,idx=[%d,%d],state=[%d,%d],view=[%d,%d],y=%d,val=%f\n",f->featureTypeId,f->globalId,f->id,f->prevNodeIndex,f->nodeIndex,f->prevNodeState,f->nodeState,f->prevNodeView,f->nodeView,f->sequenceLabel,f->value);}
					// p(h^v_t=a|x,y) * f_k(v,t,a,x,y)
					val = condBeliefs[y].belStates[xi][f->nodeState] * f->value;
					condEValues.addValue(y, f->globalId, val);
				} 
			} 
		}
	}
	else
	{
		gateProbWeightSum.resize(1,nbGates);
		HSSgateF = (HSSGateNodeFeatures*) pFeatureGen->getFeatureById(HSS_GATE_NODE_FEATURE_ID);

		for(y=0; y<nbSeqLabels; y++) { 	
			for(xi=0; xi<seqLength; xi++) {
				gateProbWeightSum.set(0); // don't forget to reset this to zero			
				pFeatureGen->getFeatures(vecFeatures,X,m,offset+xi,-1,y);
				f = vecFeatures.getPtr();						
				for(k=0; k<vecFeatures.size(); k++, f++)
				{
					//if(m->getCurrentFeatureLayer()>0) {printf("[s] typeID=%d,gid=%d,id=%d,idx=[%d,%d],state=[%d,%d],view=[%d,%d],y=%d,val=%f\n",f->featureTypeId,f->globalId,f->id,f->prevNodeIndex,f->nodeIndex,f->prevNodeState,f->nodeState,f->prevNodeView,f->nodeView,f->sequenceLabel,f->value);}
					// p(h^v_t=a|x,y) * f_k(v,t,a,x,y)
					val = condBeliefs[y].belStates[xi][f->nodeState] * f->value;
					condEValues.addValue(y, f->globalId, val);
					if( f->featureTypeId == HSS_GATE_NODE_FEATURE_ID) {
						val = W->getValue(f->globalId) * ((1.0-f->value)*f->value);
						gidx = f->prevNodeState; // gate index (quick-and-dirty solution)
						gateProbWeightSum.addValue(gidx, condBeliefs[y].belStates[xi][f->nodeState]*val);
					}
				} 
						
				HSSgateF->getPreGateFeatures(vecFeatures,X,m,offset+xi,-1,y);
				f = vecFeatures.getPtr();
				for(k=0; k<vecFeatures.size(); k++, f++) 
				{
					//if(m->getCurrentFeatureLayer()>0) {printf("[s] typeID=%d,gid=%d,id=%d,idx=[%d,%d],state=[%d,%d],view=[%d,%d],y=%d,val=%f\n",f->featureTypeId,f->globalId,f->id,f->prevNodeIndex,f->nodeIndex,f->prevNodeState,f->nodeState,f->prevNodeView,f->nodeView,f->sequenceLabel,f->value);}
					gidx = f->prevNodeState; // gate index 
					val = f->value*gateProbWeightSum[gidx];
					condEValues.addValue(y, f->globalId, val);
				}
			} 
		}
	}

	// Compute gradient for pairwise features
	for(y=0; y<nbSeqLabels; y++)
	{ 	
		// Loop over edges to compute features and update the gradient
		for(xi=1; xi<seqLength; xi++) {
			pFeatureGen->getFeatures(vecFeatures,X,m,offset+xi,offset+xi-1,y);
			f = vecFeatures.getPtr();				
			for(k=0; k<vecFeatures.size(); k++, f++) 
			{
				//if(m->getCurrentFeatureLayer()>0) {printf("[p] typeID=%d,gid=%d,id=%d,idx=[%d,%d],state=[%d,%d],view=[%d,%d],y=%d,val=%f\n",f->featureTypeId,f->globalId,f->id,f->prevNodeIndex,f->nodeIndex,f->prevNodeState,f->nodeState,f->prevNodeView,f->nodeView,f->sequenceLabel,f->value);}
				// p(h^vi_ti=a,h^vj_tj=b|x,y) * f_k(vi,ti,vj,tj,x,y)
				val = condBeliefs[y].belEdges[xi-1](f->prevNodeState,f->nodeState) * f->value;
				condEValues.addValue(y, f->globalId, val);
			} 
		} 	
	}

	// Step 3: Compute Joint Expected Values
	dVector JointEValues(nbFeatures), rowJ(nbFeatures);
	for(y=0; y<nbSeqLabels; y++) {
		condEValues.getRow(y, rowJ);
		rowJ.multiply( exp(Partition[y]-lZx) );
		JointEValues.add(rowJ);
	}
  
	// Step 4 Compute Gradient as Exi[i,*,*] -Exi[*,*,*], that is difference
	// between expected values conditioned on Sequence Labels and Joint expected values
	condEValues.getRow(X->getSequenceLabel(), rowJ); // rowJ=Expected value conditioned on Sequence label Y
	JointEValues.negate();
	rowJ.add(JointEValues);

	vecGradient.add(rowJ);
	//if(m->getCurrentFeatureLayer()>0) getchar();	
	return f_val;
}
