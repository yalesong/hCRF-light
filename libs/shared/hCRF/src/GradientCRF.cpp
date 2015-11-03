//-------------------------------------------------------------
// Hidden Conditional Random Field Library - GradientCRF
// Component
//
//	February 2, 2006

#include "gradient.h"

GradientCRF::GradientCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen) 
: Gradient(infEngine, featureGen)
{}

double GradientCRF::computeGradient(dVector& vecGradient, Model* m, DataSequence* X)
{
	if( m->isMaxMargin() )
		return computeGradientMaxMargin(vecGradient,m,X);
	else
		return computeGradientMLE(vecGradient,m,X);
}

double GradientCRF::computeGradientMLE(dVector& vecGradient, Model* m, DataSequence* X)
{
	int xi, xj, yi, yj, k, nbFeatures;
	double val, phi, partition;
	Beliefs bel;
	
	// Compute beliefs
	pInfEngine->computeBeliefs(bel,pFeatureGen, X, m, false);
	phi = pFeatureGen->evaluateLabels(X,m);
	partition = bel.partition;
 
	// Compute gradients

	// Check the size of vecGradient
	nbFeatures = pFeatureGen->getNumberOfFeatures();
	if(vecGradient.getLength() != nbFeatures)
		vecGradient.create(nbFeatures);

	feature* f;
	featureVector vecFeatures;

	// Loop over nodes to compute features and update the gradient
	for(xi=0; xi<X->length(); xi++) {
		yi = X->getStateLabels(xi);
		
		//Get nodes features
		pFeatureGen->getFeatures(vecFeatures,X,m,xi,-1);
		f = vecFeatures.getPtr();						
		for(k=0; k<vecFeatures.size(); k++, f++) 
		{
			if(f->nodeState == yi)
				vecGradient[f->id] += f->value;

			// p(y_i=s|x)*f_k(xi,s,x) is subtracted from the gradient 
			val = bel.belStates[xi][f->nodeState]*f->value;
			vecGradient[f->id] -= val;
		}
	}

	//Loop over edges to compute features and update the gradient
	for(xi=0; xi<X->length()-1; xi++) {
		xj = xi+1;
		yi = X->getStateLabels(xi);
		yj = X->getStateLabels(xj);

		//Get nodes features
		pFeatureGen->getFeatures(vecFeatures,X,m,xj,xi);
		f = vecFeatures.getPtr();						
		for(k=0; k<vecFeatures.size(); k++, f++)
		{
			if(f->prevNodeState == yi && f->nodeState == yj)
				vecGradient[f->id] += f->value;

			//p(y_i=s1,y_j=s2|x)*f_k(i,j,s1,s2,x) is subtracted from the gradient 
			val = bel.belEdges[xi](f->prevNodeState,f->nodeState)*f->value;
			vecGradient[f->id] -= val;
		}
	}

	//Return -log instead of log() [Moved to Gradient::ComputeGradient by LP]
	return partition-phi;
}


// Based on A.2 Structured Loss in Teo et al. JMLR 2010, p343-344, 
double GradientCRF::computeGradientMaxMargin(dVector& vecGradient, Model* m, DataSequence* X)
{
	int xi, xj, yi, yj, k, nbFeatures;
	double val, phi_star=0, phi_true=0, hamming_loss=0;

	Beliefs bel;
	pInfEngine->computeBeliefs(bel,pFeatureGen, X, m, false);

	// Compute Hamming loss
	iVector ystar; dVector pystar;
	viterbiDecoding(bel,ystar,pystar);
	for(xi=0; xi<X->length(); xi++) 
		if( X->getStateLabels(xi) != ystar[xi] ) 
			hamming_loss++;

	// Compute gradients
	feature* f;
	featureVector vecFeatures;

	nbFeatures = pFeatureGen->getNumberOfFeatures();
	if(vecGradient.getLength() != nbFeatures)
		vecGradient.create(nbFeatures);

	dVector *w = m->getWeights();
	dVector localGrad(nbFeatures); 

	// Loop over nodes to compute features and update the gradient
	for(xi=0; xi<X->length(); xi++)
	{
		// Read the label for this state
		yi = X->getStateLabels(xi);

		//Get nodes features
		pFeatureGen->getFeatures(vecFeatures,X,m,xi,-1);
		f = vecFeatures.getPtr();						
		for(k=0; k<vecFeatures.size(); k++, f++) {
			if(f->nodeState==yi) {
				phi_true += w->getValue(f->id) * f->value; 
				localGrad[f->id] -= f->value;
			}
			else if(f->nodeState==ystar[xi]) {
				phi_star += w->getValue(f->id) * f->value;
				localGrad[f->id] += f->value;
			}
			val = bel.belStates[xi][f->nodeState]*f->value;
			vecGradient[f->id] -= val;
		}
	}

	//Loop over edges to compute features and update the gradient
	for(xi=0; xi<X->length()-1; xi++) {
		xj = xi+1;
		yi = X->getStateLabels(xi);
		yj = X->getStateLabels(xj);

		//Get nodes features
		pFeatureGen->getFeatures(vecFeatures,X,m,xj,xi);
		f = vecFeatures.getPtr();						
		for(k=0; k<vecFeatures.size(); k++, f++)
		{
			if(f->prevNodeState == yi && f->nodeState == yj) {
				phi_true += w->getValue(f->id) * f->value;
				localGrad[f->id] -= f->value;
			}
			else if(f->prevNodeState==ystar[xi] && f->nodeState==ystar[xj]) {
				phi_star += w->getValue(f->id) * f->value;
				localGrad[f->id] += f->value;
			}
			val = bel.belEdges[xi](f->prevNodeState,f->nodeState)*f->value;
			localGrad[f->id] -= val;
		}
	}

	// Taskar et al. (2004) vs Tsochantaridis et al. (2005)
	bool useTaskar = false; 
	double scale = (useTaskar) ? 1 : hamming_loss;

	// Done!
	localGrad.multiply(scale); 
	vecGradient.add(localGrad);
 

	return hamming_loss + scale*(exp(phi_star-bel.partition) - exp(phi_true-bel.partition));
}
 