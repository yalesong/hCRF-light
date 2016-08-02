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


#include "hcrf/toolbox.h"

ToolboxMVLDCRF::ToolboxMVLDCRF(): ToolboxLDCRF()
{}


ToolboxMVLDCRF::ToolboxMVLDCRF(eGraphTypes gt, int nv,
                               std::vector<int> nhs, std::vector<std::vector<int> > rfi)
: ToolboxLDCRF(), m_nbViews(nv), m_rawFeatureIndex(rfi), m_graphType(gt)
{
  m_nbHiddenStatesMultiView = new int[m_nbViews]; // For LDCRF, this is per label
  for( int i=0; i<m_nbViews; i++ )
    m_nbHiddenStatesMultiView[i] = nhs[i];
  
  // This is necessary to prevent Toolbox::addFeatureFunction() from discarding
  // same typed features with different view index
  pModel->setNumberOfViews(m_nbViews);
  pModel->setAdjacencyMatType(m_graphType);
  pModel->setRawFeatureIndexMV(m_rawFeatureIndex);
  pModel->setNumberOfStatesMV(m_nbHiddenStatesMultiView);
}


ToolboxMVLDCRF::~ToolboxMVLDCRF()
{
  if( m_nbHiddenStatesMultiView ) {
    delete [] m_nbHiddenStatesMultiView;
    m_nbHiddenStatesMultiView = 0;
  }
}


void ToolboxMVLDCRF::initToolbox()
{
  // Add features
  if (!pFeatureGenerator)
    pFeatureGenerator = new FeatureGenerator();
  
  if (!pFeatureGenerator->getFeatureByBasicType(NODE_FEATURE)) {
    for(int i=0; i<m_nbViews; i++) {
      addFeatureFunction(MV_GAUSSIAN_WINDOW_RAW_FEATURE_ID, i, 0);
    }
  }
  
  // Edge features (pairwise potentials)
  for( int i=0; i<m_nbViews; i++ )
  {
    // add MV_EDGE_WITHIN_VIEW (c==d, s+1=t)
    addFeatureFunction(MV_EDGE_FEATURE_ID, i, i);
    
    // add MV_EDGE_BETWEEN_VIEW (c!=d, s==t)
    if( m_graphType==MV_GRAPH_LINKED || m_graphType==MV_GRAPH_LINKED_COUPLED )
      for( int j=i+1; j<m_nbViews; j++ )
        addFeatureFunction(MV_EDGE_FEATURE_ID, i, j);
    
    // add MV_EDGE_CROSS_VIEW (c!=d, s+1=t)
    if( m_graphType==MV_GRAPH_COUPLED || m_graphType==MV_GRAPH_LINKED_COUPLED ) {
      for( int j=0; j<m_nbViews; j++ ) {
        if( i==j ) continue;
        addFeatureFunction(MV_EDGE_FEATURE_ID, i, j);
      }
    }
  }
  
  if( !pInferenceEngine )
    pInferenceEngine = new InferenceEngineLoopyBP();
  if( !pGradient )
    pGradient = new GradientMVLDCRF(pInferenceEngine, pFeatureGenerator);
  if( !pEvaluator )
    pEvaluator = new EvaluatorMVLDCRF(pInferenceEngine, pFeatureGenerator);
}


void ToolboxMVLDCRF::initModel(DataSet &X)
{
  int nbStateLabels = X.searchNumberOfStates();
  int* nbStates = new int[m_nbViews];
  for(int i=0; i<m_nbViews; i++)
    nbStates[i] = m_nbHiddenStatesMultiView[i] * nbStateLabels;
  pModel->setNumberOfStatesMV(nbStates);
  delete[] nbStates; nbStates = 0;
  
  pModel->setNumberOfStateLabels(nbStateLabels);
  pModel->setStateMatType(STATES_BASED_ON_LABELS);
  
  pFeatureGenerator->initFeatures(X, *pModel); 
}
