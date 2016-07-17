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

//-------------------------------------------------------------
// Implementation of Non-convex Regularized Bundle Method
//
// Brief description :
//
//     min 0.5*lambda*((w-wreg).*reg)*((w-wreg).*reg)' + R(w)  (a)
//
// Where
//     wreg : regularization point (default w0)
//      reg : regularization weight (default ones(dim,1))
//
// Let
//     wnew = (w-wreg).* reg  <=> w = wnew ./ reg + wreg
// Then
//     R(w) = R( wnew ./ reg + wreg)
// Let
//     Rn(wnew) = R( wnew ./ reg + wreg)
//
// Minimizing (a) is equivalent to minimizing
//
//     min 0.5*lambda*wnew*wnew' + Rn(wnew)                    (b)
//
// where Rn(wnew) = R( wnew ./ reg + wreg)
//
// The gradient is computed by:
//  d Rn(wnew) / d wnew
//    = d R( wnew ./ reg + w0) / d wnew
//    = (d R(w) / d w) * ( d w / d wnew )
//    = (d R(w) / d wnew) ./ reg
//
// [1] Do and Artieres, "Large Margin Training for Hidden Markov
//     Models with Partially Observed States." ICML 2009.
// [2] http://www.idiap.ch/~do/pmwiki/pmwiki.php/Main/Codes
//
//
// Yale Song  July, 2011

#ifdef USENRBM

#include "hcrf/optimizer.h"

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#define isnan(x) ((x) != (x))
#define HISTORY_BUF 1000000
#define INDEX(ROW,COL,DIM) ((COL*DIM)+ROW)

///////////////////////////////////////////////////////////////////////////
// CONSTRUCTOR / DESTRUCTOR
//
OptimizerNRBM::OptimizerNRBM()
: Optimizer(), m_Model(NULL), m_Dataset(NULL), m_Evaluator(NULL), m_Gradient(NULL)
{}

OptimizerNRBM::~OptimizerNRBM()
{}

///////////////////////////////////////////////////////////////////////////
// PUBLIC
//

void OptimizerNRBM::optimize(Model *m, DataSet *X, Evaluator *eval, Gradient *grad)
{
  m_Model = m;
  m_Dataset = X;
  m_Evaluator = eval;
  m_Gradient= grad;
  
  int nbWeights = m_Model->getWeights()->getLength();
		
  dVector w0 = *(m->getWeights()); // initial solution
  dVector wstar(nbWeights);		 // optimal solution
  
  reg_params p_reg;
  p_reg.lambda = m->getRegL2Sigma(); //1e-3;
  if( p_reg.lambda==0.f ) p_reg.lambda = 1e-3;
  p_reg.wreg.resize(1,nbWeights);
  p_reg.reg.resize(1,nbWeights);
  p_reg.reg.set(1.0);
  
  nrbm_hyper_params p_nrbm;
  p_nrbm.bComputeGapQP	= false;
  p_nrbm.bCPMinApprox		= false;
  p_nrbm.bLineSearch		= true;
  p_nrbm.bRconvex			= true;
  p_nrbm.bRpositive		= true;
  p_nrbm.epsilon			= 1e-2;
  p_nrbm.maxNbCP			= 200;
  p_nrbm.maxNbIter		= maxit; // member variable
  
  wolfe_params p_wolfe;
  p_wolfe.maxNbIter = 5;
  p_wolfe.a0 = 0.01;
  p_wolfe.a1 = 0.5;
  p_wolfe.c1 = 1.000e-04;
  p_wolfe.c2 = 0.9;
  p_wolfe.amax = 1.1;
  
  // Start the NRBM optimization
  NRBM(w0, wstar, p_reg, p_nrbm, p_wolfe);
  
  // Save the optimal weights
  m->setWeights(wstar);
}


///////////////////////////////////////////////////////////////////////////
// PRIVATE
//
//

// Rn(wnew) = R(w) = R( wnew ./ reg + wreg)
void OptimizerNRBM::NRBM(dVector w0, dVector& wbest,
                         const reg_params& p_reg, const nrbm_hyper_params& p_nrbm, const wolfe_params& p_wolfe)
{
  dVector wn0(w0); wn0.subtract(p_reg.wreg); wn0.eltMpy(p_reg.reg); // wn0=(w0-wreg).*reg;
  dVector wnbest(w0.getLength());	// return value from NRBM_kernel
  
  // Run NRBM
  NRBM_kernel(wn0, wnbest, p_reg, p_nrbm, p_wolfe);
  
  // compute (wbest = wnbest ./ reg + wreg);
  wbest.set(wnbest); wbest.eltDiv(p_reg.reg); wbest.add(p_reg.wreg);
}

void OptimizerNRBM::FGgrad_objective(dVector wn, double& fval, dVector& grad, const reg_params& p_reg)
{
  // Compute F and Grad
  dVector w(wn); w.eltDiv(p_reg.reg); w.add(p_reg.wreg); // w = wnew./reg + wreg;
  m_Model->setWeights(w);
  
  fval = m_Gradient->computeGradient(grad,m_Model,m_Dataset);
}

// Basic non-convex regularized bundle method for solving the unconstrained problem
// min_w 0.5 * lambda * ||w||^2 + R(w)
void OptimizerNRBM::NRBM_kernel(dVector w0, dVector& wbest,
                                const reg_params& p_reg, const nrbm_hyper_params& p_nrbm, const wolfe_params& p_wolfe)
{
  int nbWeights = w0.getLength();
  int i,r,c, tbest, cpbest, nbNew, numfeval;
  double s, sum_dist, fcurrent, fbest, Rbest, dual, fstart, astar, gap;
  
  i=r=c=tbest=cpbest=nbNew=numfeval=0;
  s=sum_dist=fcurrent=fbest=Rbest=dual=fstart=astar=gap=0;
  
  dMatrix gVec, Q, Qtmp, newW, newGrad;
  dVector w, alpha, bias, cum_dist, Wtmp, newGrad0, newF;
  uVector cp_iter, inactive;
  dVector wLineSearch(nbWeights), gLineSearch(nbWeights), w1(nbWeights), g1(nbWeights); // for linesearch
  
  // [1] Variable initialization
  gVec.resize(p_nrbm.maxNbCP,nbWeights);		// set of gradients
  Q.resize(p_nrbm.maxNbCP,p_nrbm.maxNbCP);	// precompute gVec*gVec'
  
  w.set(w0);							// Current solution
  alpha.resize(1,p_nrbm.maxNbCP);		// lagrangian multipliers
  bias.resize(1,p_nrbm.maxNbCP);		// bias term
  cum_dist.resize(1,p_nrbm.maxNbCP);	// cumulate distance
  cp_iter.resize(1,p_nrbm.maxNbCP);	// the iteration where cp is built
  inactive.resize(1,p_nrbm.maxNbCP);	// number of consecutive iteration that the cutting plane is inactive (alpha=0)
  Wtmp.resize(1,nbWeights);			// should always be the same size (for efficiency)
  
  inactive.set(p_nrbm.maxNbCP);
  cp_iter.set(1);
  cp_iter[p_nrbm.maxNbCP-1] = p_nrbm.maxNbIter+1; // the last slot is reserved for aggregation cutting plane
  
  s = 0;
  sum_dist = 0;
  fbest = DBL_MAX;
  Rbest = DBL_MAX;
  dual  = -DBL_MAX;
  
  tbest  = 0;	// t, iteration index
  cpbest = 0;	// cp, cutting plane index
  
  // [2] Workspace for cutting plane to be added
  newW.resize(p_wolfe.maxNbIter+1,nbWeights);
  newF.resize(1,p_wolfe.maxNbIter+1);
  newGrad.resize(p_wolfe.maxNbIter+1,nbWeights);
  newGrad0.resize(1,nbWeights);
  
  FGgrad_objective(w0, fstart, newGrad0, p_reg);
  numfeval = 1; // counts num of inference performed
  
  newF[0] = fstart;
  for(int c=0; c<nbWeights; c++) newGrad(c,0) = newGrad0[c];
  for(int c=0; c<nbWeights; c++) newW(c,0) = w0[c];
  
  astar = p_wolfe.a0;
  gap = DBL_MAX;
  nbNew = 1; // one cutting plane to be added in the next iteration
  
  // [3] Main loop
  std::list<int>::iterator it, it_a, it_b;
  std::list<std::pair<int,int> >::iterator it_pair;
  int t;
  for(t=0; t<p_nrbm.maxNbIter; t++)
  {
    // ---------------------------------------------------------------------------------
    // [3-1] Find memory slots of new cutting planes
    // ---------------------------------------------------------------------------------
    std::list<std::pair<int,int> > inactive_cp;
    for(i=0; i<inactive.getLength(); i++)
      inactive_cp.push_back( std::pair<int,int>(i,inactive[i]*p_nrbm.maxNbIter*10-cp_iter[i]) );
    inactive_cp.sort(Desc);
    
    std::list<int> listCP, listCPold, listCPnew;
    for(i=0; i<inactive.getLength(); i++)
      if( inactive[i]<p_nrbm.maxNbCP )
        listCPold.push_back(i);
    for(i=0, it_pair=inactive_cp.begin(); i<nbNew; i++, it_pair++)
      listCPnew.push_back( (*it_pair).first );
    
    listCPold = matlab_setdiff(listCPold,listCPnew);
    for(it=listCPnew.begin(); it!=listCPnew.end(); it++) {
      inactive[*it] = 0;
      cp_iter[*it] = t;
    }
    for(i=0; i<inactive.getLength(); i++)
      if( inactive[i]<p_nrbm.maxNbCP )
        listCP.push_back(i);
    
    // ---------------------------------------------------------------------------------
    // [3-2] Precompute Q for new cutting planes
    // ---------------------------------------------------------------------------------
    // gVec(:,listCPnew) = newGrad(:,1:nbNew) - lambda*newW(:,1:nbNew);
    for(i=0, it=listCPnew.begin(); i<nbNew; i++, it++)
      for(r=0; r<nbWeights; r++)
        gVec(r,*it) = newGrad(r,i) - p_reg.lambda*newW(r,i);
    
    // Q(listCPnew,:) = gVec(:,listCPnew)' * gVec ;
    for(it=listCPnew.begin(); it!=listCPnew.end(); it++) {
      for(c=0; c<p_nrbm.maxNbCP; c++) {
        Q(*it,c) = 0;
        for(r=0; r<nbWeights; r++) {
          // gVec is very sparse, don't multiply unless non-zero
          if( gVec(r,c)==0 ) continue;
          Q(*it,c) += gVec(r,*it) * gVec(r,c);
        }
      }
    }
    //Q(:,listCPnew) = Q(listCPnew,:)';
    Qtmp.resize((int)listCPnew.size(),p_nrbm.maxNbCP);
    for(i=0,it=listCPnew.begin(); it!=listCPnew.end(); i++,it++)
      for(c=0; c<p_nrbm.maxNbCP; c++)
        Qtmp(c,i) = Q(*it,c);
    for(i=0,it=listCPnew.begin(); it!=listCPnew.end(); i++,it++)
      for(r=0; r<p_nrbm.maxNbCP; r++)
        Q(r,*it) = Qtmp(r,i);
    
    // ---------------------------------------------------------------------------------
    // [3-3] Precompute Q for aggregation cutting plane.
    //       This part could be optimized by working only on Q
    // ---------------------------------------------------------------------------------
    // Q(maxCP,:) = gVec(:,maxCP)' * gVec;
    for(c=0; c<p_nrbm.maxNbCP; c++) {
      Q(p_nrbm.maxNbCP-1,c) = 0;
      for(r=0; r<nbWeights; r++)
        Q(p_nrbm.maxNbCP-1,c) += gVec(r,p_nrbm.maxNbCP-1) * gVec(r,c);
    }
    // Q(:,maxCP) = Q(maxCP,:)';
    Qtmp.resize(1,p_nrbm.maxNbCP);
    for(c=0; c<p_nrbm.maxNbCP; c++)
      Qtmp(c,0) = Q(p_nrbm.maxNbCP-1,c);
    for(r=0; r<p_nrbm.maxNbCP; r++)
      Q(r,p_nrbm.maxNbCP-1) = Qtmp(r,0);
    
    
    // ---------------------------------------------------------------------------------
    // [3-4] Add each cutting plane to bundle
    // ---------------------------------------------------------------------------------
    fcurrent = 0;
    dVector wbestold(wbest);
    for(int k=0; k<nbNew; k++) {
      double reg_val, bias_val, Remp, dist, score, gamma, U, L;
      for(i=0; i<nbWeights; i++)
        Wtmp[i] = newW(i,k);
      reg_val = (0.5*p_reg.lambda) * Wtmp.l2Norm(false);
      Remp = newF[k] - reg_val;
      it=listCPnew.begin(); std::advance(it,k); int idx_cp=*it; // idx_cp = j (in original lib)
#if _DEBUG
      assert(idx_cp>=0 && idx_cp<p_nrbm.maxNbCP);
#endif
      bias_val=0;
      for(i=0; i<nbWeights; i++)
        bias_val += gVec(i,idx_cp)*newW(i,k);
      bias[idx_cp] = Remp - bias_val;
      fcurrent = newF[k];
#if _DEBUG
      assert(!isnan(fcurrent));
#endif
      //printf("t=%d, k=%d, idx_cp=%d, fcurrent=%.3f, reg=%.3f, Remp=%.3f, gap=%f%\n", t,k,idx_cp,fcurrent,reg_val,Remp,gap*100/fabs(fbest));
      for(i=0; i<nbWeights; i++)
        Wtmp[i] = wbest[i]-newW(i,k);
      cum_dist[idx_cp] = Wtmp.l2Norm(false);
      
      if( fbest > fcurrent ) {
        fbest = fcurrent;
        Rbest = Remp;
        for(i=0; i<nbWeights; i++)
          wbest[i] = newW(i,k);
        tbest = t;
        cpbest = idx_cp;
        for(i=0; i<nbWeights; i++)
          Wtmp[i] = wbest[i]-wbestold[i];
        dist = Wtmp.l2Norm(false);
        sum_dist = sum_dist + dist;
        //printf("  norm_best=%f, dist=%f, sum_dist=%f\n", wbest.l2Norm(), dist, sum_dist);
        cum_dist[idx_cp] = 0;
      }
      
      // For non-convex optimization, solve conflict
      if( !p_nrbm.bRconvex )
      {
        if( cpbest == idx_cp ) { // CASE 1: DESCENT STEP
          // list = [listCPold;listCPnew(1:k-1);
          std::list<int> listCPo, listCPn;
          for(it=listCPold.begin(); it!=listCPold.end(); it++)
            listCPo.push_back(*it);
          for(i=0, it=listCPnew.begin(); i<k-1 && it!=listCPnew.end(); it++)
            listCPn.push_back(*it);
          
          for(i=0, it=listCPo.begin(); it!=listCPo.end(); it++) {
            score = (0.5*p_reg.lambda)*wbest.l2Norm(false) + bias[i];
            for(int j=0; j<nbWeights; j++)
              score += gVec(j,i)*wbest[j];
            gamma = MAX(0,score - fbest + (0.5*p_reg.lambda)*cum_dist[i]);
            bias[i] -= gamma;
          }
          for(i=0, it=listCPn.begin(); it!=listCPn.end(); it++) {
            score = (0.5*p_reg.lambda)*wbest.l2Norm(false) + bias[i];
            for(int j=0; j<nbWeights; j++)
              score += gVec(j,i)*wbest[j];
            gamma = MAX(0,score - fbest + (0.5*p_reg.lambda)*cum_dist[i]);
            bias[i] -= gamma;
          }
        }
        else { // CASE 2: NULL STEP
          // Estimate g_t at w_tbest
          dist = cum_dist[idx_cp];
          score = (0.5*p_reg.lambda)*dist + bias[idx_cp];
          for(i=0; i<nbWeights; i++)
            score += gVec(i,idx_cp)*wbest[i];
          if( score > Rbest ) { // CONFLICT!
            // Solve conflict by descent g_t so that g_t(w_t) = fbest
            U = Rbest - (0.5*p_reg.lambda)*dist;
            L = fbest - reg_val;
            for(int j=0; j<nbWeights; j++) {
              U -= gVec(j,idx_cp) * wbest[j];
              L -= gVec(j,idx_cp) * newW(j,k);
            }
            //printf("NULL_STEP_CONFLICT Rbest=%f, score=%f, L=%f, U=%f, dist=%f\n", Rbest, score, L, U, dist);
            if( L<=U ) {
              //printf("NULL_STEP_CONFLICT LEVEL_1\n");
              bias[idx_cp] = L;
            }
            else {
              //printf("NULL_STEP_CONFLICT LEVEL_2\n");
              // gVec(:,j) = - lambda * wbest;
              for(i=0; i<nbWeights; i++)
                gVec(i,idx_cp) = -p_reg.lambda*wbest[i];
              // bias(j) = fbest - reg - gVec(:,j)'*newW(:,k);
              bias[idx_cp] = fbest - reg_val;
              for(i=0; i<nbWeights; i++)
                bias[idx_cp] -= gVec(i,idx_cp)*newW(i,k);
              // Q(j,:) = gVec(:,j)' * gVec;
              for(i=0; i<p_nrbm.maxNbCP; i++) {
                Q(idx_cp,i) = 0;
                for(int j=0; j<nbWeights; j++)
                  Q(idx_cp,i) += gVec(j,idx_cp) * gVec(j,i);
              }
              // Q(:,j) = Q(j,:)';
              Qtmp.resize(1,p_nrbm.maxNbCP);
              for(i=0; i<p_nrbm.maxNbCP; i++)
                Qtmp(i,0) = Q(idx_cp,i);
              for(i=0; i<p_nrbm.maxNbCP; i++)
                Q(i,idx_cp) = Qtmp(i,0);
            }
            score = (0.5*p_reg.lambda)*dist + bias[idx_cp];
            for(i=0; i<nbWeights; i++)
              score += gVec(i,idx_cp)*wbest[i];
            //printf("new_score = %f\n", score);
          }
        }
      }
    }
    
    // ---------------------------------------------------------------------------------
    // [3-5] Solve QP program
    // ---------------------------------------------------------------------------------
    // [alpha(listCP),dual] = minimize_QP(lambda,Q(listCP,listCP),bias(listCP),Rpositive ,epsilon);
    // Qtmp = Q(listCP,listCP);
    int size_tmp = (int)listCP.size();
    Qtmp.resize(size_tmp,size_tmp);
    for(r=0, it_a=listCP.begin(); it_a!=listCP.end(); r++, it_a++)
      for(c=0, it_b=listCP.begin(); it_b!=listCP.end(); c++, it_b++)
        Qtmp(r,c) = Q(*it_a,*it_b);
    
    dVector V_tmp(size_tmp), bias_tmp(size_tmp);
    for(i=0, it=listCP.begin(); it!=listCP.end(); i++, it++) {
      V_tmp[i] = alpha[*it];
      bias_tmp[i] = bias[*it];
    }
    
    minimize_QP(p_reg.lambda, Qtmp, bias_tmp, p_nrbm.bRpositive, p_nrbm.epsilon, V_tmp, dual);
    for(i=0, it=listCP.begin(); it!=listCP.end(); i++, it++)
      alpha[*it] = V_tmp[i];
    
    // ---------------------------------------------------------------------------------
    // [3-6] Get QP program solution, update inactive countings and the weight w
    // ---------------------------------------------------------------------------------
    std::list<int> listA, listI, listCPA;
    for(i=0, it=listCP.begin(); it!=listCP.end(); i++, it++) {
      if( V_tmp[i]>0 ) {
        inactive[*it]=0;
        listA.push_back(i);
        listCPA.push_back(*it);
      }
      else if( V_tmp[i]==0 ) {
        inactive[*it]++;
        listI.push_back(i);
      }
    }
    if( listCPA.size()>0 ) { // CHECK IF THIS IS OKAY!!!!!
      V_tmp.resize(1,(int)listCPA.size());
      for(i=0,it=listCPA.begin(); it!=listCPA.end(); i++,it++)
        V_tmp[i] = -alpha[*it]/p_reg.lambda; // warning: V_tmp is overwritten here
      w_sum_row(gVec, V_tmp, listCPA, w);
    }
    inactive[cpbest]=0; // make sure that the best point is always in the set
    
    
    // ---------------------------------------------------------------------------------
    // [3-7] Gradient aggregation
    // ---------------------------------------------------------------------------------
    for(i=0; i<nbWeights; i++)
      gVec(i,p_nrbm.maxNbCP-1) = -p_reg.lambda * w[i];
    bias[p_nrbm.maxNbCP-1] = dual + 0.5*p_reg.lambda*w.l2Norm(false);
    V_tmp.resize(1,(int)listCP.size());
    for(i=0,it=listCP.begin(); it!=listCP.end(); i++,it++)
      V_tmp[i] = cum_dist[*it]; // warning: V_tmp is overwritten here
    for(i=0,it=listCP.begin(); it!=listCP.end(); i++,it++)
      cum_dist[p_nrbm.maxNbCP-1] = alpha[*it] * V_tmp[i];
    inactive[p_nrbm.maxNbCP-1]=0; // make sure that aggregation cp is always active
    
    
    // ---------------------------------------------------------------------------------
    // [3-8] Estimate the gap of approximated dual problem
    // ---------------------------------------------------------------------------------
    if( p_nrbm.bComputeGapQP ) {
      // scoreQP = (w' * gVec)' + bias;
      dVector scoreQP(bias);
      for(i=0; i<p_nrbm.maxNbCP; i++)
        for(int j=0; j<nbWeights; j++)
          scoreQP[i] += w[j] * gVec(j,i);
      double primalQP = 0.5*p_reg.lambda*w.l2Norm(false);
      double max_scoreQP = -DBL_MAX;
      for(it=listCP.begin(); it!=listCP.end(); it++)
        if( scoreQP[*it] > max_scoreQP ) max_scoreQP = scoreQP[*it];
      if( !(max_scoreQP<0&&p_nrbm.bRpositive) ) primalQP += max_scoreQP;
      //double gapQP = primalQP - dual;
      //printf("  quadratic_programming: primal=%f, dual=%f, gap=%f\n",primalQP,dual,gapQP*100/fabs(primalQP));
    }
    gap = fbest-dual;
    // skip the freport thingy
    
    // ---------------------------------------------------------------------------------
    // [3-9] Output
    // ---------------------------------------------------------------------------------
    if( m_Model->getDebugLevel() >= 1 ) {
      printf("  t=%d, nfeval=%d, f=%f, f*=%f, R*=%f, gap=%.2f\n", t, numfeval,
             fcurrent, fbest, Rbest, gap*100/fabs(fbest));
    }
    if( gap/fabs(fbest)<p_nrbm.epsilon || gap<1e-6 || t>=p_nrbm.maxNbIter )
      break;
    if( !p_nrbm.bLineSearch ) {
      nbNew = 1;
      double newF0 = 0;
      FGgrad_objective(w,newF0,newGrad0,p_reg);
      newF[0] = newF0;
      for(i=0; i<nbWeights; i++)
        newGrad(i,0) = newGrad0[i];
      for(i=0; i<nbWeights; i++)
        newW(i,0) = w[i];
      numfeval++;
      continue;
    }
    
    // ---------------------------------------------------------------------------------
    // [3-10] Perform line search from wbest to w
    // ---------------------------------------------------------------------------------
    dVector search_direction(w); search_direction.subtract(wbest);
    double norm_direction = search_direction.l2Norm();
    double astart;
    if( p_nrbm.bCPMinApprox || t==0 ) {
      astart = 1.0;
    }
    else {
      astart = MIN(astar/norm_direction,1.0);
      if( astart==0 ) astart=1.0;
    }
    dVector g0(wbest); g0.multiply(p_reg.lambda);
    for(i=0; i<nbWeights; i++)
      g0[i] += gVec(i,cpbest);
    double fLineSearch,f1;
    numfeval += myLineSearchWolfe(
                                  wbest,fbest,g0,search_direction,astart,p_reg,p_wolfe,
                                  astar,wLineSearch,fLineSearch,gLineSearch,w1,f1,g1);
    if( f1!=fLineSearch ) {
      nbNew = 2;
      newF[0]=f1; newF[1]=fLineSearch;
      for(i=0; i<nbWeights; i++) {
        newW(i,0)=w1[i];	   newW(i,1)=wLineSearch[i];
        newGrad(i,0)=g1[i];	newGrad(i,1)=gLineSearch[i];
      }
    }
    else {
      nbNew = 1;
      newF[0] = fLineSearch;
      for(i=0; i<nbWeights; i++) {
        newW(i,0)=wLineSearch[i];
        newGrad(i,0)=gLineSearch[i];
      }
    }
    if( fbest<=fLineSearch && astart!=1 ) {
      numfeval++; nbNew++;
      double newF0=0; newGrad0.set(0); FGgrad_objective(w,newF0,newGrad0,p_reg);
      newF[nbNew-1] = newF0;
      for(i=0; i<nbWeights; i++) {
        newGrad(i,nbNew-1) = newGrad0[i];
        newW(i,nbNew-1) = w[i];
      }
    }
    astar = astar*norm_direction; // true step length
    //printf("> step_length = %f\n", astar);
  }
  lastFunctionError = fcurrent;
  lastNbIterations = t;
  printf("DONE_DRBM numfeval=%d\n", numfeval);
}

void OptimizerNRBM::minimize_QP(double lambda, dMatrix Q, dVector B, bool Rpositive, double EPS,
                                dVector& alpha, double& dual)
{
  int i,r,c;
  int size_var = Q.getHeight();
  if( size_var+Rpositive==1 ) {
    alpha[0] = 1;
    dual = -0.5 * Q(0,0)/lambda + B[0];
    return;
  }
  if( Rpositive ) {
    dMatrix Qtmp(Q);
    dVector Btmp(B);
    Q.resize(Q.getWidth()+1,Q.getHeight()+1);
    B.resize(1,B.getLength()+1);
    for(i=0; i<Btmp.getLength(); i++)
      B[i] = Btmp[i];
    for(r=0; r<Qtmp.getHeight(); r++)
      for(c=0; c<Qtmp.getWidth(); c++)
        Q(r,c) = Qtmp(r,c);
  }
  double scale = Q.absmax() / (1000*lambda);
  double tmp = (scale!=0) ? 1/scale : DBL_MAX;
  Q.multiply(tmp);
  B.multiply(tmp);
  
  dMatrix Qtmp(Q); Qtmp.multiply(1/lambda); // This must be positive definite
  dVector Btmp(B); Btmp.negate(); Btmp.transpose();
  
  std::list<std::string> list_solvers;
  std::list<std::string>::iterator it;
  list_solvers.push_back(std::string("imdm"));
  list_solvers.push_back(std::string("kowalczyk"));
  list_solvers.push_back(std::string("keerthi"));
  
  qp_params p_qp;
  p_qp.absTolerance = 0.0;
  p_qp.relTolerance = 1e-2*EPS;
  p_qp.threshLB = DBL_MAX;
  
  int exit_flag = -1;
  for(int k=6; k<=10; k++) {
    for(it=list_solvers.begin(); it!=list_solvers.end(); it++)
    {
      // Call the QP solver
      p_qp.solver = (*it).c_str();
      p_qp.maxNbIter = (int)pow((float)10,(float)k);
      exit_flag = QP_kernel(Qtmp, Btmp, alpha, dual, p_qp);
      
      // verify the solution of the approximated dual problem.
      double eps = 1e-4;
      if( alpha.min()<-eps || fabs(alpha.sum()-1)>eps )
        exit_flag = 0;
      if( exit_flag>0 )
        break;
    }
    if( exit_flag!=0 )
      break;
  }
  
#if _DEBUG
  if( exit_flag==0 )
    printf("Warning: minimize_QP() solving QP of approx. problem failed. Did not reach enough accuracy\n");
#endif
  
  B.multiply(scale);
  dual = -dual * scale;
  if( Rpositive ) {
    dVector V_tmp(alpha);
    alpha.resize(1,size_var);
    for(i=0; i<size_var; i++)
      alpha[i] = V_tmp[i];
  }
}



void OptimizerNRBM::w_sum_row(dMatrix H, dVector a, std::list<int> idx, dVector &w)
{
  int M = H.getHeight();
  //int N = H.getWidth();
  int Da = a.getLength();
  //int Di = (int) idx.size();
  //int Dw = w.getLength();
  
#if _DEBUG
  if( Di!=Da || Da>N || M!=Dw ) {
    printf("w_sum_row() dimension mismatch!");
    printf(" H[%d x %d], a[%d], idx[%d], w[%d]\n", M,N,Da,Di,Dw);
    getchar();
    exit(-1);
  }
#endif
  int i,j;
  std::list<int>::iterator it;
  for(j=0; j<M; j++) {
    w[j] = 0.0;
    for(i=0,it=idx.begin(); i<Da; i++,it++)
      w[j] += H(j,*it) * a[i];
  }
}

void OptimizerNRBM::w_sum_col(dMatrix H, dVector a, std::list<int> idx, dVector &w)
{
  //int M = H.getHeight();
  int N = H.getWidth();
  int Da = a.getLength();
  //int Di = (int) idx.size();
  //int Dw = w.getLength();
  
#if _DEBUG
  if( Di!=Da || Da>N || N!=Dw ) {
    printf("w_sum_col() dimension mismatch!");
    getchar();
    printf(" H[%d x %d], a[%d], idx[%d], w[%d]\n", M,N,Da,Di,Dw);
    exit(-1);
  }
#endif
  int i,j;
  std::list<int>::iterator it;
  for(j=0; j<N; j++) {
    w[j] = 0.0;
    for(i=0,it=idx.begin(); i<Da; i++,it++)
      w[j] += H(*it,j) * a[i];
  }
}


// Reimplementation of some Matlab functions
std::list<int> OptimizerNRBM::matlab_setdiff(
                                             std::list<int> list_a, std::list<int> list_b)
{
  std::list<int> diff_sorted;
  std::list<int>::iterator it_a, it_b, it_c;
  for(it_a=list_a.begin(); it_a!=list_a.end(); it_a++)
    diff_sorted.push_back(*it_a);
  for(it_b=list_b.begin(); it_b!=list_b.end(); it_b++) {
    for(it_c=diff_sorted.begin(); it_c!=diff_sorted.end(); ) {
      if( *it_b==*it_c )
        diff_sorted.erase(it_c++);
      else
        it_c++;
    }
  }
  diff_sorted.sort();
  diff_sorted.unique();
  return diff_sorted;
}



//-------------------------------------------------------------
// Implementation of Line search method satisfying strong
// Wolfe condition.
//
// Determines a line search step size that satisfies the strong
// Wolfe condition. The algorithm implements what was described
// in Numerial Optimization (Nocedal and Wright, Springer 1999),
// Algol 3.5 (Line Search Algorithm) and Algol. 3.6 (Zoom).
//
// Inputs:
//	lambda: regularization factor for FGgrad_objective(...)
//      x0: current parameter
//      f0: function value at step size 0
//      g0: gradient of f with respect to its parameter at step size 0
//      s0: search direction at step size 0
//      a1: initial step size to start
//       p: wolfe constants
// 	  amax:  maximum step size allowed
// 	    c1:  the constant for sufficient reduction (Wolfe condition 1),
//      c2:  the constant for curvature condition (Wolfe condition 2).
// maxiter: maxium iteration to search for ste size
//
// Outputs
//   x1,f1,g1 are the solution, fval and grad correspondant to stepsize a1
//
// Yale Song October, 2011


int OptimizerNRBM::myLineSearchWolfe(
                                     dVector x0, double f0, dVector g0, dVector s0, double a1,
                                     const reg_params& p_reg, const wolfe_params& p_wolfe,
                                     double &astar, dVector &xstar, double &fstar, dVector &gstar,
                                     dVector &x1, double &f1, dVector &g1)
{
  int numeval = 0; // FGgrad_objective() call count. Value to be returned
  
  dVector Wtmp;
  
  double ai,ai_1,fi,fi_1;
  ai_1 = 0;
  ai = a1;		// initial step size to start
  fi_1 = f0;		// function value at step size 0
  
  double linegrad0, linegradi, linegradi_1;
  Wtmp.set(g0); Wtmp.eltMpy(s0); linegrad0 = Wtmp.sum();
  linegradi_1 = linegrad0;
  
  x1.set(s0); x1.multiply(ai); x1.add(x0); // x1 = x0 + ai*s0;
  FGgrad_objective(x1,f1,g1,p_reg); numeval++; // save initial objective values
  
  for(int i=0;; i++) {
    xstar.set(s0); xstar.multiply(ai); xstar.add(x0); // xstar = x0 + ai*s0;
    fi=0; dVector gi(x0.getLength());
    FGgrad_objective(xstar,fi,gi,p_reg); numeval++;
    
    Wtmp.set(gi); Wtmp.eltMpy(s0);
    linegradi = Wtmp.sum();
    if( fi>=fi_1 || fi>(f0+p_wolfe.c1*ai*linegrad0) ) {
      numeval += myLineSearchZoom(
                                  ai_1,ai,x0,f0,g0,s0,p_reg,p_wolfe,
                                  linegrad0,fi_1,linegradi_1,fi,linegradi,
                                  astar,xstar,fstar,gstar);
      return numeval;
    }
    
    if( fabs(linegradi)<=-p_wolfe.c2*linegrad0 ) {
      astar=ai; fstar=fi; gstar.set(gi);
      return numeval;
    }
    
    if( linegradi>=0 ) {
      numeval += myLineSearchZoom(
                                  ai,ai_1,x0,f0,g0,s0,p_reg,p_wolfe,
                                  linegrad0,fi,linegradi,fi_1,linegradi_1,
                                  astar,xstar,fstar,gstar);
      return numeval;
    }
    i++;
    if( fabs(ai-p_wolfe.amax)<=0.01*p_wolfe.amax || i>=p_wolfe.maxNbIter ) {
      astar=ai; fstar=fi; gstar.set(gi);
      return numeval;
    }
    ai_1 = ai;
    fi_1 = fi;
    linegradi_1 = linegradi;
    ai = (ai+p_wolfe.amax)/2;
  }
}

int OptimizerNRBM::myLineSearchZoom(
                                    double alo, double ahi, dVector x0, double f0, dVector g0, dVector s0,
                                    const reg_params& p_reg, const wolfe_params& p_wolfe,
                                    double linegrad0, double falo, double galo, double fhi, double ghi,
                                    double &astar, dVector &xstar, double &fstar, dVector &gstar)
{
  int numeval = 0; // FGgrad_objective() call count. Value to be returned
  
  dVector Wtmp;
  
  double aj,fj;
  double d1,d2;
  double linegradj;
  
  for(int i=0;;i++) {
    d1 = galo+ghi - 3*(falo-fhi)/(alo-ahi);
    d2 = sqrt(MAX(0,d1*d1 - galo*ghi));
    aj = ahi - (ahi-alo)*(ghi+d2-d1)/(ghi-galo+2*d2);
    if( isnan(aj) ) aj=-DBL_MAX; // to avoid numerical error
    if( alo<ahi ) {
      if( aj<alo || aj>ahi )
        aj = (alo+ahi)/2;
    }
    else {
      if( aj>alo || aj<ahi )
        aj = (alo+ahi)/2;
    }
    xstar.set(s0); xstar.multiply(aj); xstar.add(x0); // xstar = x0 + aj*s0;
    fj=0; dVector gj(x0.getLength());
    FGgrad_objective(xstar,fj,gj,p_reg); numeval++;
    
    if( fj>falo || fj>(f0+p_wolfe.c1*aj*linegrad0) ) {
      ahi=aj; fhi=fj;
      Wtmp.set(gj); Wtmp.eltMpy(s0); ghi=Wtmp.sum();
    }
    else {
      Wtmp.set(gj); Wtmp.eltMpy(s0); linegradj=Wtmp.sum();
      if( fabs(linegradj)<=-p_wolfe.c2*linegrad0 ) {
        astar=aj; fstar=fj; gstar.set(gj);
        return numeval;
      }
      if( linegradj*(ahi-alo)>=0 ) {
        ahi=alo; fhi=falo; ghi=galo;
      }
      alo=aj; falo=fj; galo=linegradj;
    }
    
    if( fabs(alo-ahi)<=0.01*alo || i>=p_wolfe.maxNbIter ) {
      astar=aj; fstar=fj; gstar.set(gj);
      return numeval;
    }
  }
}






//-------------------------------------------------------------
// Implementation of General QP Solver, based on STPRtoolbox [1]
//
// The Generalized Minimal Norm Problem to solve
//    min_x 0.5*x'*H*x + f'*x  subject to: sum(x) = 1 and x >= 0.
//
// H is symetric positive-definite matrix. The GMNP is a special
// instance of the Quadratic Programming (QP) task. The GMNP
// is solved by one of the following algorithms:
//    mdm        Mitchell-Demyanov-Malozemov
//    imdm       Improved Mitchell-Demyanov-Malozemov (default).
//    iimdm      Improved (version 2) Mitchell-Demyanov-Malozemov.
//    kozinec    Kozinec algorithm.
//    keerthi    Derived from NPA algorithm by Keerthi et al.
//    kowalczyk  Based on Kowalczyk's maximal margin perceptron.
//
// The optimization halts if one of the following stopping conditions is satisfied:
//	tmax <= t                -> exit_flag = 0
//	tolabs >= UB-LB          -> exit_flag = 1
//	tolrel*abs(UB) >= UB-LB  -> exit_flag = 2
//	thlb < LB                -> exit_flag = 3
// where t is number of iterations, UB/LB are upper/lower bounds on the optimal solution.
//
//  For more info refer to V.Franc: Optimization Algorithms for Kernel
//  Methods. Research report. CTU-CMP-2005-22. CTU FEL Prague. 2005.
//  ftp://cmp.felk.cvut.cz/pub/cmp/articles/franc/Franc-PhD.pdf .
//
// Input:
//	H [dim x dim] Symmetric positive definite matrix.
//	c [dim x 1] Vector.
//	solver [string] GMNP solver: options are 'mdm', 'imdm', 'iimdm','kowalczyk','keerthi','kozinec'.
//	tmax [1x1] Maximal number of iterations.
//	tolabs [1x1] Absolute tolerance stopping condition.
//	tolrel [1x1] Relative tolerance stopping condition.
//	thlb [1x1] Threshold on lower bound.
//
// Output:
//  alpha [dim x 1] Solution vector.
//  exitflag [1x1] Indicates which stopping condition was used:
//    UB-LB <= tolabs           ->  exit_flag = 1   Abs. tolerance.
//    UB-LB <= UB*tolrel        ->  exit_flag = 2   Relative tolerance.
//    LB > th                   ->  exit_flag = 3   Threshold on LB.
//    t >= tmax                 ->  exit_flag = 0   Number of iterations.
//  t [1x1] Number of iterations.
//  access [1x1] Access to elements of the matrix H.
//  History [2x(t+1)] UB and LB with respect to number of iterations.
//
// [1] http://cmp.felk.cvut.cz/cmp/software/stprtool/
//
// Yale Song (yalesong@csail.mit.edu)
// October, 2011



int OptimizerNRBM::QP_kernel(dMatrix H, dVector c, dVector& alpha, double& dual, const qp_params& p_qp)
{
  int dim = H.getWidth();
  
#ifdef _DEBUG
  if( H.getHeight()!=H.getWidth() ) {
    fprintf(stderr, "QP_kernel(): H must be squared.\n"); getchar(); exit(-1);
  }
  if( H.getHeight()!=c.getLength() ) {
    fprintf(stderr, "QP_kernel(): H and c dimension mismatch.\n"); getchar(); exit(-1);
  }
#endif
  
  dVector diag_H(dim); for(int i=0; i<dim; i++) diag_H[i] = H(i,i);
  
  int exitflag = 0;
  if( !strcmp(p_qp.solver,"mdm") )
    exitflag = QP_mdm(H, diag_H, c, alpha, dual, p_qp);
  else if( !strcmp(p_qp.solver,"imdm") )
    exitflag = QP_imdm(H, diag_H, c, alpha, dual, p_qp);
  else if( !strcmp(p_qp.solver,"iimdm") )
    exitflag = QP_iimdm(H, diag_H, c, alpha, dual, p_qp);
  else if( !strcmp(p_qp.solver,"keerthi") )
    exitflag = QP_keerthi(H, diag_H, c, alpha, dual, p_qp);
  else if( !strcmp(p_qp.solver,"kowalczyk") )
    exitflag = QP_kowalczyk(H, diag_H, c, alpha, dual, p_qp);
  else if( !strcmp(p_qp.solver,"kozinec") )
    exitflag = QP_kozinec(H, diag_H, c, alpha, dual, p_qp);
  else {
    fprintf(stderr, "Unknown QP solver (%s).\n", p_qp.solver); getchar(); exit(-1);
  }
  
  return exitflag;
  
}

int OptimizerNRBM::QP_mdm(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, const qp_params& p_qp)
{
  int dim = H.getHeight();
  
  // Return values
  alpha.resize(1,dim);
  dual = 0;
  
  // Variables
  double lambda, LB,UB, aHa,ac, tmp,tmp1, Huu,Huv,Hvv, beta,min_beta,max_beta;
  int u,v, new_u,new_v, i,t,exitflag; //,History_size;
  dVector Ha; //, History;
  
  lambda=LB=UB=aHa=ac=tmp=tmp1=Huu=Huv=Hvv=beta=min_beta=max_beta=0;
  u=v=new_u=new_v=i=t=exitflag=0;
  
  // Initialization
  Ha.create(dim);
  //History_size = (p_qp.maxNbIter<HISTORY_BUF) ? p_qp.maxNbIter+1 : HISTORY_BUF;
  //History.create(History_size*2);
  
  tmp1 = DBL_MAX;
  for(i=0; i<dim; i++) {
    tmp = 0.5*diag_H[i] + c[i];
    if( tmp1 > tmp ) {
      tmp1 = tmp;
      v = i;
    }
  }
  
  min_beta = DBL_MAX;
  for(i=0; i<dim; i++) {
    alpha[i] = 0;
    Ha[i] = H(i,v);
    beta = Ha[i] + c[i];
    if( beta < min_beta ) {
      min_beta = beta;
      u = i;
    }
  }
  
  alpha[v] = 1;
  aHa = diag_H[v];
  ac = c[v];
  
  UB = 0.5*aHa + ac;
  LB = min_beta - 0.5*aHa;
  t = 0;
  //History[INDEX(0,0,2)] = LB;
  //History[INDEX(1,0,2)] = UB;
  
  //printf("QP_mdm(): init: UB=%f, LB=%f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB,LB,UB-LB,(UB-LB)/UB);
  
  // Stopping conditions
  if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
  else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
  else if( LB > p_qp.threshLB )					exitflag = 3;
  else											exitflag = -1;
  
  // Main QP Optimization Loop
  while( exitflag==-1 )
  {
    // Adaptation rule and update
    Huu = diag_H[u];
    Hvv = diag_H[v];
    Huv = H(v,u);
    
    lambda = (Ha[v]-Ha[u]+c[v]-c[u]) / (alpha[v]*(Huu-2*Huv+Hvv));
    if( lambda<0 ) lambda=0; else if( lambda>1 ) lambda=1;
    
    aHa = aHa + 2*alpha[v]*lambda*(Ha[u]-Ha[v])
    + lambda*lambda*alpha[v]*alpha[v]*(Huu-2*Huv+Hvv);
    
    ac = ac + lambda*alpha[v]*(c[u]-c[v]);
    
    tmp = alpha[v];
    alpha[u] += lambda*alpha[v];
    alpha[v] -= lambda*alpha[v];
    
    UB = 0.5*aHa + ac;
    
    min_beta = DBL_MAX;
    max_beta = -DBL_MAX;
    for(i=0; i<dim; i++) {
      Ha[i] += lambda*tmp*(H(i,u)-H(i,v));
      beta = Ha[i] + c[i];
      if( alpha[i]!=0 && max_beta<beta ) {
        new_v = i;
        max_beta = beta;
      }
      
      if( beta<min_beta ) {
        new_u = i;
        min_beta = beta;
      }
    }
    
    LB = min_beta - 0.5*aHa;
    u = new_u;
    v = new_v;
    
    // Stopping conditions
    if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
    else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
    else if( LB > p_qp.threshLB )					exitflag = 3;
    else if( t >= p_qp.maxNbIter )					exitflag = 0;
    
    //printf("QP_mdm():   %d: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", t, UB, LB, UB-LB, (UB-LB)/UB);
    
    // Store selected values
    //		if( t < History_size ) {
    //			History[INDEX(0,t,2)] = LB;
    //			History[INDEX(1,t,2)] = UB;
    //		}
    //		else
    //			printf("QP_mdn(): WARNING: History() is too small. Won't be logged\n");
    
    t++;
  }
  
  // Print info about last iteration
  //printf("QP_mdm():   exit: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB, LB, UB-LB, (UB-LB)/UB);
  
  dual = UB;
  return exitflag;
}

int OptimizerNRBM::QP_imdm(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, const qp_params& p_qp)
{
  int dim = H.getHeight();
  
  // Return values
  alpha.resize(1,dim);
  dual = 0;
  
  // Variables
  double lambda, LB,UB, aHa,ac, tmp,tmp1, Huu,Huv,Hvv, beta,min_beta, max_improv,improv;
  int u,v, new_u, i,t,exitflag; //,History_size;
  dVector Ha; //, History;
  
  lambda=LB=UB=aHa=ac=tmp=tmp1=Huu=Huv=Hvv=beta=min_beta=max_improv=improv=0;
  u=v=new_u=i=t=exitflag=0;
  
  // Initialization
  Ha.create(dim);
  //History_size = (p_qp.maxNbIter<HISTORY_BUF) ? p_qp.maxNbIter+1 : HISTORY_BUF;
  //History.create(History_size*2);
  
  tmp1 = DBL_MAX;
  for(i=0; i<dim; i++) {
    tmp = 0.5*diag_H[i] + c[i];
    if( tmp1 > tmp ) {
      tmp1 = tmp;
      v = i;
    }
  }
  
  min_beta = DBL_MAX;
  for(i=0; i<dim; i++) {
    alpha[i] = 0;
    Ha[i] = H(i,v);
    beta = Ha[i] + c[i];
    if( beta < min_beta ) {
      min_beta = beta;
      u = i;
    }
  }
  
  alpha[v] = 1;
  aHa = diag_H[v];
  ac = c[v];
  
  UB = 0.5*aHa + ac;
  LB = min_beta - 0.5*aHa;
  t = 0;
  //History[INDEX(0,0,2)] = LB;
  //History[INDEX(1,0,2)] = UB;
  
  //printf("QP_imdm(): init: UB=%f, LB=%f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB,LB,UB-LB,(UB-LB)/UB);
  
  // Stopping conditions
  if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
  else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
  else if( LB > p_qp.threshLB )					exitflag = 3;
  else											exitflag = -1;
  
  // Main QP Optimization Loop
  int u0 = u;
  while( exitflag==-1 )
  {
    // Adaptation rule and update
    Huu = diag_H[u];
    Hvv = diag_H[v];
    Huv = H(v,u0);
    
    lambda = (Ha[v]-Ha[u]+c[v]-c[u]) / (alpha[v]*(Huu-2*Huv+Hvv));
    if( lambda<0 ) lambda=0; else if( lambda>1 ) lambda=1;
    
    aHa = aHa + 2*alpha[v]*lambda*(Ha[u]-Ha[v])
    + lambda*lambda*alpha[v]*alpha[v]*(Huu-2*Huv+Hvv);
    
    ac = ac + lambda*alpha[v]*(c[u]-c[v]);
    
    tmp = alpha[v];
    alpha[u] += lambda*alpha[v];
    alpha[v] -= lambda*alpha[v];
    
    UB = 0.5*aHa + ac;
    
    min_beta = DBL_MAX;
    for(i=0; i<dim; i++) {
      Ha[i] += lambda*tmp*(H(i,u0)-H(i,v));
      beta = Ha[i] + c[i];
      
      if( beta<min_beta ) {
        new_u = i;
        min_beta = beta;
      }
    }
    
    LB = min_beta - 0.5*aHa;
    u = new_u;
    u0 = u;
    
    // Search for the optimal v while u is fixed
    max_improv = -DBL_MAX;
    for(i=0; i<dim; i++) {
      if( alpha[i]!=0 ) {
        beta = Ha[i] + c[i];
        if( beta >= min_beta ) {
          tmp = diag_H[u] - 2*H(i,u0) + diag_H[i];
          if( tmp!=0 ) {
            improv = (0.5*(beta-min_beta)*(beta-min_beta))/tmp;
            if( improv>max_improv ) {
              max_improv = improv;
              v = i;
            }
          }
        }
      }
    }
    
    // Stopping conditions
    if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
    else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
    else if( LB > p_qp.threshLB )					exitflag = 3;
    else if( t >= p_qp.maxNbIter )					exitflag = 0;
    
    //printf("QP_mdm():   %d: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", t, UB, LB, UB-LB, (UB-LB)/UB);
    
    // Store selected values
    //		if( t < History_size ) {
    //			History[INDEX(0,t,2)] = LB;
    //			History[INDEX(1,t,2)] = UB;
    //		}
    //		else
    //			printf("QP_mdn(): WARNING: History() is too small. Won't be logged\n");
    
    t++;
  }
  
  // Print info about last iteration
  //printf("QP_mdm():   exit: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB, LB, UB-LB, (UB-LB)/UB);
  
  dual = UB;
  return exitflag;
  return 0;
}

int OptimizerNRBM::QP_iimdm(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, const qp_params& p_qp)
{
  int dim = H.getHeight();
  
  // Return values
  alpha.resize(1,dim);
  dual = 0;
  
  // Variables
  double lambda, LB,UB, aHa,ac, tmp,tmp1, Huu,Huv,Hvv,
		beta,min_beta,max_beta, max_improv1,max_improv2,improv;
  int u,v, new_u,new_v, i,t,exitflag; //,History_size;
  dVector Ha; //, History;
  
  
  lambda=LB=UB=aHa=ac=tmp=tmp1=Huu=Huv=Hvv=beta=min_beta=max_beta=max_improv1=max_improv2=improv=0;
  u=v=new_u=new_v=i=t=exitflag=0;
  
  // Initialization
  Ha.create(dim);
  //History_size = (p_qp.maxNbIter<HISTORY_BUF) ? p_qp.maxNbIter+1 : HISTORY_BUF;
  //History.create(History_size*2);
  
  tmp1 = DBL_MAX;
  for(i=0; i<dim; i++) {
    tmp = 0.5*diag_H[i] + c[i];
    if( tmp1 > tmp ) {
      tmp1 = tmp;
      v = i;
    }
  }
  
  min_beta = DBL_MAX;
  for(i=0; i<dim; i++) {
    alpha[i] = 0;
    Ha[i] = H(i,v);
    beta = Ha[i] + c[i];
    if( beta < min_beta ) {
      min_beta = beta;
      u = i;
    }
  }
  
  alpha[v] = 1;
  aHa = diag_H[v];
  ac = c[v];
  
  UB = 0.5*aHa + ac;
  LB = min_beta - 0.5*aHa;
  t = 0;
  //History[INDEX(0,0,2)] = LB;
  //History[INDEX(1,0,2)] = UB;
  
  //printf("QP_mdm(): init: UB=%f, LB=%f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB,LB,UB-LB,(UB-LB)/UB);
  
  // Stopping conditions
  if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
  else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
  else if( LB > p_qp.threshLB )					exitflag = 3;
  else											exitflag = -1;
  
  // Main QP Optimization Loop
  int u0 = u;
  int v0 = v;
  while( exitflag==-1 )
  {
    // Adaptation rule and update
    Huu = diag_H[u];
    Hvv = diag_H[v];
    Huv = H(v,u0);
    
    lambda = (Ha[v]-Ha[u]+c[v]-c[u]) / (alpha[v]*(Huu-2*Huv+Hvv));
    if( lambda<0 ) lambda=0; else if( lambda>1 ) lambda=1;
    
    aHa = aHa + 2*alpha[v]*lambda*(Ha[u]-Ha[v])
    + lambda*lambda*alpha[v]*alpha[v]*(Huu-2*Huv+Hvv);
    
    ac = ac + lambda*alpha[v]*(c[u]-c[v]);
    
    tmp = alpha[v];
    alpha[u] += lambda*alpha[v];
    alpha[v] -= lambda*alpha[v];
    
    UB = 0.5*aHa + ac;
    
    min_beta = DBL_MAX;
    max_beta = -DBL_MAX;
    for(i=0; i<dim; i++) {
      Ha[i] += lambda*tmp*(H(i,u0)-H(i,v0));
      beta = Ha[i] + c[i];
      
      if( beta<min_beta ) {
        new_u = i;
        min_beta = beta;
      }
      
      if( alpha[i]!=0 && max_beta<beta ) {
        new_v = i;
        max_beta = beta;
      }
    }
    
    LB = min_beta - 0.5*aHa;
    u0 = new_u;
    v0 = new_v;
    
    // Search for the optimal v while u is fixed
    max_improv1 = max_improv2 = -DBL_MAX;
    for(i=0; i<dim; i++) {
      beta = Ha[i] + c[i];
      if( alpha[i]!=0 && beta>min_beta ) {
        tmp = diag_H[new_u] - 2*H(i,u0) + diag_H[i];
        if( tmp!=0 ) {
          if( (beta-min_beta)/(alpha[i]*tmp) < 1 )
            improv = (0.5*(beta-min_beta)*(beta-min_beta))/tmp;
          else
            improv = alpha[i]*(beta-min_beta) - 0.5*alpha[i]*alpha[i]*tmp;
          
          if( improv>max_improv1 ) {
            max_improv1 = improv;
            v = i;
          }
        }
      }
      
      if( max_beta>beta ) {
        tmp = diag_H[new_v] - 2*H(i,v0) + diag_H[i];
        if( tmp!=0 ) {
          if( (max_beta-beta)/(alpha[new_v]*tmp) < 1 )
            improv = (0.5*(max_beta-beta)*(max_beta-beta))/tmp;
          else
            improv = alpha[new_v]*(max_beta-beta) - 0.5*alpha[new_v]*alpha[new_v]*tmp;
          
          if( improv>max_improv2 ) {
            max_improv2 = improv;
            u = i;
          }
        }
      }
    }
    
    if( max_improv1 > max_improv2 ) {
      u = new_u;
      v0 = v;
    }
    else {
      v = new_v;
      u = u0;
    }
    
    // Stopping conditions
    if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
    else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
    else if( LB > p_qp.threshLB )					exitflag = 3;
    else if( t >= p_qp.maxNbIter )					exitflag = 0;
    
    //printf("QP_mdm():   %d: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", t, UB, LB, UB-LB, (UB-LB)/UB);
    
    // Store selected values
    //		if( t < History_size ) {
    //			History[INDEX(0,t,2)] = LB;
    //			History[INDEX(1,t,2)] = UB;
    //		}
    //		else
    //			printf("QP_mdn(): WARNING: History() is too small. Won't be logged\n");
    
    t++;
  }
  
  // Print info about last iteration
  //printf("QP_mdm():   exit: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB, LB, UB-LB, (UB-LB)/UB);
  
  dual = UB;
  return exitflag;
}

int OptimizerNRBM::QP_keerthi(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, const qp_params& p_qp)
{
  int dim = H.getHeight();
  
  // Return values
  alpha.resize(1,dim);
  dual = 0;
  
  // Variables
  double LB,UB, aHa,ac, tmp,tmp1, Huu,Huv,Hvv, beta,min_beta,max_beta;
  double den,gamma,omega,a1,a2,a3,a4,a5,x10,x11,x12,x13,x20,x22,x23,x30,x33;
  double UB123,gamma1,gamma2,gamma3,tmp_aHa1,tmp_aHa2,tmp_aHa3,tmp_ac1,tmp_ac2,tmp_ac3,UB1,UB2,UB3;
  int nearest_segment, u,v, i,t,exitflag; //,History_size;
  dVector Ha; //, History;
  
  LB=UB=aHa=ac=tmp=tmp1=Huu=Huv=Hvv=beta=min_beta=max_beta=den=gamma=omega
		=a1=a2=a3=a4=a5=x10=x11=x12=x13=x20=x22=x23=x30=x33=UB123
		=gamma1=gamma2=gamma3=tmp_aHa1=tmp_aHa2=tmp_aHa3=tmp_ac1=tmp_ac2=tmp_ac3
		=UB1=UB2=UB3=0;
  nearest_segment=u=v=i=t=exitflag=0;
  
  // Initialization
  Ha.create(dim);
  //History_size = (p_qp.maxNbIter<HISTORY_BUF) ? p_qp.maxNbIter+1 : HISTORY_BUF;
  //History.create(History_size*2);
  
  tmp1 = DBL_MAX;
  for(i=0; i<dim; i++) {
    tmp = 0.5*diag_H[i] + c[i];
    if( tmp1 > tmp ) {
      tmp1 = tmp;
      v = i;
    }
  }
  
  min_beta = DBL_MAX;
  for(i=0; i<dim; i++) {
    alpha[i] = 0;
    Ha[i] = H(i,v);
    beta = Ha[i] + c[i];
    if( beta < min_beta ) {
      min_beta = beta;
      u = i;
    }
  }
  
  alpha[v] = 1;
  aHa = diag_H[v];
  ac = c[v];
  
  UB = 0.5*aHa + ac;
  LB = min_beta - 0.5*aHa;
  t = 0;
  //History[INDEX(0,0,2)] = LB;
  //History[INDEX(1,0,2)] = UB;
  
  //printf("QP_mdm(): init: UB=%f, LB=%f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB,LB,UB-LB,(UB-LB)/UB);
  
  // Stopping conditions
  if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
  else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
  else if( LB > p_qp.threshLB )					exitflag = 3;
  else											exitflag = -1;
  
  // Main QP Optimization Loop
  while( exitflag==-1 )
  {
    // Adaptation rule and update
    Huu = diag_H[u];
    Hvv = diag_H[v];
    Huv = H(v,u);
    
    x11 = aHa;
    x12 = Ha[u];
    x13 = aHa + alpha[v]*(Ha[u]-Ha[v]);
    x22 = Huu;
    x23 = Ha[u] + alpha[v]*(Huu-Huv);
    x33 = aHa + 2*alpha[v]*(Ha[u]-Ha[v]) + alpha[v]*alpha[v]*(Huu-2*Huv+Hvv);
    
    x10 = ac;
    x20 = c[u];
    x30 = ac + alpha[v]*(c[u]-c[v]);
    
    a1 = x11 - x12 - x13 + x23;
    a2 = x11 - 2*x12 + x22;
    a3 = x12 - x11 + x20 - x10;
    a4 = x11 - 2*x13 + x33;
    a5 = x13 - x11 + x30 - x10;
    
    den = a1*a1 - a2*a4;
    if( den ) {
      gamma = (a3*a4 - a1*a5)/den;
      omega = (a2*a5 - a3*a1)/den;
      
      if( gamma>0 && omega>0 && 1-gamma-omega>0 ) {
        // Ha = Ha*(1-gamma) + H(:,u)*(gamma+alpha(v)*omega)-H(:,v)*alpha(v)*omega;
        tmp = alpha[v]*omega;
        for(i=0; i<dim; i++)
          Ha[i] = Ha[i]*(1-gamma) + H(i,u)*(gamma+tmp) - H(i,v)*tmp;
        
        // aHa = (1-omega-gamma)^2*x11 + gamma^2*x22 + omega^2*x33 + ...
        //       2*(1-omega-gamma)*gamma*x12 + 2*(1-omega-gamma)*omega*x13 + ...
        //       2*gamma*omega*x23;
        aHa = (1-omega-gamma)*(1-omega-gamma)*x11 + gamma*gamma*x22
        + omega*omega*x33 + 2*(1-omega-gamma)*gamma*x12
        + 2*(1-omega-gamma)*omega*x13 + 2*gamma*omega*x23;
        ac = (1-gamma-omega)*x10 + gamma*x20 + omega*x30;
        
        
        // alpha1 = zeros(dim,1);
 			    // alpha1(u) = 1;
        // alpha2 = alpha;
        // alpha2(u) = alpha(u)+alpha(v);
        // alpha2(v) = 0;
        // alpha = alpha*(1-gamma-omega) + alpha1*gamma + alpha2*omega;
        for(i=0; i<dim; i++)
          alpha[i] *= (1-gamma);
        
        alpha[u] += (gamma + tmp);
        alpha[v] -= tmp;
        
        UB123 = 0.5*aHa + ac;
      }
      else {
        UB123 = DBL_MAX;
      }
    }
    else {
      UB123 = DBL_MAX;
    }
    
    if( UB123 == DBL_MAX ) {
      // line segment between alpha and alpha1
      gamma1   = (x11-x12+x10-x20)/(x11-2*x12+x22);
      gamma1   = MIN(1,gamma1);
      tmp_aHa1 = (1-gamma1)*(1-gamma1)*x11 + 2*gamma1*(1-gamma1)*x12 + gamma1*gamma1*x22;
      tmp_ac1  = (1-gamma1)*x10 + gamma1*x20;
      UB1      = 0.5*tmp_aHa1 + tmp_ac1;
      
      // line segment between alpha and alpha2
      gamma2   = (x11-x13+x10-x30)/(x11-2*x13+x33);
      gamma2   = MIN(1,gamma2);
      tmp_aHa2 = (1-gamma2)*(1-gamma2)*x11 + 2*gamma2*(1-gamma2)*x13 + gamma2*gamma2*x33;
      tmp_ac2  = (1-gamma2)*x10 + gamma2*x30;
      UB2      = 0.5*tmp_aHa2 + tmp_ac2;
      
      //  line segment between alpha1 and alpha2
      den = (x22 - 2*x23 + x33);
      if( den ) {
        gamma3 = (x22-x23+x20-x30)/den;
        if( gamma3 > 1 ) gamma3 = 1;
        if( gamma3 < 0 ) gamma3 = 0;
        tmp_aHa3 = (1-gamma3)*(1-gamma3)*x22 + 2*gamma3*(1-gamma3)*x23 + gamma3*gamma3*x33;
        tmp_ac3  = (1-gamma3)*x20 + gamma3*x30;
        UB3      = 0.5*tmp_aHa3 + tmp_ac3;
      }
      else {
        UB3 = UB;
      }
      
      // nearest_segment = argmin( UB1, UB2, UB3 )
      if( UB1<=UB2 ) { if( UB1<=UB3 ) nearest_segment=1; else nearest_segment=3; }
      else { if( UB2<=UB3 ) nearest_segment=2; else nearest_segment=3; }
      
      switch( nearest_segment ) {
        case 1:
          aHa = tmp_aHa1;
          ac = tmp_ac1;
          for(i=0; i<dim; i++) {
            Ha[i] = Ha[i]*(1-gamma1) + gamma1*H(i,u);
            alpha[i] = alpha[i]*(1-gamma1);
          }
          alpha[u] += gamma1;
          break;
          
        case 2:
          aHa = tmp_aHa2;
          ac = tmp_ac2;
          tmp = alpha[v]*gamma2;
          for(i=0; i<dim; i++)
            Ha[i] = Ha[i] + tmp*(H(i,u)-H(i,v));
          alpha[u] += tmp;
          alpha[v] -= tmp;
          break;
          
        case 3:
          aHa = tmp_aHa3;
          ac = tmp_ac3;
          tmp = alpha[v]*gamma3;
          for(i=0; i<dim; i++) {
            Ha[i] = gamma3*Ha[i] + H(i,u)*(1-gamma3+tmp) - tmp*H(i,v);
            alpha[i] = alpha[i]*gamma3;
          }
          alpha[u] += (1 - gamma3 + tmp);
          alpha[v] -= tmp;
          break;
      }
    }
    
    UB = 0.5*aHa + ac;
    min_beta = DBL_MAX;
    max_beta = -DBL_MAX;
    for(i=0; i<dim; i++) {
      beta = Ha[i] + c[i];
      if( alpha[i]!=0 && max_beta<beta ) {
        v = i;
        max_beta = beta;
      }
      if( beta<min_beta ) {
        u = i;
        min_beta = beta;
      }
    }
    
    LB = min_beta - 0.5*aHa;
    
    // Stopping conditions
    if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
    else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
    else if( LB > p_qp.threshLB )					exitflag = 3;
    else if( t >= p_qp.maxNbIter )					exitflag = 0;
    
    //printf("QP_mdm():   %d: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", t, UB, LB, UB-LB, (UB-LB)/UB);
    
    // Store selected values
    //		if( t < History_size ) {
    //			History[INDEX(0,t,2)] = LB;
    //			History[INDEX(1,t,2)] = UB;
    //		}
    //		else
    //			printf("QP_mdn(): WARNING: History() is too small. Won't be logged\n");
    
    t++;
  }
  
  // Print info about last iteration
  //printf("QP_mdm():   exit: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB, LB, UB-LB, (UB-LB)/UB);
  
  dual = UB;
  return exitflag;
}

int OptimizerNRBM::QP_kowalczyk(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, const qp_params& p_qp)
{
  int dim = H.getHeight();
  
  // Return values
  alpha.resize(1,dim);
  dual = 0;
  
  // Variables
  double LB,UB, aHa,ac, tmp,tmp1, beta,min_beta;
  double x10,x11,x12,x20,x22, gamma,delta,tmp_UB,tmp_gamma,tmp_aHa,tmp_ac;
  int inx, i,t,exitflag; //,History_size;
  dVector Ha; //, History;
  
  LB=UB=aHa=ac=tmp=tmp1=beta=min_beta=x10=x11=x12=x20=x22=gamma=delta=tmp_UB=tmp_gamma=tmp_aHa=tmp_ac=0;
  inx=i=t=exitflag=0;
  
  // Initialization
  Ha.create(dim);
  //History_size = (p_qp.maxNbIter<HISTORY_BUF) ? p_qp.maxNbIter+1 : HISTORY_BUF;
  //History.create(History_size*2);
  
  tmp1 = DBL_MAX;
  for(i=0; i<dim; i++) {
    tmp = 0.5*diag_H[i] + c[i];
    if( tmp1 > tmp ) {
      tmp1 = tmp;
      inx = i;
    }
  }
  
  min_beta = DBL_MAX;
  for(i=0; i<dim; i++) {
    alpha[i] = 0;
    Ha[i] = H(i,inx);
    beta = Ha[i] + c[i];
    if( beta < min_beta ) {
      min_beta = beta; 
    }
  }
  
  alpha[inx] = 1;
  aHa = diag_H[inx];
  ac = c[inx];
  
  UB = 0.5*aHa + ac;
  LB = min_beta - 0.5*aHa;
  t = 0;
  //History[INDEX(0,0,2)] = LB;
  //History[INDEX(1,0,2)] = UB;
  
  //printf("QP_mdm(): init: UB=%f, LB=%f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB,LB,UB-LB,(UB-LB)/UB);
  
  // Stopping conditions
  if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
  else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
  else if( LB > p_qp.threshLB )					exitflag = 3;
  else											exitflag = -1;
  
  // Main QP Optimization Loop
  while( exitflag==-1 )
  { 
    x11 = aHa;
    x10 = ac;
    
    // search for the rule that yields the biggest improvement
    for(i=0; i<dim; i++) {
      delta = Ha[i] + c[i] - aHa - ac;
      
      tmp_UB = DBL_MAX;
      if( delta<0 ) {
        // Kozinec rule
        x12 = Ha[i];
        x20 = c[i];
        x22 = diag_H[i];
        
        tmp_gamma = (x11-x12+x10-x20)/(x11-2*x12+x22);
        tmp_gamma = MIN(1,tmp_gamma);
        tmp_aHa = (1-tmp_gamma)*(1-tmp_gamma)*x11 + 2*(1-tmp_gamma)*tmp_gamma*x12 + tmp_gamma*tmp_gamma*x22;
        tmp_ac  = (1-tmp_gamma)*x10 + tmp_gamma*x20;
        tmp_UB  = 0.5*tmp_aHa + tmp_ac;
      }
      else if( delta>0 && alpha[i]<1 && alpha[i]>0 ) {
        x12 = (x11 - alpha[i]*Ha[i])/(1-alpha[i]);
        x22 = (x11 - 2*alpha[i]*Ha[i] + alpha[i]*alpha[i]*diag_H[i])/((1-alpha[i])*(1-alpha[i]));
        x20 = (x10 - alpha[i]*c[i])/(1-alpha[i]);
        
        tmp_gamma = (x11-x12+x10-x20)/(x11-2*x12+x22);
        tmp_gamma = MIN(1,tmp_gamma);
        tmp_aHa = (1-tmp_gamma)*(1-tmp_gamma)*x11 + 2*(1-tmp_gamma)*tmp_gamma*x12 + tmp_gamma*tmp_gamma*x22;
        tmp_ac  = (1-tmp_gamma)*x10 + tmp_gamma*x20;
        tmp_UB  = 0.5*tmp_aHa + tmp_ac;
      }
      
      if( tmp_UB<UB ) {
        UB = tmp_UB;
        gamma = tmp_gamma;
        aHa = tmp_aHa;
        ac = tmp_ac;
        inx = i;
      }
    }
    
    // Use the update with the biggest improvement
    delta = Ha[inx] + c[inx] - x11 - x10;
    if( delta<0 ) {
      // Kozinec rule
      for(i=0; i<dim; i++) {
        Ha[i] = Ha[i]*(1-gamma) + gamma*H(i,inx);
        alpha[i] = alpha[i]*(1-gamma);
      }
      alpha[inx] = alpha[inx] + gamma;
    }
    else {
      // Inverse Kozinec rule
      tmp = gamma*alpha[inx];
      tmp1 = 1-alpha[inx];
      for(i=0; i<dim; i++) {
        Ha[i] = (Ha[i]*(tmp+tmp1) - tmp*H(i,inx))/tmp1;
        alpha[i] = alpha[i]*(1-gamma) + gamma*alpha[i]/tmp1;
      }
      alpha[inx] = alpha[inx] - tmp/tmp1;
    }
    
    min_beta = DBL_MAX;
    for(i=0; i<dim; i++) {
      beta = Ha[i] + c[i];
      if( beta<min_beta ) 
        min_beta = beta;
    }
    
    LB = min_beta - 0.5*aHa;
    
    // Stopping conditions
    if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
    else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
    else if( LB > p_qp.threshLB )					exitflag = 3;
    else if( t >= p_qp.maxNbIter )					exitflag = 0;
    
    //printf("QP_mdm():   %d: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", t, UB, LB, UB-LB, (UB-LB)/UB);
    
    // Store selected values
    //		if( t < History_size ) {
    //			History[INDEX(0,t,2)] = LB;
    //			History[INDEX(1,t,2)] = UB;
    //		} 
    //		else 
    //			printf("QP_mdn(): WARNING: History() is too small. Won't be logged\n"); 
    
    t++;
  }
  
  // Print info about last iteration
  //printf("QP_mdm():   exit: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB, LB, UB-LB, (UB-LB)/UB); 
  
  dual = UB;
  return exitflag;
}

int OptimizerNRBM::QP_kozinec(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, const qp_params& p_qp)
{
  return 0;
}

#endif
