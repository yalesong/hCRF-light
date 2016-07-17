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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

//hCRF Library includes
#include "hcrf/dataset.h"
#include "hcrf/model.h"
#include "hcrf/gradient.h"
#include "hcrf/evaluator.h"
#include "hcrf/matrix.h"

class Optimizer {
public:
  Optimizer();
  virtual ~Optimizer();
  
  virtual void optimize(Model* m, DataSet* X, Evaluator* eval, Gradient* grad);
  virtual void setMaxNumIterations(int maxiter);
  virtual int getMaxNumIterations();
  virtual int getLastNbIterations();
  virtual double getLastFunctionError();
  virtual double getLastNormGradient();
  
  // Block optimization for hierarchcal HCRF
  virtual void optimizeBlock(Model* m, DataSet* X, Evaluator* eval, Gradient* grad) {};
  
protected:
  int maxit;
  void setConvergenceTolerance(double tolerance);
  int lastNbIterations;
  double lastFunctionError;
  double lastNormGradient;
};

struct USER_DEFINES;
typedef long int ALLOC_INT;


#ifdef USELBFGS
typedef double lbfgsfloatval_t;
class OptimizerLBFGS: public Optimizer
{
public:
  ~OptimizerLBFGS();
  OptimizerLBFGS();
  OptimizerLBFGS(const OptimizerLBFGS&);
  OptimizerLBFGS& operator=(const OptimizerLBFGS&){
    throw std::logic_error("Optimizer should not be copied");
  }
  void optimize(Model* m, DataSet* X,Evaluator* eval, Gradient* grad);
  void optimizeBlock(Model* m, DataSet* X, Evaluator* eval, Gradient* grad);
  
protected:
  static lbfgsfloatval_t _evaluate( void *instance, const lbfgsfloatval_t *x,
                                   lbfgsfloatval_t *g, const int n,
                                   const lbfgsfloatval_t)
  {
    return reinterpret_cast<OptimizerLBFGS*>(instance)->Eval(x, g, n);
  }
  
  static int _progress( void *instance, const lbfgsfloatval_t *x,
                       const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
                       const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
                       const lbfgsfloatval_t step, int n, int k, int ls )
  {
    return reinterpret_cast<OptimizerLBFGS*>(instance)->progress(
                                                                 x, g, fx, xnorm, gnorm, step, n , k, ls);
  }
  
  double Eval(const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n);
  int progress( const lbfgsfloatval_t *x, const lbfgsfloatval_t *g,
               const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm,
               const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step,
               int n, int k, int ls);
  
  // Block optimization
  static lbfgsfloatval_t _evaluateBlock( void *instance, const lbfgsfloatval_t *x,
                                        lbfgsfloatval_t *g, const int n, const lbfgsfloatval_t) {
    return reinterpret_cast<OptimizerLBFGS*>(instance)->evaluateBlock(x, g, n);
  }
  double evaluateBlock(const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n);
private:
  Model* currentModel;
  DataSet* currentDataset;
  Evaluator* currentEvaluator;
  Gradient* currentGradient;
  dVector vecGradient;
};
#endif

#ifdef USENRBM
class OptimizerNRBM: public Optimizer
{
  
public:
  OptimizerNRBM();
  ~OptimizerNRBM();
  void optimize(Model* m, DataSet* X, Evaluator* eval, Gradient* grad);
  
private:
  struct nrbm_hyper_params {
    bool bRpositive;	// Tell if R(w) is always positive
    bool bRconvex;		// Tell if R(W) is convex
    bool bComputeGapQP;	// Verity the solution of approximation
    bool bLineSearch;	// Activate line search
    bool bCPMinApprox;	// Build cutting plane at minimier of approx. problem
    int maxNbIter;		// max number of iteration
    int maxNbCP;		// max number of cutting plane in approx. problem (NRBM working memory = maxNbCP * dim)
    double epsilon;		// relative tolerance (wrt. value of f)
  };
  
  struct wolfe_params {
    int maxNbIter;
    double a0;
    double a1;
    double c1;		// constant for sufficient reduction (Wolfe condition 1),
    double c2;		// constant for curvature condition (Wolfe condition 2).
    double amax;	// maximum step size allowed
  };
  
  struct reg_params {
    double lambda;	// regularization factor. should be m->regFactor(L1 or L2)
    dVector wreg;	// regularization point
    dVector reg;	// regularization weight
  };
  
  struct qp_params {
    const char* solver;	 // Solver to be used: {mdm|imdm|iimdm|kozinec|kowalczyk|keerthi}.
    int maxNbIter;		 // Maximal number of iterations (default inf).
    double absTolerance; // Absolute tolerance stopping condition (default 0.0).
    double relTolerance; // Relative tolerance stopping condition (default 1e-6).
    double threshLB;	 // Thereshold on the lower bound (default inf).
  };
  
  Model*		m_Model;
  DataSet*	m_Dataset;
  Evaluator*	m_Evaluator;
  Gradient*	m_Gradient;
  
  void NRBM(
            dVector w0,
            dVector& wbest,
            const reg_params& p_reg,
            const nrbm_hyper_params& p_nrbm,
            const wolfe_params& p_wolfe);
  
  // Basic non-convex regularized bundle method for solving the unconstrained problem
  // min_w 0.5 lambda ||w||^2 + R(w)
  void NRBM_kernel(
                   dVector w0,
                   dVector& wnbest,
                   const reg_params& p_reg,
                   const nrbm_hyper_params& p_nrbm,
                   const wolfe_params& p_wolfe);
  
  void FGgrad_objective(
                        dVector w,
                        double& fval,
                        dVector& grad,
                        const reg_params& p_reg);
  
  void minimize_QP(
                   double lambda,
                   dMatrix Q,
                   dVector B,
                   bool Rpositive,
                   double EPS,
                   dVector& alpha,
                   double& dual);
  
  
  int QP_kernel(dMatrix H, dVector c, dVector& alpha, double& dual, const qp_params& p_qp);
  int QP_mdm(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, const qp_params& p_qp);
  int QP_imdm(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, const qp_params& p_qp);
  int QP_iimdm(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, const qp_params& p_qp);
  int QP_kowalczyk(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, const qp_params& p_qp);
  int QP_keerthi(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, const qp_params& p_qp);
  int QP_kozinec(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, const qp_params& p_qp);
  
  //
  // Predicate for sorting std::pair<int,int> wrt. the pair.second() in descending order
  static bool Desc(std::pair<int,int> a, std::pair<int,int> b) { return (a.second>b.second); };
  
  int myLineSearchWolfe(
                        dVector x0,		// current parameter
                        double f0,		// function value at step size 0
                        dVector g0,		// gradient of f wrt its parameter at step size 0
                        dVector s0,		// search direction at step size 0
                        double a1,		// initial step size to start
                        const reg_params& p_reg,
                        const wolfe_params& p_wolfe,
                        double& astar,
                        dVector& xstar,
                        double& fstar,
                        dVector& gstar,
                        dVector& x1,
                        double& f1,
                        dVector& g1);
  
  int myLineSearchZoom(
                       double alo,
                       double ahi,
                       dVector x0,
                       double f0,
                       dVector g0,
                       dVector s0,
                       const reg_params& p_reg,
                       const wolfe_params& p_wolfe,
                       double linegrad0,
                       double falo,
                       double galo,
                       double fhi,
                       double ghi,
                       double& astar,
                       dVector& xstar,
                       double& fstar,
                       dVector& gstar);
  
  
  
  // INPUT
  //		H	[M x N] matrix
  //		a	[D x 1] vector
  //		idx	[D x 1] vector
  // OUTPUT
  //		w	[M x 1] vector
  void w_sum_row(dMatrix H, dVector a, std::list<int> idx, dVector& w);
  
  // INPUT
  //		H	[M x N] matrix
  //		a	[D x 1] vector
  //		idx [D x 1] vector
  // OUTPUT
  //		w = sum( H .* repmat(a,1,N));
  void w_sum_col(dMatrix H, dVector a, std::list<int> idx, dVector& w);
  
  
  // SETDIFF(A,B) when A and B are vectors returns the values in A that are not in B.
  // The result will be sorted.
  std::list<int> matlab_setdiff(std::list<int> list_a, std::list<int> list_b);
};
#endif


#endif



