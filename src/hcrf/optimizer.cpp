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


#include "hcrf/optimizer.h"

Optimizer::Optimizer() : maxit(-1),
lastNbIterations(-1),
lastFunctionError(-1),
lastNormGradient(-1)
{
}

Optimizer::~Optimizer()
{
}


void Optimizer::optimize(Model* m, DataSet* X,
                         Evaluator* eval, Gradient* grad)
{
}


void Optimizer::setMaxNumIterations(int maxiter)
{
  maxit = maxiter;
}


int Optimizer::getMaxNumIterations()
{
  return maxit;
}

int Optimizer::getLastNbIterations()
{
  return lastNbIterations;
}

double Optimizer::getLastFunctionError()
{
  return lastFunctionError;
}

double Optimizer::getLastNormGradient()
{
  return lastNormGradient;
}
