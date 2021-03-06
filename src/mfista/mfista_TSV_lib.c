/* 
   Copyright (C) 2015   Shiro Ikeda <shiro@ism.ac.jp>

   This is file 'mfista_lib.c'. An optimization algorithm for imaging
   of interferometry. The idea of the algorithm was from the following
   two papers,

   Beck and Teboulle (2009) SIAM J. Imaging Sciences, 
   Beck and Teboulle (2009) IEEE trans. on Image Processing 


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
#include "mfista.h"

double calc_F_TSV_part(int *M, int NX, int NY, double *yAx, double *xvec, double lambda_tsv)
{
  int inc = 1;
  double term1, term2;

  term1 = ddot_(M, yAx, &inc, yAx, &inc);
  term2 = TSV(NX, NY, xvec);

  return(term1/2+lambda_tsv*term2);
}

void dL_dx(int *M, int *N, double *yAx, double *Amatrix, double *dfdx)
{
  int inc = 1;
  double beta = 1, gamma = 0;

  /* A' (y - A x) */
  dgemv_("T", M, N, &beta, Amatrix, M, yAx, &inc, &gamma, dfdx, &inc);

}

void dF_dx(int *M, int *N, int NX, int NY,
	   double *yAx, double *Amatrix, 
	   double *xvec, double lambda_tsv,
	   double *dFdx)
{
  int inc = 1;
  double alpha = 1, beta = -lambda_tsv;

  d_TSV(NX, NY, xvec, dFdx);

  /* A' (y - A x) -lambda_tsv*d_TSV*/
  
  dgemv_("T", M, N, &alpha, Amatrix, M, yAx, &inc, &beta, dFdx, &inc);

}

int mfista_L1_TSV_core(double *yvec, double *Amat, 
		       int *M, int *N, int NX, int NY,
		       double lambda_l1, double lambda_tsv, double cinit,
		       double *xvec, int nonneg_flag)
{
  void (*soft_th)(double *vector, int length, double eta, double *newvec);
  int i, iter, inc = 1;
  double *xtmp, *xnew, *ytmp, *zvec, *dfdx,
    Qcore, Fval, Qval, c, cinv, tmpa, l1cost, costtmp, *cost,
    mu=1, munew, alpha = 1;

  printf("computing image with MFISTA.\n");

  /* allocate memory space start */ 

  cost  = alloc_vector(MAXITER);
  dfdx  = alloc_vector(*N);
  xnew  = alloc_vector(*N);
  xtmp  = alloc_vector(*N);
  ytmp  = alloc_vector(*M);
  zvec  = alloc_vector(*N);

  /* defining soft_thresholding */
  
  if(nonneg_flag == 0)
    soft_th=soft_threshold;
  else if(nonneg_flag == 1)
    soft_th=soft_threshold_nonneg;
  else {
    printf("nonneg_flag must be chosen properly.\n");
    return(0);
  }

  /* initialize xvec */
  dcopy_(N, xvec, &inc, zvec, &inc);

  c = cinit;

  calc_yAz(M, N, yvec, Amat, xvec, ytmp);

  if( lambda_tsv > 0)
    costtmp = calc_F_TSV_part(M, NX, NY, ytmp, xvec, lambda_tsv);
  else
    costtmp = ddot_(M, ytmp, &inc, ytmp, &inc)/2;
  
  l1cost = dasum_(N, xvec, &inc);
  costtmp += lambda_l1*l1cost;

  for(iter = 0; iter < MAXITER; iter++){

    cost[iter] = costtmp;

    if((iter % 100) == 0)
      printf("%d cost = %f \n",(iter+1), cost[iter]);

    calc_yAz(M, N,  yvec, Amat, zvec, ytmp);

    if(lambda_tsv > 0){
      dF_dx(M, N, NX, NY, ytmp, Amat, zvec, lambda_tsv, dfdx);
      Qcore = calc_F_TSV_part(M, NX, NY, ytmp, zvec, lambda_tsv);
    }
    else{
      dL_dx(M, N, ytmp, Amat, dfdx);
      Qcore = ddot_(M, ytmp, &inc, ytmp, &inc)/2;
    }

    for( i = 0; i < MAXITER; i++){
      dcopy_(N, dfdx, &inc, xtmp, &inc);
      cinv = 1/c;
      dscal_(N, &cinv, xtmp, &inc);
      daxpy_(N, &alpha, zvec, &inc, xtmp, &inc);
      soft_th(xtmp, *N, lambda_l1/c, xnew);

      calc_yAz(M, N, yvec, Amat, xnew, ytmp);

      if(lambda_tsv > 0){
	Fval = calc_F_TSV_part(M, NX, NY, ytmp, xnew, lambda_tsv);
	Qval = calc_Q_part(N, xnew, zvec, c, dfdx, xtmp);
      }
      else{
	Fval = ddot_(M, ytmp, &inc, ytmp, &inc)/2;
	Qval = calc_Q_part(N, xnew, zvec, c, dfdx, xtmp);
      }

      Qval += Qcore;

      if(Fval<=Qval) break;

      c *= ETA;
    }

    c /= ETA;

    munew = (1+sqrt(1+4*mu*mu))/2;

    l1cost = dasum_(N, xnew, &inc);

    Fval += lambda_l1*l1cost;

    if(Fval < cost[iter]){

      costtmp = Fval;
      dcopy_(N, xvec, &inc, zvec, &inc);

      tmpa = (1-mu)/munew;
      dscal_(N, &tmpa, zvec, &inc);

      tmpa = 1+((mu-1)/munew);
      daxpy_(N, &tmpa, xnew, &inc, zvec, &inc);
	
      dcopy_(N, xnew, &inc, xvec, &inc);
	    
    }	
    else{
      dcopy_(N, xvec, &inc, zvec, &inc);

      tmpa = 1-(mu/munew);
      dscal_(N, &tmpa, zvec, &inc);
      
      tmpa = mu/munew;
      daxpy_(N, &tmpa, xnew, &inc, zvec, &inc);

      /* another stopping rule */
      if((iter>1) && (dasum_(N, xvec, &inc) == 0)) break;
    }

    /* stopping rule start */
     
    if((iter>=MINITER) && ((cost[iter-TD]-cost[iter])<EPS)) break;

    /* stopping rule end */

    mu = munew;
  }
  if(iter == MAXITER){
    printf("%d cost = %f \n",(iter), cost[iter-1]);
    iter = iter -1;
  }
  else
    printf("%d cost = %f \n",(iter+1), cost[iter]);

  printf("\n");

  /* clear memory */

  free(cost);
  free(dfdx);
  free(xnew);
  free(xtmp);
  free(ytmp);
  free(zvec);
  return(iter+1);
}

