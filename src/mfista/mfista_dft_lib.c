/*
   Copyright (C) 2015   Shiro Ikeda <shiro@ism.ac.jp>

   This is file 'mfista.c'. An optimization algorithm for imaging of
   interferometry. The idea of the algorithm was from the following
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

void mfista_core(
			 double *y, double *A,
			 int *M, int *N, int NX, int NY,
			 double lambda_l1, double lambda_tv, double lambda_tsv,
			 double cinit, double *xinit, double *xout,
			 int nonneg_flag, int looe_flag,
			 struct RESULT *mfista_result)
{
  double s_t, e_t;
  int    iter = 0, inc = 1;
  struct timespec time_spec1, time_spec2;

  dcopy_(N, xinit, &inc, xout, &inc);

  get_current_time(&time_spec1);

  /* main loop */

  if( lambda_tv <= 0){
    iter = mfista_L1_TSV_core(y, A, M, N, NX, NY,
			      lambda_l1, lambda_tsv, cinit, xout, nonneg_flag);
  }
  else if( lambda_tv > 0  && lambda_tsv <= 0 ){
    iter = mfista_L1_TV_core(y, A, M, N, NX, NY,
			     lambda_l1, lambda_tv, cinit, xout, nonneg_flag);
  }
  else{

			printf("You cannot set both of lambda_TV and lambda_TSV positive. %f %f\n", lambda_tv, lambda_tsv);
    return;
  }

  get_current_time(&time_spec2);

  /* main loop end */

  s_t = (double)time_spec1.tv_sec + (10e-10)*(double)time_spec1.tv_nsec;
  e_t = (double)time_spec2.tv_sec + (10e-10)*(double)time_spec2.tv_nsec;

  mfista_result->comp_time = e_t-s_t;
  mfista_result->ITER = iter;
  mfista_result->nonneg = nonneg_flag;

  calc_result(y, A, M, N, NX, NY,
	      lambda_l1, lambda_tv, lambda_tsv, xout, nonneg_flag, looe_flag,
	      mfista_result);

  return;
}

void mfista_imaging(
			 double *y,
			 int M, int NX, int NY,
			 int dftsign, double *u,double *v,double *xpix,double *ypix, double *yerr,
			 double lambda_l1, double lambda_tv, double lambda_tsv,
			 double cinit, double *xinit, double *xout,
			 int nonneg_flag, int looe_flag,
			 struct RESULT *mfista_result)
{
  int N=NX*NY; /* Total number of Imaging Pixels */
	double *A;

  /* initialize matrix */
  printf("Initialize A matrix\n");
  A = alloc_matrix(M, N);
  clear_matrix(A, M, N);
  calc_A_imaging(M, N, dftsign, u, v, xpix, ypix, A, yerr);

  /* initialize xvec and yvec */
  printf("Run mfista_core\n");
	mfista_core(y, A, &M, &N, NX, NY,
				 lambda_l1, lambda_tv, lambda_tsv,
				 cinit, xinit, xout,
				 nonneg_flag, looe_flag,
				 mfista_result);

  /* clear memory */
  free(A);
}

void mfista_imaging_results(
			 double *y,
			 int M, int NX, int NY,
			 int dftsign, double *u,double *v,double *xpix,double *ypix, double *yerr,
			 double lambda_l1, double lambda_tv, double lambda_tsv,
			 double cinit, double *xinit,
			 int nonneg_flag, int looe_flag,
			 struct RESULT *mfista_result)
{
  int N=NX*NY; /* Total number of Imaging Pixels */
  double *A;

  /* initialize matrix */
  printf("Initialize A matrix\n");
  A = alloc_matrix(M, N);
  clear_matrix(A, M, N);
  calc_A_imaging(M, N, dftsign, u, v, xpix, ypix, A, yerr);

	if( lambda_tv > 0 && lambda_tsv > 0){
		printf("You cannot set both of lambda_TV and lambda_TSV positive. %f %f\n", lambda_tv, lambda_tsv);
		return;
	}
  /* initialize xvec and yvec */
  mfista_result->comp_time = 0e0;
  mfista_result->ITER = 0;
  mfista_result->nonneg = nonneg_flag;

  calc_result(y, A, &M, &N, NX, NY,
	      lambda_l1, lambda_tv, lambda_tsv, xinit, nonneg_flag, looe_flag,
	      mfista_result);

  /* clear memory */
  free(A);
}

void calc_A_imaging(
  int M,int N,int dftsign,
  double *u,double *v,double *xpix,double *ypix,double *A,double *yerr)
{
  int i,j,k;            /* loop variables */
  int halfM=M/2;        /* actual data number */
  int incx=1,incA=1;    /* increment of each vector */

  double factor,xfactor,yfactor;

  /* Define a factor for DFT */
  if (dftsign>0) {
    factor = 2*M_PI;
  }
  else {
    factor = -2*M_PI;
  }

  /* Calculate Phase */
  #ifdef _OPENMP
   #pragma omp parallel for default(shared)\
    private(j,xfactor,yfactor)\
    firstprivate(factor,halfM,M,N,incx,incA)
  #endif
  for (i=0; i<N; ++i) {
    j = i*M;

    xfactor = factor*xpix[i];
    daxpy_(&halfM, &xfactor, u, &incx, &A[j], &incA);
    yfactor = factor*ypix[i];
    daxpy_(&halfM, &yfactor, v, &incx, &A[j], &incA);
  }

  /* Calculate Fourier Matrix */
  #ifdef _OPENMP
   #pragma omp parallel for default(shared)\
    private(j,k)\
    firstprivate(yerr,halfM,M,N)
  #endif
  for (i=0; i<N; ++i) {
    for (j=0; j<halfM; ++j) {
      k = i*M;
      A[k+j+halfM] = sin(A[k+j])/yerr[j]*2/sqrt(halfM);
      A[k+j] = cos(A[k+j])/yerr[j]*2/sqrt(halfM);
    }
  }
}

void mfista_ft(
			 double *y,
		   int M,int N,double *lambsq,double *phi,double *yerr,
			 double lambda_l1, double lambda_tv, double lambda_tsv,
			 double cinit, double *xinit, double *xout,
			 int nonneg_flag, int looe_flag,
			 struct RESULT *mfista_result)
{
	double *A;

  /* initialize matrix */
  printf("Initialize A matrix\n");
  A = alloc_matrix(M, N);
  clear_matrix(A, M, N);
  calc_A_ft(M,N,lambsq,phi,A,yerr);

  /* initialize xvec and yvec */
  printf("Run mfista_core\n");
	mfista_core(y, A, &M, &N, 1, N,
				 lambda_l1, lambda_tv, lambda_tsv,
				 cinit, xinit, xout,
				 nonneg_flag, looe_flag,
				 mfista_result);

  /* clear memory */
  free(A);
}

void mfista_ft_results(
			 double *y,
		   int M,int N,double *lambsq,double *phi,double *yerr,
			 double lambda_l1, double lambda_tv, double lambda_tsv,
			 double cinit, double *xinit,
			 int nonneg_flag, int looe_flag,
			 struct RESULT *mfista_result)
{
	double *A;

  /* initialize matrix */
  printf("Initialize A matrix\n");
  A = alloc_matrix(M, N);
  clear_matrix(A, M, N);
  calc_A_ft(M,N,lambsq,phi,A,yerr);


	if( lambda_tv > 0 && lambda_tsv > 0){
		printf("You cannot set both of lambda_TV and lambda_TSV positive.\n");
		return;
	}

  /* initialize xvec and yvec */
  mfista_result->comp_time = 0e0;
  mfista_result->ITER = 0;
  mfista_result->nonneg = nonneg_flag;

  calc_result(y, A, &M, &N, 1, N,
	      lambda_l1, lambda_tv, lambda_tsv, xinit, nonneg_flag, looe_flag,
	      mfista_result);

  /* clear memory */
  free(A);
}

void calc_A_ft(
  int M,int N,
  double *lambsq,double *phi,double *A,double *yerr)
{
  int i,j,k;            /* loop variables */
  int halfM=M/2;        /* actual data number */
  int halfN=N/2;        /* actual FDF pixel number */

  double factor1, factor2, factor3;

  /* Calculate Fourier Matrix */
  #ifdef _OPENMP
   #pragma omp parallel for default(shared)\
    private(j,k,factor1,factor2,factor3)\
    firstprivate(yerr,halfM,halfN,M,N)
  #endif
	for (i=0; i<halfN; ++i) {
	  for (j=0; j<halfM; ++j) {
			factor1 = 2*phi[i]*lambsq[j];
			factor2 = cos(factor1)/yerr[j]*2/sqrt(halfM);
			factor3 = sin(factor1)/yerr[j]*2/sqrt(halfM);

			k = i*M;
      A[k+j] = factor2; /* cos */
			A[k+j+halfM] = factor3; /* sin */

			k = (i+halfN)*M;
      A[k+j] = -factor3; /* -sin */
      A[k+j+halfM] = factor2; /* cos */
    }
  }
}
