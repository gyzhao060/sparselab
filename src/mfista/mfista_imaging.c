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


void mfista_imaging(
  double* Iin, double* Iout, double* x,double* y, int NX, int NY,
  double* u, double* v, double* V,double* Vsigma, int M, int dftsign,
  double lambda_l1, double lambda_tv, double lambda_tsv,
  int nonneg_flag, int looe_flag, double cinit,
  struct RESULT *mfista_result)
{
  int inc=1;
  int N=NX*NY; /* Total number of Imaging Pixels */
  double *xvec,*yvec,*A;


  /* size of data and output image */
  printf("Number of Data        : %d\n",M/2);
  printf("Number of Image Pixels: %d = %d x %d\n",N,NX,NY);


  /* initialize matrix */
  printf("Initialize A matrix\n");
  A = alloc_matrix(M, N);
  clear_matrix(A, M, N);
  calc_A(M, N, dftsign, u, v, x, y, A, Vsigma);


  /* initialize xvec and yvec */
  printf("Initialize xvec and yvec\n");
  xvec = alloc_vector(N);
  yvec = alloc_vector(M);
  dcopy_(&N, Iin, &inc, xvec, &inc);
  dcopy_(&M, V, &inc, yvec, &inc);

  printf("lambda_l1 = %g\n",lambda_l1);
  if (lambda_tv>0) {
    printf("lambda_tv = %g\n",lambda_tv);
  }
  else if (lambda_tsv>0) {
    printf("lambda_tsv = %g\n",lambda_tsv);
  }
  printf("c = %g\n",cinit);


  /* processs flags */
  if (nonneg_flag==1) {
    printf("x is nonnegative.\n");
  }
  mfista_result->nonneg = nonneg_flag;


  if (looe_flag==1)  {
    printf("Approximation of LOOE will be computed.\n\n");
  }
  else  {
    printf("\n");
  }


  /* run mfista core-routine */
  if (lambda_tv<0 && lambda_tsv<0) {
    mfista_L1_core(yvec, A, &M, &N, lambda_l1, cinit,
                   xvec, nonneg_flag, looe_flag, mfista_result);
  }
  else if (lambda_tv>0) {
    if(nonneg_flag==0) {
      mfista_L1_TV_core(yvec, A, &M, &N, NX, NY, lambda_l1, lambda_tv, cinit,
                        xvec, mfista_result);
    }
    else if (nonneg_flag==1) {
      mfista_L1_TV_core_nonneg(yvec, A, &M, &N, NX, NY, lambda_l1, lambda_tv,
                               cinit, xvec, mfista_result);
    }
  }
  else if (lambda_tsv>0) {
    mfista_L1_TSV_core(yvec, A, &M, &N, NX, NY, lambda_l1, lambda_tsv, cinit,
                       xvec, nonneg_flag, looe_flag, mfista_result);
  }


  /* copy xvec to Iout */
  dcopy_(&N, xvec, &inc, Iout, &inc);

  /* clear memory */
  free(A);
  free(xvec);
  free(yvec);
}


void calc_A(int M,int N,int dftsign,
             double *u,double *v,double *x,double *y,double *A,double *Verr)
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

    xfactor = factor*x[i];
    daxpy_(&halfM, &xfactor, u, &incx, &A[j], &incA);
    yfactor = factor*y[i];
    daxpy_(&halfM, &yfactor, v, &incx, &A[j], &incA);
  }

  /* Calculate Fourier Matrix */
  #ifdef _OPENMP
   #pragma omp parallel for default(shared)\
    private(j,k)\
    firstprivate(Verr,halfM,M,N)
  #endif
  for (i=0; i<N; ++i) {
    for (j=0; j<halfM; ++j) {
      k = i*M;
      A[k+j+halfM] = sin(A[k+j])/Verr[j];
      A[k+j] = cos(A[k+j])/Verr[j];
    }
  }
}
