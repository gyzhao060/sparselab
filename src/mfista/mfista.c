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


void mfista(double* u,double* v,double* x,double* y,
            double* Iin,double* Iout,double* V,double* Vsigma,
            int dftsign,int M,int N,int NX,int NY,
            double cinit,double lambda_l1,double lambda_tv,double lambda_tsv,
            int nonneg_flag,int rec_flag,int looe_flag,int log_flag,
            char *log_fname,struct RESULT *mfista_result)
{
  int inc=1;
  double *xvec,*yvec,*A;
  FILE* log_fid;


  /* size of data and output image */
  printf("M (Number of Data)         is %d\n",M);
  printf("N (Number of Image Pixels) is %d\n",N);


  /* initialize matrix */
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

  if (rec_flag==0) {
    NX = (int)sqrt(N);
  }
  printf("NX is %d\n",NX);
  NY = (int)N/NX;

  if (looe_flag==1)  {
    printf("Approximation of LOOE will be computed.\n\n");
  }
  else  {
    printf("\n");
  }

  if(log_flag==1) {
    printf("Log will be saved to %s.\n",log_fname);
  }


  /* preparation end */
  /* main loop */
  if (lambda_tv<0 && lambda_tsv<0) {
    mfista_L1_core(yvec, A, &M, &N, lambda_l1, cinit, xvec, nonneg_flag, looe_flag, mfista_result);
  }
  else if (lambda_tv>0) {
    if(nonneg_flag==0) {
      mfista_L1_TV_core(yvec, A, &M, &N, NX, NY, lambda_l1, lambda_tv, cinit, xvec, mfista_result);
    }
    else if (nonneg_flag==1) {
      mfista_L1_TV_core_nonneg(yvec, A, &M, &N, NX, NY, lambda_l1, lambda_tv, cinit, xvec, mfista_result);
    }
  }
  else if (lambda_tsv>0) {
    mfista_L1_TSV_core(yvec, A, &M, &N, NX, NY, lambda_l1, lambda_tsv, cinit, xvec, nonneg_flag, looe_flag, mfista_result);
  }


  /* copy xvec to Iout */
  printf("Results copy to Iout.\n");
  dcopy_(&N, xvec, &inc, Iout, &inc);
  /* main loop end */

  /* output log */
  if (log_flag==1) {
    log_fid = fopenw(log_fname);
    show_result(log_fid,log_fname,mfista_result);
    fclose(log_fid);
  }

  /* clear memory */
  free(A);
  free(xvec);
  free(yvec);
}


void calc_A(int M,int N,int dftsign,
             double *u,double *v,double *x,double *y,double *A,double *Verr)
{
  int i,j,k,halfM=M/2,incx=1,incA=1;
  double factor,xfactor,yfactor;

  /* Define a factor for DFT */
  if (dftsign>0) {
    factor = 2*M_PI;
  }
  else {
    factor = -2*M_PI;
  }

  /* Calculate A */
  #ifdef _OPENMP
    printf("  calculated by openmp\n");
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
}
