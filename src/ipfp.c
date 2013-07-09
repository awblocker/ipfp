/*
 * Author: Alex Blocker
 * Email: ablocker@fas.harvard.edu
 * Description: Runs IPF for A x = y via BLAS interface
 * 		R wrapper in ipf.R
 * 		Compile with R CMD SHLIB ipf.c
 */

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <R_ext/BLAS.h>

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>

/* Function to calculate scalar product z = x * y */
void vecProduct(int n, double * x, int incx, double * y, int incy,
		double * z, int incz)
{
	int i = 0;
	for (i=0; i<n; i++)
	{
		z[i*incz] = x[i*incx]*y[i*incy];
	}
}

/* Function to run IPFP procedure, bringing x0 into agreement with A x = y */
SEXP ipfp (SEXP y, SEXP A, SEXP dims, SEXP x,
        SEXP tol, SEXP maxit, SEXP verbose)
{
	// Set number of entries in return list
	const int nReturn = 3;

	// Setup BLAS convenience variables
	const int incx = 1;
	const char notrans = 'n';
	double alpha;
	double beta;

    // Get dimensions of A and y
	int nProtected=0;
    int nrow, ncol;
    nrow = INTEGER(dims)[0];
    ncol = INTEGER(dims)[1];

    // Copy x to xx; will return latter, former will not be affected
    SEXP xx;
    PROTECT( xx = allocVector(REALSXP, ncol) );
    ++nProtected;

    dcopy_(&ncol, &REAL(x)[0], &incx, &REAL(xx)[0], &incx);

    // Setup vector for x[ A[j,] ]
    double * xAj;
    xAj = (double *) R_alloc(ncol, sizeof(double));

    // Setup vector for err
    double * errVec;
    errVec = (double *) R_alloc(nrow, sizeof(double));

    // Setup counters and scale factor
    int j;
    int iter = 0, converged = 0;
    double scale;
    double errNorm;

    // Outer loop for IPFP iterations
    for (iter=0; iter < INTEGER(maxit)[0]; iter++ )
    {
    	// Inner loop for row updates
    	for (j=0; j < nrow; j++)
    	{
			// scale = y[j] / x %*% A[j,]
    		scale = ddot_(&ncol, &REAL(xx)[0], &incx, &REAL(A)[j], &nrow);
    		scale = REAL(y)[j] / scale;
    		scale = scale - 1;

    		// x[ A[j,] ] = x * A[j,]
    		vecProduct(ncol, &REAL(xx)[0], incx, &REAL(A)[j], nrow, xAj, incx);

    		// x[ A[j,] ] = x[ A[j,] ] * scale
    		daxpy_(&ncol, &scale, xAj, &incx, &REAL(xx)[0], &incx);
    	}

    	// Calculate err = A x - y
    	dcopy_(&nrow, &REAL(y)[0], &incx, errVec, &incx);
    	alpha = 1;
    	beta = -1;
    	dgemv_(&notrans, &nrow, &ncol, &alpha, &REAL(A)[0], &nrow,
    			&REAL(xx)[0], &incx, &beta, errVec, &incx);

    	// Calculate L2 norm of err
    	errNorm = dnrm2_(&nrow, errVec, &incx);

    	if(LOGICAL(verbose)[0])
    	{
    		Rprintf("iteration %d:\t%g\n", iter, errNorm);
    	}

    	// Check for convergence
    	if ( errNorm < REAL(tol)[0] )
    	{
    		converged = 1;
    		break;
    	}
    }

    // Setup names for return list
    SEXP names;
    PROTECT( names = allocVector(STRSXP, nReturn) );
    ++nProtected;

    SET_STRING_ELT(names, 0, mkChar("x"));
    SET_STRING_ELT(names, 1, mkChar("iter"));
    SET_STRING_ELT(names, 2, mkChar("errNorm"));

    // Setup return list
    SEXP iterSEXP, errNormSEXP;
    PROTECT( iterSEXP = allocVector(INTSXP, 1) );
    ++nProtected;
    PROTECT( errNormSEXP = allocVector(REALSXP, 1) );
	++nProtected;

	INTEGER(iterSEXP)[0] = iter;

	REAL(errNormSEXP)[0] = errNorm;

    SEXP ans;
    PROTECT( ans = allocVector(VECSXP, nReturn) );
    ++nProtected;

    SET_VECTOR_ELT(ans, 0, xx);
    SET_VECTOR_ELT(ans, 1, iterSEXP);
    SET_VECTOR_ELT(ans, 2, errNormSEXP);

    setAttrib(ans, R_NamesSymbol, names);

    // Unprotect memory
    UNPROTECT(nProtected);

    // Return list
    return(ans);
}
