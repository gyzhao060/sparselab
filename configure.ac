#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
AC_INIT([params.f90], [1], [kakiyama@mit.edu])
AC_CONFIG_SRCDIR([./src/futils])

AC_PREFIX_PROGRAM([gfortran])
AC_LANG([C])
AC_LANG([Fortran])

# debug options
AC_ARG_ENABLE([dbg],
              [AC_HELP_STRING([--enable-dbg],
                              [Compile sources with DEBUG options])],
              [dbg_flag="True"],
              [dbg_flag="False"])
AC_SUBST([dbg_flag])

# gcc
AC_PROG_CC

# gfortran
AC_PROG_FC([gfortran], [95])

# f2py
AC_CHECK_PROGS([F2PY],
               [f2py],
               [AC_MSG_ERROR([f2py was not found.])])

# blas
AC_CHECK_LIB(openblas, dgemm,
             [blaslib="-lopenblas"],
             [AC_CHECK_LIB(blas, dgemm,
                           [blaslib="-lblas"],
                           [AC_MSG_ERROR([blas was not found.])]
             )]
)
AC_SUBST([blaslib])

# lapack
AC_CHECK_LIB(lapack, dgesv,
             [lapacklib="-llapack"],
             [AC_MSG_ERROR([lapack was not found.])]
)
AC_SUBST([lapacklib])

# Makefiles
AC_CONFIG_FILES([./makefile])
AC_CONFIG_FILES([./src/lbfgsb/makefile])
AC_CONFIG_FILES([./src/slsqp/makefile])
AC_CONFIG_FILES([./src/mfista/makefile])
AC_CONFIG_FILES([./src/futils/makefile])
AC_OUTPUT
