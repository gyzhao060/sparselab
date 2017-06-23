module blas
  implicit none

  interface
    real(kind(1.0d0)) function dasum(n, x, incx)
      integer,            intent(in) :: n, incx
      real(kind(1.0d0)),  intent(in) :: x(n)
    end function
  end interface

  interface
    real(kind(1.0d0)) function ddot(n, x, incx, y, incy)
      integer,            intent(in) :: n, incx, incy
      real(kind(1.0d0)),  intent(in) :: x(n), y(n)
    end function
  end interface
end module
