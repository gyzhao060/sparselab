module static_imaging_lib
  !$use omp_lib
  use nrtype, only : dp, tol
  implicit none
contains


subroutine calc_I2d(Iin,xidx,yidx,I2d,Npix,Nx,Ny)
  implicit none
  !
  integer, intent(in) :: Npix,Nx,Ny
  integer, intent(in) :: xidx(Npix), yidx(Npix)
  real(dp),intent(in) :: Iin(Npix)
  real(dp),intent(inout) :: I2d(Nx,Ny)
  !
  integer :: ipix
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Npix,Iin,xidx,yidx) &
  !$OMP   PRIVATE(ipix) 
  do ipix=1,Npix
    I2d(xidx(ipix),yidx(ipix))=Iin(ipix)
  end do
  !$OMP END PARALLEL DO
end subroutine


real(dp) function tv(I2d,Nx,Ny)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny
  real(dp),intent(in)  :: I2d(Nx,Ny)
  !
  ! variables
  integer :: Nxy
  integer :: i1,j1,i2,j2,ixy
  real(dp):: dIx,dIy
  !
  ! initialize tv term
  tv = 0d0
  Nxy= Nx*Ny
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nxy,Nx,Ny,I2d) &
  !$OMP   PRIVATE(i1,j1,i2,j2,dIx,dIy) &
  !$OMP   REDUCTION(+:tv)
  do ixy=1,Nxy
    call ixy2ixiy(ixy,i1,j1,Nx)
    i2 = i1 + 1             ! i+1
    j2 = j1 + 1             ! j+1
    !
    ! dIx = I(i+1,j) - I(i,j)
    if (i2 > Nx) then
      dIx = 0d0
    else
      dIx = I2d(i2,j1) - I2d(i1,j1)
    end if
    !
    ! dIy = I(i,j+1) - I(i,j)
    if (j2 > Ny) then
      dIy = 0d0
    else
      dIy = I2d(i1,j2) - I2d(i1,j1)
    end if
    !
    tv = tv + sqrt(dIx*dIx+dIy*dIy)
  end do
  !$OMP END PARALLEL DO
end function


real(dp) function tsv(I2d,Nx,Ny)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny
  real(dp),intent(in)  :: I2d(Nx,Ny)
  !
  ! variables
  integer :: Nxy
  integer :: i1,j1,i2,j2,ixy
  real(dp):: dIx,dIy
  !
  ! initialize tsv term
  tsv = 0d0
  Nxy= Nx*Ny
  !
  !$OMP PARALLEL DO DEFAULT(SHARED) &
  !$OMP   FIRSTPRIVATE(Nxy,Nx,Ny,I2d) &
  !$OMP   PRIVATE(i1,j1,i2,j2,dIx,dIy,ixy) &
  !$OMP   REDUCTION(+:tsv)
  do ixy=1,Nxy
    call ixy2ixiy(ixy,i1,j1,Nx)
    i2 = i1 + 1             ! i+1
    j2 = j1 + 1             ! j+1
    !
    ! dIx = I(i+1,j) - I(i,j)
    if (i2 > Nx) then
      dIx = 0d0
    else
      dIx  = I2d(i2,j1) - I2d(i1,j1)
    end if
    !
    ! dIy = I(i,j+1) - I(i,j)
    if (j2 > Ny) then
      dIy = 0d0
    else
      dIy  = I2d(i1,j2) - I2d(i1,j1)
    end if
    !
    tsv = tsv + dIx*dIx+dIy*dIy
  end do
  !$OMP END PARALLEL DO
end function


real(dp) function gradtve(xidx,yidx,I2d,Nx,Ny)
  implicit none
  !
  integer, intent(in) :: Nx,Ny
  integer, intent(in) :: xidx, yidx
  real(dp),intent(in) :: I2d(Nx,Ny)
  !
  ! variables
  integer :: i0,j0,i1,j1,i2,j2
  real(dp):: dIx,dIy,tve
  !
  ! initialize tsv term
  gradtve = 0d0
  !
  ! take indice
  i1 = xidx
  j1 = yidx
  i0 = i1 - 1
  j0 = j1 - 1
  i2 = i1 + 1
  j2 = j1 + 1
  !
  !-------------------------------------
  ! (i2,j1)-(i1,j1), (i1,j2)-(i1,j1)
  !-------------------------------------
  ! dIx = I(i+1,j) - I(i,j)
  if (i2 > Nx) then
    dIx = 0d0
  else
    dIx = I2d(i2,j1) - I2d(i1,j1)
  end if
  !
  ! dIy = I(i,j+1) - I(i,j)
  if (j2 > Ny) then
    dIy = 0d0
  else
    dIy = I2d(i1,j2) - I2d(i1,j1)
  end if
  !
  tve = sqrt(dIx*dIx+dIy*dIy)
  if (tve > tol) then
    gradtve = gradtve - (dIx + dIy)/tve
  end if
  !
  !-------------------------------------
  ! (i1,j1)-(i0,j1), (i0,j2)-(i0,j1)
  !-------------------------------------
  if (i0 > 0) then
    ! dIx = I(i,j) - I(i-1,j)
    dIx = I2d(i1,j1) - I2d(i0,j1)
    
    ! dIy = I(i-1,j+1) - I(i,j)
    if (j2 > Ny) then
      dIy = 0d0
    else
      dIy = I2d(i0,j2) - I2d(i0,j1)
    end if
    
    tve = sqrt(dIx*dIx+dIy*dIy)
    if (tve > tol) then
      gradtve = gradtve + dIx/tve
    end if
  end if
  !
  !-------------------------------------
  ! (i2,j0)-(i1,j0), (i1,j1)-(i1,j0)
  !-------------------------------------
  if (j0 > 0) then
    ! dIy = I(i,j) - I(i,j-1)
    dIy = I2d(i1,j1) - I2d(i1,j0)
    
    ! dIx = I(i+1,j-1) - I(i,j-1)
    if (i2 > Nx) then
      dIx = 0d0
    else
      dIx = I2d(i2,j0) - I2d(i1,j0)
    end if
    
    tve = sqrt(dIx*dIx+dIy*dIy)
    if (tve > tol) then
      gradtve = gradtve + dIy/tve
    end if
  end if
  !
end function


real(dp) function gradtsve(xidx,yidx,I2d,Nx,Ny)
  implicit none
  !
  integer, intent(in)  :: Nx,Ny
  integer, intent(in)  :: xidx, yidx
  real(dp),intent(in)  :: I2d(Nx,Ny)
  !
  ! variables
  integer :: i0,j0,i1,j1,i2,j2
  !
  ! initialize tsv term
  gradtsve = 0d0
  !
  ! take indice
  i1 = xidx
  j1 = yidx
  i0 = i1 - 1
  j0 = j1 - 1
  i2 = i1 + 1
  j2 = j1 + 1
  !
  ! dIx = I(i+1,j) - I(i,j)
  if (i2 <= Nx) then
    gradtsve = gradtsve - 2*(I2d(i2,j1) - I2d(i1,j1))
  end if
  !
  ! dIy = I(i,j+1) - I(i,j)
  if (j2 <= Ny) then
    gradtsve = gradtsve - 2*(I2d(i1,j2) - I2d(i1,j1))
  end if
  !
  if (i0 > 0) then
    gradtsve = gradtsve + 2*(I2d(i1,j1) - I2d(i0,j1))
  end if
  !
  if (j0 > 0) then
    gradtsve = gradtsve + 2*(I2d(i1,j1) - I2d(i1,j0))
  end if
  !
end function


subroutine ixy2ixiy(ixy,ix,iy,Nx)
  implicit none
  
  ! arguments
  integer, intent(in):: ixy,Nx
  integer, intent(out):: ix,iy
  !
  ix = mod(ixy-1,Nx)+1
  iy = (ixy-1)/Nx+1
end subroutine


subroutine ixiy2ixy(ix,iy,ixy,Nx)
  implicit none
  
  ! arguments
  integer, intent(in):: ix,iy,Nx
  integer, intent(out):: ixy
  !
  ixy = ix + (iy-1) * Nx
end subroutine


end module
