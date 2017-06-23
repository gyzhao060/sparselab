module phasecon_lib
  !$use omp_lib
  use param, only: dp, pi
  implicit none
contains
!
!  subroutines and functions
!
subroutine pc_car2d(x,y,u,v,Vreal,Vimag,weight,&
                    PC1,PC2,PC3,PC4,dmap, &
                    Nxy,Nuv)
  !
  implicit none
  !
  integer,  intent(in) :: Nuv, Nxy
  real(dp), intent(in) :: x(Nxy), y(Nxy)
  real(dp), intent(in) :: u(Nuv), v(Nuv), Vreal(Nuv), Vimag(Nuv), weight(Nuv)
  real(dp), intent(out) :: PC1(Nxy),PC2(Nxy),PC3(Nxy),PC4(Nxy),dmap(Nxy)

  integer :: ixy,iuv
  real(dp) :: wasum,wacsum1,wacsum2,wassum1,wassum2
  real(dp) :: wsum,wcsum1,wcsum2,wssum1,wssum2
  real(dp) :: A(1:Nuv), phi(1:Nuv)
  real(dp) :: barphi1, barphi2

  !$OMP PARALLEL DO DEFAULT(SHARED)&
  !$OMP   FIRSTPRIVATE(x,y,u,v,Vreal,Vimag,weight,Nxy,Nuv) &
  !$OMP   PRIVATE(ixy,iuv,wasum,wacsum1,wacsum2,wassum1,wassum2, &
  !$OMP           wsum,wcsum1,wcsum2,wssum1,wssum2,A,phi,barphi1,barphi2)
  do ixy=1,Nxy
    ! using full complex visibilities
    wasum=0d0
    wacsum1=0d0
    wacsum2=0d0
    wassum1=0d0
    wassum2=0d0
    dmap(ixy)=0
    ! without amplitude
    wsum=0d0
    wcsum1=0d0
    wcsum2=0d0
    wssum1=0d0
    wssum2=0d0
    do iuv=1,Nuv
      A(iuv)=sqrt(Vreal(iuv)*Vreal(iuv)+Vimag(iuv)*Vimag(iuv))
      phi(iuv)=atan2(Vimag(iuv),Vreal(iuv))-2*pi*(u(iuv)*x(ixy)+v(iuv)*y(ixy))
      ! with amplitude
      wasum   = wasum   + weight(iuv)*A(iuv)
      wacsum1 = wacsum1 + weight(iuv)*A(iuv)*cos(phi(iuv))
      wassum1 = wassum1 + weight(iuv)*A(iuv)*sin(phi(iuv))
      ! without amplitude
      wsum   = wsum   + weight(iuv)
      wcsum1 = wcsum1 + weight(iuv)*cos(phi(iuv))
      wssum1 = wssum1 + weight(iuv)*sin(phi(iuv))
      dmap(ixy) = dmap(ixy) + A(iuv)*cos(phi(iuv))
    end do

    barphi1 = atan2(wassum1,wacsum1) ! with amplitude
    barphi2 = atan2(wssum1, wcsum1)  ! without amplitude
    do iuv=1,Nuv
      wacsum2 = wacsum2 +     weight(iuv)*A(iuv)*cos(phi(iuv)-barphi1)
      wassum2 = wassum2 + abs(weight(iuv)*A(iuv)*sin(phi(iuv)-barphi1))
      ! without amplitude
      wcsum2 = wcsum2 +     weight(iuv)*cos(phi(iuv)-barphi2)
      wssum2 = wssum2 + abs(weight(iuv)*sin(phi(iuv)-barphi2))
    end do

    PC1(ixy)=wacsum2/wasum
    PC2(ixy)=(wacsum2-wassum2)/wasum
    PC3(ixy)=wcsum2/wsum
    PC4(ixy)=(wcsum2-wssum2)/wsum
  end do
  !$OMP END PARALLEL DO
end subroutine
end module
