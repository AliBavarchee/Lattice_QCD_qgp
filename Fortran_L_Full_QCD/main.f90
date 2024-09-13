program main
  use lattice_qcd
  implicit none

  integer :: x, y, z, t, mu, i, j

  ! Initialize fields
  call initialize_gauge_fields()
  call initialize_fermion_fields()
  call initialize_momenta()  ! Make sure momenta are initialized as well

  ! Perform lattice updates
  call hmc()  ! Use the HMC algorithm for updates

  ! Print some gauge field values to the console for inspection
  x = 10
  y = 10
  z = 10
  t = 10
  mu = 4  ! Ensure mu is within the valid range [1, dim]

  print *, 'Gauge field U at (x, y, z, t) = (', x, ',', y, ',', z, ',', t, ') and direction mu =', mu
  do i = 1, Nc
    do j = 1, Nc
      print *, 'U(', i, ',', j, ') = ', real(U(x, y, z, t, i, j, mu)), aimag(U(x, y, z, t, i, j, mu))
    end do
  end do

  ! Save gauge and fermionic fields to files
  call save_fields()

  print *, 'Lattice QCD simulation initialized, first update step performed, and results saved.'

end program main
