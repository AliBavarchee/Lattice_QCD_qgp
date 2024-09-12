module lattice_qcd
  implicit none

  ! Define parameters
  integer, parameter :: dim = 4       ! Number of space-time dimensions
  integer, parameter :: nx = 10       ! Lattice size in x-direction
  integer, parameter :: ny = 10       ! Lattice size in y-direction
  integer, parameter :: nz = 10       ! Lattice size in z-direction
  integer, parameter :: nt = 10       ! Lattice size in t-direction
  integer, parameter :: Nc = 3        ! Number of colors (SU(3) gauge theory)
  real(8), parameter :: beta = 6.0    ! Gauge coupling

  ! Declare lattice arrays
  complex(8), dimension(nx, ny, nz, nt, Nc, Nc, dim) :: U  ! Gauge fields (SU(3) matrices)
  complex(8), dimension(nx, ny, nz, nt, Nc) :: psi          ! Fermion fields (quark field)

contains

  ! Subroutine to initialize gauge fields to identity matrices
  subroutine initialize_gauge_fields()
    integer :: i, j, x, y, z, t, mu
    do x = 1, nx
      do y = 1, ny
        do z = 1, nz
          do t = 1, nt
            do mu = 1, dim
              U(x, y, z, t, :, :, mu) = cmplx(0.0, 0.0)
              do i = 1, Nc
                U(x, y, z, t, i, i, mu) = cmplx(1.0, 0.0) ! Set diagonal elements to 1.0 (identity)
              end do
            end do
          end do
        end do
      end do
    end do
    ! Debug output to verify initialization
    print *, 'Initialized gauge fields:'
    x = 1; y = 1; z = 1; t = 1; mu = 1
    do i = 1, Nc
      do j = 1, Nc
        print *, 'U(', i, ',', j, ') = ', U(x, y, z, t, i, j, mu)
      end do
    end do
  end subroutine initialize_gauge_fields

  ! Subroutine to initialize fermion fields to random values
  subroutine initialize_fermion_fields()
    integer :: x, y, z, t, c
    real(8) :: rand_real
    call random_seed()  ! Initialize random number generator
    do x = 1, nx
      do y = 1, ny
        do z = 1, nz
          do t = 1, nt
            do c = 1, Nc
              call random_number(rand_real)
              psi(x, y, z, t, c) = cmplx(rand_real, rand_real) ! Initialize fermion fields with random complex values
            end do
          end do
        end do
      end do
    end do
  end subroutine initialize_fermion_fields

  ! Subroutine to update gauge fields using a basic Wilson gauge action
  subroutine update_gauge_fields()
    integer :: x, y, z, t, mu, nu, i, j
    complex(8), dimension(Nc, Nc) :: staple, delta_U

    ! Loop over all lattice sites
    do x = 1, nx
      do y = 1, ny
        do z = 1, nz
          do t = 1, nt
            do mu = 1, dim
              ! Calculate staple term (simplified for demonstration)
              staple = cmplx(0.0, 0.0)
              do nu = 1, dim
                if (nu /= mu) then
                  staple = staple + matmul(U(x, y, z, t, :, :, nu), transpose(conjg(U(x, y, z, t, :, :, nu))))
                end if
              end do
              ! Update gauge fields (simplified update rule)
              delta_U = beta * staple
              U(x, y, z, t, :, :, mu) = U(x, y, z, t, :, :, mu) + delta_U

              ! Debug output to verify updates
              print *, 'Updated gauge field U at (', x, ',', y, ',', z, ',', t, ') and direction mu =', mu
              do i = 1, Nc
                do j = 1, Nc
                  print *, 'U(', i, ',', j, ') = ', U(x, y, z, t, i, j, mu)
                end do
              end do
            end do
          end do
        end do
      end do
    end do
  end subroutine update_gauge_fields

  ! Subroutine to save gauge and fermionic fields to files
  subroutine save_fields()
    integer :: x, y, z, t, mu, i, j, c
    character(len=100) :: gauge_filename, fermion_filename

    gauge_filename = 'gauge_fields.txt'
    fermion_filename = 'fermion_fields.txt'

    ! Open files to write
    open(unit=10, file=gauge_filename, status='unknown')
    open(unit=11, file=fermion_filename, status='unknown')

    ! Write the gauge field data to file
    do x = 1, nx
      do y = 1, ny
        do z = 1, nz
          do t = 1, nt
            do mu = 1, dim
              write(10, '(A, I3, A, I3, A, I3, A, I3, A)') 'x=', x, ' y=', y, ' z=', z, ' t=', t, ' mu=', mu
              do i = 1, Nc
                do j = 1, Nc
                  ! Write the real and imaginary parts of the complex number
                  write(10, '(A, I3, A, I3, A, 2F10.5)') 'U(', i, ',', j, ') =', &
                     real(U(x, y, z, t, i, j, mu)), aimag(U(x, y, z, t, i, j, mu))
                end do
              end do
            end do
          end do
        end do
      end do
    end do
    close(10)

    print *, 'Gauge fields saved to ', gauge_filename

    ! Write the fermion field data to file
    do x = 1, nx
      do y = 1, ny
        do z = 1, nz
          do t = 1, nt
            write(11, '(A,I3,A,I3,A,I3,A,I3)') 'x=', x, ' y=', y, ' z=', z, ' t=', t
            do c = 1, Nc
              write(11, '(A, I3, A, 2F10.5)') 'psi(', c, ') =', real(psi(x, y, z, t, c)), aimag(psi(x, y, z, t, c))
            end do
          end do
        end do
      end do
    end do
    close(11)

    print *, 'Fermion fields saved to ', fermion_filename
  end subroutine save_fields

end module lattice_qcd
