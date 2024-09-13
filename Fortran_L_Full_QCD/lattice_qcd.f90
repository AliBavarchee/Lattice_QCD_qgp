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
  real(8), parameter :: epsilon = 0.01 ! Step size for leapfrog integration
  integer, parameter :: Nsteps = 10   ! Number of leapfrog steps

  ! Declare lattice arrays
  complex(8), dimension(nx, ny, nz, nt, Nc, Nc, dim) :: U  ! Gauge fields (SU(3) matrices)
  complex(8), dimension(nx, ny, nz, nt, Nc) :: psi          ! Fermion fields (quark field)
  complex(8), dimension(nx, ny, nz, nt, Nc, Nc, dim) :: P   ! Conjugate momentum for gauge fields

contains

  function matrix_trace(mat) result(tr)
      complex(8), dimension(:,:), intent(in) :: mat
      integer :: i, size1, size2
      complex(8) :: tr

      size1 = size(mat, 1)
      size2 = size(mat, 2)
      tr = (0.0, 0.0)

      if (size1 /= size2) then
          print *, "Error: Matrix is not square."
          return
      end if

      do i = 1, size1
          tr = tr + mat(i, i)
      end do
  end function matrix_trace

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
    print *, 'Initialized gauge fields.'
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

  ! Subroutine to initialize conjugate momentum fields to random values
  subroutine initialize_momenta()
    integer :: x, y, z, t, mu, i, j
    real(8) :: rand_real
    call random_seed()  ! Initialize random number generator
    do x = 1, nx
      do y = 1, ny
        do z = 1, nz
          do t = 1, nt
            do mu = 1, dim
              do i = 1, Nc
                do j = 1, Nc
                  call random_number(rand_real)
                  P(x, y, z, t, i, j, mu) = cmplx(rand_real, rand_real) ! Random momentum values
                end do
              end do
            end do
          end do
        end do
      end do
    end do
  end subroutine initialize_momenta

  ! Subroutine to compute the Hamiltonian
  real(8) function compute_hamiltonian()
    integer :: x, y, z, t, mu, nu, i, j
    complex(8), dimension(Nc, Nc) :: staple
    real(8) :: kinetic_energy, potential_energy
    complex(8) :: momentum_update

    kinetic_energy = 0.0
    potential_energy = 0.0

    ! Compute kinetic energy from conjugate momenta
    do x = 1, nx
      do y = 1, ny
        do z = 1, nz
          do t = 1, nt
            do mu = 1, dim
              do i = 1, Nc
                do j = 1, Nc
                  kinetic_energy = kinetic_energy + real(P(x, y, z, t, i, j, mu) * conjg(P(x, y, z, t, i, j, mu)))
                end do
              end do
            end do
          end do
        end do
      end do
    end do

    ! Compute potential energy using Wilson gauge action
    do x = 1, nx
      do y = 1, ny
        do z = 1, nz
          do t = 1, nt
            do mu = 1, dim
              staple = cmplx(0.0, 0.0)
              do nu = 1, dim
                if (nu /= mu) then
                  staple = staple + matmul(U(x, y, z, t, :, :, nu), transpose(conjg(U(x, y, z, t, :, :, nu))))
                end if
              end do
              potential_energy = potential_energy + real(matrix_trace(matmul(staple, transpose(conjg(U(x, y, z, t, :, :, mu))))))
            end do
          end do
        end do
      end do
    end do

    compute_hamiltonian = kinetic_energy + beta * potential_energy
  end function compute_hamiltonian

  ! Subroutine to perform leapfrog integration
  subroutine leapfrog()
    integer :: step, x, y, z, t, mu, i, j
    complex(8), dimension(Nc, Nc) :: staple
    complex(8) :: momentum_update

    ! Half step for momentum
    do x = 1, nx
      do y = 1, ny
        do z = 1, nz
          do t = 1, nt
            do mu = 1, dim
              staple = cmplx(0.0, 0.0)
              do i = 1, Nc
                do j = 1, Nc
                  momentum_update = epsilon * beta * staple(i, j)
                  P(x, y, z, t, i, j, mu) = P(x, y, z, t, i, j, mu) - momentum_update
                end do
              end do
            end do
          end do
        end do
      end do
    end do

    ! Full step for gauge fields
    do step = 1, Nsteps
      do x = 1, nx
        do y = 1, ny
          do z = 1, nz
            do t = 1, nt
              do mu = 1, dim
                do i = 1, Nc
                  do j = 1, Nc
                    U(x, y, z, t, i, j, mu) = U(x, y, z, t, i, j, mu) + epsilon * P(x, y, z, t, i, j, mu)
                  end do
                end do
              end do
            end do
          end do
        end do
      end do
    end do

    ! Another half step for momentum
    do x = 1, nx
      do y = 1, ny
        do z = 1, nz
          do t = 1, nt
            do mu = 1, dim
              staple = cmplx(0.0, 0.0)
              do i = 1, Nc
                do j = 1, Nc
                  momentum_update = epsilon * beta * staple(i, j)
                  P(x, y, z, t, i, j, mu) = P(x, y, z, t, i, j, mu) - momentum_update
                end do
              end do
            end do
          end do
        end do
      end do
    end do
  end subroutine leapfrog

  ! Subroutine to perform HMC algorithm
  subroutine hmc()
    real(8) :: H_initial, H_final, delta_H
    real(8) :: acceptance_probability
    logical :: accept

    ! Initialize gauge fields and momenta
    call initialize_gauge_fields()
    call initialize_momenta()

    ! Compute initial Hamiltonian
    H_initial = compute_hamiltonian()

    ! Perform leapfrog integration
    call leapfrog()

    ! Compute final Hamiltonian
    H_final = compute_hamiltonian()
    delta_H = H_final - H_initial

    ! Metropolis acceptance step
    acceptance_probability = exp(-delta_H)
    call random_number(acceptance_probability)
    accept = (acceptance_probability > acceptance_probability)

    if (accept) then
      print *, "Configuration accepted."
    else
      print *, "Configuration rejected. Restoring old configuration."
      ! Restore old configuration (not implemented here)
    end if
  end subroutine hmc

  ! Subroutine to save gauge and fermion fields to files
  subroutine save_fields()
    integer :: x, y, z, t, i, j

    open(unit=10, file='gauge_fields.txt', status='replace')
    do x = 1, nx
      do y = 1, ny
        do z = 1, nz
          do t = 1, nt
            do i = 1, Nc
              do j = 1, Nc
                write(10, '(5I5, 6F12.6)', advance='no') &
                     x, y, z, t, i, &
                     real(U(x, y, z, t, i, j, 1)), aimag(U(x, y, z, t, i, j, 1)), &
                     real(U(x, y, z, t, i, j, 2)), aimag(U(x, y, z, t, i, j, 2)), &
                     real(U(x, y, z, t, i, j, 3)), aimag(U(x, y, z, t, i, j, 3))
              end do
            end do
          end do
        end do
      end do
    end do
    close(10)

    open(unit=20, file='fermion_fields.txt', status='replace')
    do x = 1, nx
      do y = 1, ny
        do z = 1, nz
          do t = 1, nt
            do i = 1, Nc
              write(20, '(4I5, 2F12.6)', advance='no') &
                   x, y, z, t, &
                   real(psi(x, y, z, t, i)), aimag(psi(x, y, z, t, i))
            end do
          end do
        end do
      end do
    end do
    close(20)

    print *, 'Fields saved to files.'
  end subroutine save_fields

end module lattice_qcd
