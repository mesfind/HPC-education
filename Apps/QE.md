
$ dnf install -y git-all

QE installation requires the Intel OneAPI C and FORTRAN compilers:


$ dnf config-manager --add-repo https://yum.repos.intel.com/oneapi
$ rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
$ dnf -y install intel-basekit intel-oneapi-compiler-fortran


Configuration for QE v7.1:
./configure --prefix=/opt/ohpc/pub/apps/qe CC=icx F77=ifx F90=ifx MPIF90=mpifort CFLAGS="-O3" 
FFLAGS="-O3 -xCORE-AVX2 -pthread"  --with-scalapack=intel
