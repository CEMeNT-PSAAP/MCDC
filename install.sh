#!/bin/bash


# Install MC/DC module
pip install -e .


# Install MC/DC dependencies, reply "y" to conda prompt
conda install numpy numba matplotlib scipy h5py pytest colorama <<< "y"

# Installing visualization dependencies (required via pip for osx-arm64)
pip install ngsolve distinctipy

bash patch_numba.sh

# Install or build mpi4py
if [ $# -eq 0 ]; then
  conda install mpi4py <<< "y"
fi
while [ $# -gt 0 ]; do
  case $1 in
    --hpc)
      # Rename legacy compiler option in conda
      s=$(which python)
      s=${s//bin\/python/compiler_compat}

      if [ ! -f $s/ld.bak ] && [ -f $s/ld ]; then
        mv $s/ld $s/ld.bak
      fi

      mkdir installs; cd installs
      wget https://github.com/mpi4py/mpi4py/releases/download/3.1.4/mpi4py-3.1.4.tar.gz -q
      tar -zxf mpi4py-3.1.4.tar.gz
      cd mpi4py-3.1.4
      python setup.py install
      cd ../../
      rm -rf installs/
      ;;

    --config_cont_lib)
      bash config_cont_energy.sh
  ;;
  esac
  shift
done

