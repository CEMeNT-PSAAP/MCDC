#!/bin/bash


# Install MC/DC module
pip install -e .


# Install MC/DC dependencies, reply "y" to conda prompt
conda install numpy numba matplotlib scipy h5py pytest colorama <<< "y"


# Patch numba
s=$(python -c 'import numba; print(numba.__path__[0])')
s+='/core/types/scalars.py'

line=$(grep -n 'class Boolean(Hashable):' $s | cut -f1 -d:)

head -n $line $s > tmp-scalars.py
cat >> tmp-scalars.py << EOF
    def __init__(self, name):
        super(Boolean, self).__init__(name)
        self.bitwidth = 8
EOF

length=$(wc -l < $s)
let "length -= line"
tail -n $length $s >> tmp-scalars.py

mv tmp-scalars.py $s



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
  esac
  shift
done


