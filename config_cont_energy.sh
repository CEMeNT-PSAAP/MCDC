#!/bin/bash

echo "WARNING: Seemless continous energy functionality"
echo "is only avlible to CEMeNT members."
echo " "

# downloading library from zenodo
git clone git@github.com:CEMeNT-PSAAP/MCDC-Xsec.git

cd MCDC-Xsec

# untaring file
tar -xvzf mcdc_xs.tar.gz

# going into the cross section library directory
cd mcdc_xs

# getting the present working direcotry
xsec_dir=$(pwd)

# exporting the needed enviroment variable for current enviroment
export MCDC_XSLIB="${xsec_dir}"

# sotring that enviroment variable in bashrc for future runs
export MCDC_XSLIB="${xsec_dir}">> ~/.bashrc 

# printing a complete message
echo "MCDC_XSLIB set as $MCDC_XSLIB in ~/.bashrc"