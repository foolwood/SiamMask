#!/usr/bin/env bash

cd utils/pyvotkit
python setup.py build_ext --inplace
cd ../../

cd utils/pysot/utils/
python setup.py build_ext --inplace
cd ../../../
