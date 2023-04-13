#!/usr/bin/env bash

source scripts/setup.sh

## tests for algos
echo 'Tests on numpy MVM for sparse grid ...'

$nose -D tests.algos.sgnpmvm
# similar test for tch MVM is skipped as its included in lazy tensors

## tests for interp
# sparse grid tests for testing recursive structure in indexing
echo 'Tests on sparse grid ...'
$nose -D tests.interp.sparse.sparsegrid
$nose -D tests.interp.sparse.spconstruct

# kernels
#$nose  --pdb -w . tests.kernels.sgkernel
# $py tests/kernels/sginterpkernel.py # for gl=5, interp error should be 1e-3
# $py tests/kernels/gridinterpkernel.py -- this isn't required as we use a different interpolation strategy

# lazy
echo 'Tests for sparse grid kernel and its interpolation tensors ...'
$nose -D tests.lazy.sgkerneltensor
$nose -D tests.lazy.sginterptensor

# models
$nose -D tests.models.sinetest