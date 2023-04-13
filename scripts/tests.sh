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
$nose -D tests.kernels.sgkernel
$nose -D tests.kernels.sginterpkernel
$nose -D tests.kernels.gridinterpkernel
# $py tests/kernels/.py # -- this isn't required as we use a different interpolation strategy

# lazy
echo 'Tests for sparse grid kernel and its interpolation tensors ...'
$nose -D tests.lazy.sgkerneltensor
$nose -D tests.lazy.sginterptensor

# models
$nose -D tests.models.sinetest