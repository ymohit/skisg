# Kernel Interpolation with Sparse Grids 
This repository hosts code for our [NeurIPS-2022 paper](https://openreview.net/pdf?id=ACThGJBOctg).


## Quick set-up and examples 

### Install requirements
```
conda create -n skisg --file environment.yml
```

### Code organization 
```
├── data                    -- store datasets 
├── scripts                 -- shell scripts for tests and environ vars
├── skisg
│   ├── algos               -- MVM algorithms for sparse grid 
│   ├── interp
│   │   └── sparse          -- Interpolation methods for sparse grid 
│   ├── kernels             -- SKI kernel functions for sparse grids 
│   ├── lazy                -- LazyTensors for kernel matrices
│   └── models              -- Several GP models 
├── tests                   -- Contains test code 
├── vizs                    -- Visualization for interpolaiton, sprase grids 
│   ├── interpolation
│   ├── sgconstruct
│   └── tchinterp
└── wandb                   -- Holds wandb logs for executions 
```

### Running tests 
* Tests for mvm algorithms, interpolation, kernels, and lazy tensors. 
* All tests require less than 100 seconds.  
```
bash scripts/tests.sh
```

### Running an example 
```
$py tests/models/sinetest.py #-- is an example of using SKI with both rectlinear and sparse grids. 
```
