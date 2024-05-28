# triton_exp_demo
We provide instructions to run these codes in this section
## Local User
### Dependency Installation
```
conda create -n triton_exp python=3.8
conda activate triton_exp
pip install --editable .
```

### Check Cuda
```
Python check_cuda.py
```

## UCSD Remote User

We will utilzie ucsd dsmlp datahub to build a container to run the program.
More details: [UCSD DSMLP](https://support.ucsd.edu/its?id=kb_article_view&sysparm_article=KB0032269)

### SSH Remote Connection
```
ssh USERNAME@dsmlp-login.ucsd.edu
```

### Build a Container
```
launch-scipy-ml.sh -c 8 -m 32 -g 1 -v 2080ti -i python:latest -s
```
`launch-scipy-ml.sh`: Script to launch a pod
`-c 8`:  8 CPUs
`-m 32`: 32 RAM
`-g 1`: 1 GPU
`-v 2080ti`: 2080ti GPU Version
`-i python:latest`: python:latest docker image
`-s`:  launch only a shell terminal, inhibiting launch of Jupyter

### Installation
```
pip install --editable .
```

## Run Benchmarks of Different tasks
```
Python matrix_multiplication.py or ....
```

## Experiment Results
Experiment Settings: 8 CPUs; 32 RAM; 1 2080ti GPU

### vector addition:
