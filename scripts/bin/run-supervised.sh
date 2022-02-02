# !/bin/bash


# dSprites

python -m experiments.supervised with model.kim \
				      dataset.{dsprites,condition=recomb2range,variant=ell2tx} \
				      training.regression

python -m experiments.supervised with model.kim \
				      dataset.{dsprites,modifiers=["sparse_posX"]} \
              dataset.{condition=recomb2range,variant=ell2tx} \
				      training.regression


# 3DShapes

# All data

python -m experiments.supervised with model.kim \
				      dataset.{shapes3d,pred_type="reg"} \
				      training.regression

# OOD condition

python -m experiments.supervised with model.kim \
				      dataset.{shapes3d,modifiers="['even_ohues']"} \
              dataset.{condition=recomb2range,variant=shape2ohue} \
				      training.regression seed=148645996

# MPI3D

# All data

python -m experiments.supervised with model.montero \
          dataset.{mpi3d,norm_lats=True,modifiers='["four_shapes", "fix_hx"]'} \
          training.regression

# Recomb2range

python -m experiments.supervised with model.montero \
          dataset.{mpi3d,norm_lats=True,modifiers='["four_shapes", "fix_hx"]'} \
          dataset.{condition=recomb2range,variant=cyl2vx} \
          training.regression
