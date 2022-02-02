# !/bin/bash

# dSprites

# all data
python -m experiments.composition with dataset.dsprites \
				       model.{abdi,composition_op='interp'} \
				       training.{lr=0.0001,epochs=100}

python -m experiments.composition with dataset.dsprites \
				       model.{abdi,composition_op='fixint'} \
				       training.{lr=0.0001,epochs=100}

python -m experiments.composition with dataset.dsprites \
				       model.{abdi,composition_op='interp'} \
               training.{waemmd,batch_size=64,lr=0.0001,epochs=100}

python -m experiments.composition with dataset.dsprites \
				       model.{abdi,composition_op='fixint'} \
               training.{waemmd,batch_size=64,lr=0.0001,epochs=100}

# recomb2range
python -m experiments.composition with dataset.dsprites \
				       dataset.{condition=recomb2range,variant=sqr2tx} \
				       model.{abdi,composition_op='interp'} \
				       training.{lr=0.0001,epochs=100}

python -m experiments.composition with dataset.dsprites \
				       dataset.{condition=recomb2range,variant=sqr2tx} \
				       model.{abdi,composition_op='fixint'} \
				       training.{lr=0.0001,epochs=100}

python -m experiments.composition with dataset.dsprites \
				       dataset.{condition=recomb2range,variant=sqr2tx} \
				       model.{abdi,composition_op='interp'} \
               training.{waemmd,lr=0.0001,epochs=100}

python -m experiments.composition with dataset.dsprites \
				       dataset.{condition=recomb2range,variant=sqr2tx} \
				       model.{abdi,composition_op='fixint'} \
               training.{waemmd,lr=0.0001,epochs=100}

# 3DShapes

# All data

python -m experiments.composition with dataset.shapes3d \
				       model.{abdi,composition_op='interp'} \
				       training.{lr=0.0001,epochs=100}

python -m experiments.composition with dataset.shapes3d \
				       model.{abdi,composition_op='fixint'} \
				       training.{lr=0.0001,epochs=100}

python -m experiments.composition with dataset.shapes3d \
				       model.{abdi,composition_op='interp'} \
               training.{waemmd,lr=0.0001,epochs=100}

python -m experiments.composition with dataset.shapes3d \
				       model.{abdi,composition_op='fixint'} \
               training.{waemmd,lr=0.0001,epochs=100}

python -m experiments.composition with dataset.shapes3d \
				       model.{abdi,sbd2,composition_op='interp'} \
               training.{lr=0.0003,epochs=20,batch_size=16}

# floor and wall hue

python -m experiments.composition with dataset.shapes3d \
              dataset.{condition=recomb2range,variant=fhue2whue} \
              dataset.modifiers='["even_wnf_hues"]' \
              model.{abdi,composition_op='interp'} \
              training.{lr=0.0001,epochs=100}

python -m experiments.composition with dataset.shapes3d \
              dataset.{condition=recomb2range,variant=fhue2whue} \
              dataset.modifiers='["even_wnf_hues"]' \
              model.{abdi,composition_op='fixint'} \
              training.{lr=0.0001,epochs=100}

python -m experiments.composition with dataset.shapes3d \
              dataset.{condition=recomb2range,variant=fhue2whue} \
              dataset.modifiers='["even_wnf_hues"]' \
              model.{abdi,composition_op='interp'} \
              training.{waemmd,lr=0.0001,epochs=100}

python -m experiments.composition with dataset.shapes3d \
              dataset.{condition=recomb2range,variant=fhue2whue} \
              dataset.modifiers='["even_wnf_hues"]' \
              model.{abdi,composition_op='fixint'} \
              training.{waemmd,lr=0.0001,epochs=100}

# shape and object hue

python -m experiments.composition with dataset.shapes3d \
              dataset.{condition=recomb2element,variant=shape2ohue} \
              dataset.modifiers='["even_ohues"]' \
              model.{abdi,composition_op='interp'} \
              training.{lr=0.0001,epochs=100}

python -m experiments.composition with dataset.shapes3d \
              dataset.{condition=recomb2element,variant=shape2ohue} \
              dataset.modifiers='["even_ohues"]' \
              model.{abdi,composition_op='fixint'} \
              training.{lr=0.0001,epochs=100}

python -m experiments.composition with dataset.shapes3d \
              dataset.{condition=recomb2element,variant=shape2ohue} \
              dataset.modifiers='["even_ohues"]' \
              model.{abdi,composition_op='interp'} \
              training.{waemmd,lr=0.0001,epochs=100}

python -m experiments.composition with dataset.shapes3d \
              dataset.{condition=recomb2element,variant=shape2ohue} \
              dataset.modifiers='["even_ohues"]' \
              model.{abdi,composition_op='fixint'} \
              training.{waemmd,lr=0.0001,epochs=100}

# MPI dataset

# all data

python -m experiments.composition with dataset.mpi3d \
  dataset.modifiers='["four_shapes", "fix_hx"]' \
  model.{montero,composition_op='interp'} \
  training.{waemmd,batch_size=64,lr=0.0001,epochs=200}

python -m experiments.composition with dataset.mpi3d \
  dataset.modifiers='["four_shapes", "fix_hx"]' \
  model.{montero,composition_op='fixint'} \
  training.{waemmd,batch_size=128,lr=0.0001,epochs=200}

# recomb2range

# cylinder vs vx

python -m experiments.composition with dataset.mpi3d \
  dataset.{condition=recomb2range,variant=cyl2vx} \
  dataset.modifiers='["four_shapes", "fix_hx"]' \
  model.{montero,composition_op='interp'} \
  training.{waemmd,batch_size=64,lr=0.0001,epochs=200}

python -m experiments.composition with dataset.mpi3d \
  dataset.{condition=recomb2range,variant=cyl2vx} \
  dataset.modifiers='["four_shapes", "fix_hx"]' \
  model.{montero,composition_op='fixint'} \
  training.{waemmd,batch_size=64,lr=0.0001,epochs=200}

# cylinder vs background (fixed horizontal axis)

python -m experiments.composition with dataset.mpi3d \
  dataset.{condition=recomb2range,variant=bkg2cyl} \
  dataset.modifiers='["four_shapes", "fix_hx"]' \
  model.{montero,composition_op='interp'} \
  training.{waemmd,batch_size=64,lr=0.0001,epochs=200}

python -m experiments.composition with dataset.mpi3d \
  dataset.{condition=recomb2range,variant=bkg2cyl} \
  dataset.modifiers='["four_shapes", "fix_hx"]' \
  model.{montero,composition_op='fixint'} \
  training.{waemmd,batch_size=64,lr=0.0001,epochs=200}
