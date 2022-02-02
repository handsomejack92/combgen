# !/bin/bash

RESULTS_FOLDER="../data/results/composition"
SAVE_FOLDER="../plots/summary"

# dSprites
python -m analysis.summary --model_folders $RESULTS_FOLDER/{5,6,7,8} \
                           --name sqr2px \
                           --save $SAVE_FOLDER \
                           --all

# 3DShapes
python -m analysis.summary --model_folders $RESULTS_FOLDER/{14,15,16,17} \
                           --name whue2fhue \
                           --save $SAVE_FOLDER \
                           --all

python -m analysis.summary --model_folders $RESULTS_FOLDER/{18,19,20,21} \
                           --name shape2ohue \
                           --save $SAVE_FOLDER \
                           --all

# MPI3D
python -m analysis.summary --model_folders $RESULTS_FOLDER/{24,25} \
                           --name shape2vx \
                           --save $SAVE_FOLDER \
                           --all

python -m analysis.summary --model_folders $RESULTS_FOLDER/{26,27} \
                           --name cyl2bkg \
                           --save $SAVE_FOLDER \
                           --all

# Circles
python -m analysis.summary --model_folders $RESULTS_FOLDER/29 \
                           --name circles-corner \
                           --save $SAVE_FOLDER \
                           --all

python -m analysis.summary --model_folders $RESULTS_FOLDER/30 \
                           --name circles-midpos \
                           --save $SAVE_FOLDER \
                           --all

# Simple
python -m analysis.summary --model_folders $RESULTS_FOLDER/34 \
                           --name simple-corner \
                           --save $SAVE_FOLDER \
                           --all

python -m analysis.summary --model_folders $RESULTS_FOLDER/36 \
                           --name simple-midpos \
                           --save $SAVE_FOLDER \
                           --all

# Supervised
RESULTS_FOLDER="../data/results/predictors"
python -m analysis.summary --model_folders $RESULTS_FOLDER/{2,4,6} \
                           --name supervised \
                           --save $SAVE_FOLDER \
                           --score --latent_reps
