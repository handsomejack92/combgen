# Code for Lost in Latent Space (under review at ICML 2022)

---

This repo contains the code necessary to run the experiments for the article. The code was tested on Python 3.8 and PyTorch 1.10.

## Requirements

Running these experiments requires (among others) the following libraries installed:

* [PyTorch and Torchvision](https://pytorch.org/): Basic framework for Deep Learning models and training.
* [Ignite](https://github.com/pytorch/ignite): High-level framework to train models, eliminating the need for much boilerplate code.
* [Sacred](https://github.com/IDSIA/sacred): Libary used to define and run experiments in a systematic way.
* [Matplotlib](https://matplotlib.org/): For plotting.

A conda environment can be created using the following command:
```
conda env create -f torchlab.yml
```

## Directory structure

The repository is organized as follows:

```
data/
├── raw/
    ├── dsprites/
    ├── shapes3d/
    ├── mpi/
    ├── spriteworld/ # Which contains both the Simple and Circles datasets.
├── sims/
    ├── composition/
    ├── supervised/
├── results/
scripts/
├── configs/
    ├── vaes.py    # An example config file with VAE architectures. Other files not listed.
├── ingredients/
    ├── models.py  # Example ingredient that wrapps model initalization. Other ingredients not listed.
├── experiments/
    ├── composition.py  # Experiment script for training models wiht the image composition task.
src/
├── analysis/  # These folders contain the actual datasets, losses, model classes etc.
├── dataset/
├── models/
├── training/
```

The data structure should be self explanatory for the most part. The main thing to note is that ``src`` contains code for models that are used throughout the experiments while the ingredients contain wrappers around these to initialize them from the configuration files. Simulation results will be saved in sims. The results of the analysis were stored in a new folder (``results``, not shown). We attempted to use models with the hightes disentanglement in our analysis.

Datasets should appear in a subfolder as shown above. Right now, there is not method for automatically downloading the data, but they can be found in their corresponding repos. Alternatively, altering the source file or passing the dataset root as a parameter can be used to look for the datasets in another location[^1].

The configuration folder has the different parameters combinations used in the experiments. Following these should allow someone to define new experiments easily. Just remember to add the configurations to the appropriate ingredient using ``ingredient.named_config(config_function/yaml_file)``.

## Downloading the datasets

We provide the ``Circles`` and ``Simple`` datset as part of the repository. For the other datasets, refer to their corresponding GitHub repositories:
1. **dSprites**: https://github.com/deepmind/dsprites-dataset.
2. **3DShapes**: https://github.com/deepmind/3d-shapes.
3. **MPI3D**: https://github.com/rr-learning/disentanglement_dataset.

## Training models

To train a model, execute one of the scripts from the scripts folder with the appropraite options. We use Sacred to run and track experimetns. You can check the online documentation to understand how it works. Below is the general command used and more can be found in the ``bin`` folder.

```
cd ~/path/to/project/scripts/
python -m experiments.composition with dataset.<option> model.<option> training.<option>
```

Sacred allows passing parameters using keyword arguments. For example we can change the latent size and $\beta$ from the default values:

```
python -m experiments.composition with dataset.dsprites model.kim training.factor model.latent_size=50 training.loss.params.beta=10
```

We provide bash scripts to train models on all conditions presented in the article. These can be found in the ``bin`` folder and are prefixed by the name ``run-``.

## Producing figures

There are two scripts that produce figures. One is ``analysis.model``, which runs the analysis for one model at time. Once completed, ``analysis.summary`` will compile these results into sets of plots for the selected models containing reconstructions, Hinton matrices and latent representation visualisations. Additionally, a latex table with the scores will be generated.

Note that the uploaded versions of these files reference the models that we used for plotting. These will diverge from the ones produced when rerunning the training scripts, which means that the saved folder name will have to be changed accordingly.
