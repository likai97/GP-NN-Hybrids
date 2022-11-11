# Gaussian Process Regression and Bayesian Deep Learning for Insurance Tariff Migration 

This repo contains code for my Master Thesis "Gaussian Process Regression and Bayesian Deep Learning for Insurance Tariff Migration".


## Abstract
<p class="text-justify">
This thesis reviews the current state of Gaussian Processes and Bayesian Deep Learning hybrid models,
and their applicability to the transfer of actuarial functionalities. I conduct a benchmark study on
two insurance tariff datasets and four OpenML regression tasks and compare Deep Kernel Learning,
Deep Gaussian Processes, and Deep Sigma Point Processes with several strong baselines including 
a bayesian deep learning baseline. The model classes are examined with respect to fit, scalability,
stability, and uncertainty quantification capabilities. My results show that among the analyzed
models Variational Deep Kernel Learning, Deep Gaussian Processes, and Deep Ensembles
show the best results with respect to predictive performance and uncertainty estimation 
abilities on the insurance tariff datasets. 
</p>

## Requirements
Python: 3.9 or 3.10

Requirements: 

* pandas==1.4.0
* numpy==1.22.0
* scikit_learn==1.1.1
* scipy==1.7.3
* plotly==5.9.0
* torch==1.11.0
* gpytorch==1.6.0
* optuna==2.10.1
* pytest==7.1.2
* openml==0.12.2
* xgboost==1.5.2

It is recommended to create a separate environment and install [requirements.txt](https://github.com/likai97/GP-NN-Hybrids/blob/main/requirements.txt)

If your system supports CUDA, you can install cudatoolkit via. It is recommended to run the GP-based methods and
Deep Ensembles using a GPU for faster runtimes. 

```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

I used cudatoolkit version 10.2

## Usage

To start the HPO trial use the experiment.py File. To change the default configurations used in this Thesis, please do it immediately in the file. 

```
python experiment.py
```

You will be promted to select the Model and the dataset which the HPO trial will be run on.

Example:

```
Enter HPO Study: VDKL
Enter file: None
dataset_id = 287
Pruner Type: None
Enter time tolerance: 600
```

To run the study on the insurance tariff datasets, enter either "k2204", "r1_08_small" or
 "r1_08" for file. Make sure the trail.X folder is in a parallel folder to this repo. 

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

## Acknowledgements  

I would like to thank my thesis advisors Dr. Janek Thomas and Philipp Müller for their continous support,
encouragment and helpful input at every step throughout the creation of this thesis. I would also like to 
thank Thomas Hoffman for his input on the insurance subject and msg life for the provision of computational resources. 
