Trajectory Forecasting with Graph Neural Networks
==============================

Repository for Master's Thesis in Mathematical Modelling and Computation (MSc.) at the Technical University of Denmark

Project Organization
------------

    ├── LICENSE
    ├── Makefile                            <- Makefile with commands like `make data` or `make train`
    ├── README.md                           <- The top-level README for developers using this project.
    ├── setup.py                            <- makes project pip installable (pip install -e .) 
    |                                          so src can be imported
    ├── configs
    |   ├── nbody
    |   |   ├── datamodule                  <- Dir with yaml configuration files for datamodules
    |   |   ├── model                       <- Dir with yaml configuration files for models
    |   |   ├── regressor                   <- Dir with yaml configuration files for pl.LightningModules
    |   |   ├── trainer                     <- Dir with yaml configuration files for pl.Trainer
    |   |   └── config.yaml                 <- Default config file
    |   |
    |   └── waymo
    |       ├── datamodule                  <- Dir with yaml configuration files for datamodules
    |       ├── model                       <- Dir with yaml configuration files for models
    |       ├── regressor                   <- Dir with yaml configuration files for pl.LightningModules
    |       ├── trainer                     <- Dir with yaml configuration files for pl.Trainer
    |       └── config.yaml                 <- Default config file
    |    
    |── models                              <- Full yaml files for all models
    |   ├── nBody_results                   
    |   └── waymo_results
    |    
    ├── src                                 <- Source code for use in this project.
    │   ├── __init__.py                     <- Makes src a Python module
    │   │
    │   ├── data           
    │   │   ├── dataset_nbody.py            <- Script containing n-body torch_geometric.InMemory 
    |   |   |                                  datasets and pl.LightningDataModules 
    |   |   ├── dataset_waymo.py            <- Script containing Waymo torch_geometric.InMemory 
    |   |   |                                  datasets and pl.LightningDataModules 
    |   |   ├── make_nbody_dataset.py       <- Script to generate nbody datasets and modules
    |   |   ├── make_waymo_dataset.py       <- Script to generate Waymo datasets and modules
    |   |   └── run_simulations.py          <- Script to run n-body simulations
    │   │
    │   ├── models          
    │   │   ├── model.py                    <- Script containing all full models. 
    |   |   |                                  Type: torch.nn.Module
    │   │   ├── node_edge_blocks.py         <- Script containing node and edge update functions.
    |   |   |                                  Type: torch.nn.Module
    │   │   └── unused_models.py            <- Script containing old/developmental node and edge 
    |   |                                      update functions and full models. Type: torch.nn.Module
    │   │
    │   ├── predictions
    │   │   ├── compile_nbody_results.py    <- Script to process results
    │   │   ├── compile_waymo_results.py    <- Script to process results
    │   │   ├── evaluate_nbody.py           <- Script to validate and test n-body models
    │   │   ├── evaluate_waymo.py           <- Script to validate and test Waymo models
    │   │   ├── make_nbody_predictions.py   <- Script to predict and visualise n-body models
    │   │   ├── make_waymo_predictions.py   <- Script to predict and visualise Waymo models
    │   │   ├── make_waymo_predictions_sampled.py   <- Script to predict and visualise Waymo models
    │   │   └── make_waymo_predictions_ua.py <- Script to predict and visualise Waymo models
    │   │
    │   ├── training_modules
    │   │   ├── unused_models               <- Dir with various developmental/discontinued files
    │   │   ├── resume_training.py          <- Script to continue training of models. 
    │   │   ├── training_nbody_model.py     <- Script to train/validate/test/predict nbody models. 
    |   |   |                                  Type: pl.LightningModule
    │   │   └── training_waymo_model.py     <- Script to train/validate/test/predict waymo models. 
    |   |                                      Type: pl.LightningModule
    │   │   
    │   ├── visualization                   <- Dir with various visualisation scripts
    │   |   ├── visualise_local_map.py
    │   |   ├── visualise_maps.py
    │   |   ├── visualise_maps_w_cars.py
    │   |   └── visualize_nbody_sequence.py
    │   │   
    |   └── utils.py                        <- Various helper functions in data generation
    |
    └── visualisations                      <- All visualisations used in the thesis
        ├── nbody
        └── waymo    

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
