Humans forage for reward in reinforcement learning tasks. 
by Meriam Zid, Veldon-James Laurie, Alix Levine-Champagne, Akram Shourkeshti, Dameon Harrell, Alexander B. Herman, R. Becket Ebitz. Code and supplementary information. 

Link to paper: [Version 1](https://www.biorxiv.org/content/10.1101/2024.07.08.602539v1.article-metrics)

Enclosed here you will find several scripts and notebooks to generate the data and plot the figures in the article "Humans forage for reward in reinforcement learning tasks."

This repository contains multiple scripts and notebooks that should be run as follows:
Python should be installed before, including all packages found in _requirement.txt_ .

These scripts were tested on MacOS Version 12.5.1 with python version 3.9.16

### Foraging and RL fitting via maximum likelihood estimation 

1. Download _FitModels_Experiment1_ jupyter notebook.
2. Download _Experiment1_data.pickle_ the sample dataset from experiment 1 (restless 2-armed bandit).
3. Run _FitModels_Experiment1_ using the sample data, by following the steps provided in the notebook. [run time ~ 13 minutes]
4. Save the fitted parameters and corresponding likelihoods to plot the results. You should have 2 csv files named (ForagingParams.csv and RLParams.csv).
The expected csv files are also available under the same names. 
6. Download _FigureFitExperiment1_ jupyter notebook. Run the script using the csv files from step (4) and follow the steps provided in the notebook.
   
### Simulating Foraging and RL agents in a restless 2-armed bandit environment

1. Download _Simulate_Experiment1_ jupyter notebook.
2. Download _Walk.py_ . This script will will be called by _Simulate_Experiment1_ notebook to generate the reward walks.
3. Download _Agent.py_ . This script will will be called by _Simulate_Experiment1_ notebook to simulate both RL and Foraging agents.
4. Follow the instructions found in _Simulate_Experiment1_.  You should have 1 pickle file named (bestperfDic.pickle). [run time ~ 3 minutes]
The expected pickle file is also available in the repository under the same names. 
6. Download _PlotAgentPerf_ jupyter notebook. Run the script using the pickle files from step (4) and execute all cells to visualize the plots.

### Dependencies 
All software dependencies (including version numbers) can be found in requirements.txt



### Licence

MIT License
Copyright (c) 2024 Meriam Zid and R. Becket Ebitz
