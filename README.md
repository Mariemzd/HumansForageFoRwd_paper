**Authors:** Meriam Zid, Veldon-James Laurie, Jorge Ramírez-Ruiz, Alix Lavigne-Champagne, Akram Shourkeshti, Dameon Harrell, Alexander B. Herman, R. Becket Ebitz.

**Link to paper:** [Version 2](https://www.biorxiv.org/content/10.1101/2024.07.08.602539v2)  
**Link to official version:** to be added.

## Overview
This repository contains the scripts and Jupyter notebooks required to generate the data, perform analyses, and reproduce the figures presented in the article "Humans forage for reward in reinforcement learning tasks."

## Reproducing the Figures

## Figure 1 & Supplementary Figure 1
*   **Experiment 1 Behavior Analysis:** Run the notebook `Human_data_analysis_2AB.ipynb`.
*   **Experiment 1 Choice Predictive GLM:** Scripts are located in the `\glms\` directory.

## Figure 2 & Supplementary Figure 2
*   **Environment Features Analysis:** Use the code found in `behByEnvFeatures\quantifyingRichnessByDifference_env.m`.

## Figure 3 & Supplementary Figure 4 (Model Fitting & Agent Performance) 
*   **Optimal Agent Simulations:** Scripts are located in the `\modelling_python\` directory. Run `Simulate_Experiment1.ipynb` ;  use `PlotAgentPerf.ipynb` to visualize the results.
*   **Model Fitting:** Fitting code is provided in both Python  (under `\modelling_python\`) and MATLAB  (under `\modelling_matlab\`).
    *   **Python:** 
    	*   **Fitting:** `fitModels.ipynb`: This notebook contains the python version of the the models used to generate Figure 3 and Supplementary Figure 4.
	*   **Plotting:** `FigureFitRichnessExperiment1_supplementary.ipynb`: Run this notebook to generate the visualisations. These scripts were developed and tested using Python 3.11
    *   **MATLAB:** 
        *   **Fitting:** Run `fitRLtoMTurk_RLvsForagingByRichness.m`. This is the main script that fits the data by calling the individual models (which are modularized into their 		own separate scripts). The original fits were generated using Matlab version
        *   **Plotting (Panels G–K):** Run `evalFits_RLvsForagingByRichness.m` to plot the resulting model fits and log-likelihoods.

## Figure 4
Scripts are found in `\modelling_matlab\simulations`
*   **Simulations:** Generate the simulations shown in Figure 4 using the MATLAB script `simulateAgentsExperiment1_vMatchingEnv_RLvsForagingByRichness.m`.
*   **Plotting & Mixture Model:** To plot the results and run the exponential mixture model, use `compareTimes_RLvsForagingByRichness`.

## Figure 5 

*   **Simulations:** Generate the simulations shown in Figure 5 using the MATLAB script `simulateAgents_Experiment{n}_RLvsForagingByRichness.m`. These scripts can be found in `\modelling_matlab\simulations`
*   **Visualisation for Figure 5:** Generates all figures in figure 5 `Figure5_Experiment{n}.m`. These scripts can be found in `\modelling_matlab\figure5`

## Other
*   **Supplementary Figure 3:** open the folder `strategyChange/` and run `fitsStability.m`
*   **Supplementary Figure 6:** run `unchosenOption_exploreStrategy.m`
* HMM fits and explore/exploit dynamics analyses can be found in `\MATLAB\simulations\exploreDynamics.m`

## Data Availability
All behavioral data from all experiments, all model fits, and all simulations used for these analyses are available on figshare under : https://doi.org/10.6084/m9.figshare.32193990 