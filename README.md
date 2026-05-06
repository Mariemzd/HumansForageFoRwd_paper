\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{a4paper, margin=1in}

\title{\vspace{-2cm}Humans Forage for Reward in Reinforcement Learning Tasks\\
\large Code and Supplementary Information}
\author{Meriam Zid, Veldon-James Laurie, Jorge Ramírez-Ruiz, Alix Lavigne-Champagne, \\
Akram Shourkeshti, Dameon Harrell, Alexander B. Herman, R. Becket Ebitz.}
\date{}

\begin{document}

\maketitle

\noindent \textbf{Link to paper:} \href{https://www.biorxiv.org/content/10.1101/2024.07.08.602539v2}{[Version 2]}\\
\textbf{Link to official version:} to be added.

\section*{Overview}
This repository contains the scripts and Jupyter notebooks required to generate the data, perform analyses, and reproduce the figures presented in the article ``Humans forage for reward in reinforcement learning tasks.''

\section*{Reproducing the Figures}

\subsection*{Figure 1 \& Supplementary Figure 1}
\begin{itemize}
    \item \textbf{Experiment 1 Behavior Analysis:} Run the notebook \texttt{\textbackslash Human\_data\_analysis\_2AB.ipynb}.
    \item \textbf{Experiment 1 Choice Predictive GLM:} Scripts are located in the \texttt{\textbackslash glms\textbackslash} directory.
\end{itemize}

\subsection*{Figure 2 \& Supplementary Figure 2}
\begin{itemize}
    \item \textbf{Environment Features Analysis:} Use the code found in \texttt{behByEnvFeatures/quantifyingRichnessByDifference\_env}.
\end{itemize}

\subsection*{Figure 3 \& Supplementary Figure 3 (Model Fitting \& Agent Performance)}
\begin{itemize}
    \item \textbf{Optimal Agent Simulations:} First, run \texttt{Simulate\_Experiment1.ipynb}, and then use \texttt{PlotAgentPerf.ipynb} to visualize the results.
    \item \textbf{Model Fitting:} Fitting code is provided in both Python and MATLAB.
    \begin{itemize}
        \item \textbf{Python:} \texttt{fitModels.ipynb} --- This notebook contains all Python models used to generate Figure 3 and Supplementary Figure 3.
        \item \textbf{MATLAB:} Each model is in a separate script and \texttt{fitRLtoMTurk\_RLvsForagingByRichness.m} fits them.
        \item \textbf{(Panels G--K):} \texttt{evalFits\_RLvsForagingByRichness.m} --- This script is used specifically for plotting the model fits and log-likelihoods.
    \end{itemize}
\end{itemize}

\subsection*{Figure 4}
\begin{itemize}
    \item \textbf{Simulations:} Generate the simulations shown in Figure 4 using the MATLAB script \texttt{simulateAgentsExperiment1\_vMatchingEnv\_RLvsForagingByRichness.m}.
    \item \textbf{Plotting \& Mixture Model:} To plot the results and run the exponential mixture model, use \texttt{compareTimes\_RLvsForagingByRichness}.
    \item \textbf{Data Availability:} The pre-generated simulation datasets used for these analyses are available in the ``SimulationsOutputs'' folder on Figshare.
\end{itemize}

\end{document}

These scripts were tested on MacOS Version 12.5.1 with python version 3.9.16

### Dependencies 
All software dependencies (including version numbers) can be found in requirements.txt

