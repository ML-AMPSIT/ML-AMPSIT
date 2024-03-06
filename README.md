

# ML-AMPSIT Tutorial: Getting Started

## Introduction

ML-AMPSIT is a machine learning based sensitivity tool that uses data-driven models to make predictions based on data that typically comes from a small ensemble of high-fidelity, more computationally intensive simulations. The goal of these models is to build a faster surrogate model to reduce the computational burden at the cost of approximating the original input-output functional relationships. 

Surrogate models are built within ML-AMPSIT to return to the user the weights of each input feature, i.e. the importance of the model parameters. 

Currently, the tool is limited to the following regression algorithms Random Forest (RF), Least Absolute Shrinkage Operator (LASSO), Support Vector Machine (SVM), Bayesian Ridge Regression (BRR), Gaussian Process Regression (GPR), Extreme Gradient Boosting (Xgboost), and Classification And Regression Trees (CART). Most of these regression algorithms provide regression weights by default, while BRR and GPR are probabilistic methods that are used within ML-AMPSIT to compute sensitivity indices via an implementation of the Sobol method ( https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis ). 

Although the Sobol method is known as a well refined method, it is also known for its very high computational cost. For fast models, the method has been used in the literature with over 10000 runs. For slow methods such as WRF, this could be significantly infeasible. The minimum number of simulations required to obtain consistent sensitivity indices also depends on the specificity of each problem and cannot be known a priori. Thanks to the use of surrogate models, complex models such as WRF can be replaced by a much simpler and faster model, allowing to compute the Sobol indices with drastically less computational resources.

The rationale for adopting a multi-method approach in our analysis tool stems from the recognition that different regression algorithms, despite aiming to achieve the same goal, operate based on fundamentally distinct principles. For instance, LASSO (Least Absolute Shrinkage and Selection Operator) employs a least squares approach with regularization, making it well-suited for linear problem-solving. In contrast, algorithms like Random Forest are based on decision tree ensembles and are inherently more adept at capturing non-linear relationships.

This diversity in algorithmic approaches is particularly valuable in scenarios where the underlying characteristics of the problem are not clearly understood a priori. For example, in cases dominated by linear relationships, methods like LASSO might offer more accurate predictions. Conversely, for problems with complex, non-linear dynamics, a method like Random Forest could potentially yield better results.

Given this context, our tool allows users to explore a variety of methodologies, providing a comprehensive perspective on the problem at hand. After evaluating the performance of different algorithms, users have the flexibility to either average the results for a more balanced view or select the best-performing method based on their specific dataset and analysis goals. This multi-faceted approach equips users with the necessary tools to effectively tackle a wide range of problems, including those that are complex or not fully understood, thereby enhancing the robustness and reliability of the analysis.

ML-AMPSIT.ipynb is the core script of the tool, and all other scripts outside of ML-AMPSIT.ipynb are supplementary. Most of these additional scripts are designed to work with the WRF model and parameters from Noah-MP's MPTABLE. However, they can be used optionally based on the specific needs of the user. The key consideration is that as long as the file format required by ML-AMPSIT.ipynb matches the expectations of the script, ML-AMPSIT can work with any model output and any set of parameters.

# Included scripts

## Additional Scripts
These additional scripts are primarily tailored for use with the WRF model and Noah-MP parameters, but are not mandatory. They can be omitted if they do not meet the specific needs of the user. ML-AMPSIT.ipynb remains adaptable and will work with any generic model and parameter set as long as the file formats are compatible.

### sobol.ipynb
This script is used to generate Sobol sequences for each parameter to be analyzed. 

Sobol sequences ( https://en.wikipedia.org/wiki/Sobol_sequence ) are quasi-random, low-discrepancy sequences that are used to generate samples for sensitivity analysis, in order to maximize the exploration of the parameter space. The Sobol sequences are used for sampling where all parameters are varied together. 

The script also includes the ability to generate one-at-time sequences, but the use of ML-AMPSIT.ipynb with this sampling approach has not yet been tested. 

To work properly, the script reads specifications from configAMPSIT.json, which will be explained later. The parameter values generated by sobol.ipynb are stored in an X.txt file, where each row corresponds to a different run and each column corresponds to a different parameter.

### autofill.sh
This script aims to automate the process of generating multiple simulations according to the perturbations defined in X.txt.

It creates N copies of the original simulations, with N the size of the matrix contained in X.txt, and then changes the parameter values in each new copy. 

Currently, it only works for the MPTABLE parameters in the vegetation type classification, as it is tailored to modify these specific files and values. The user can achieve the same effect by manually preparing N folders and changing the required parameter values without using this script.

### checksize.sh && extractdom.sh
These two optional scripts can be customized by the user to simplify some operations. The checksize.sh script is used to automatically rename all the simulations, adding the suffixes _1, _2, ..., _N necessary to make them readable by the following scripts, and to move them all into a single folder. The script extractdom.sh, to be placed inside the simulations folder, allows the user to extract only the relevant variables and to reduce the size of the domain, as this could drastically reduce the space needed to store the entire ensemble. It is up to the user to modify the lines of the scripts according to their specific needs.  

### WRFload.ipynb
After all the simulations are run, the user must place them all in a single folder so that WRFload.ipynb can read them and generate the main files that ML-AMPSIT.ipynb can read to perform the sensitivity analysis. 

The simulation files must follow specific formatting criteria; more instructions on this are provided later in this document.

## ML-AMPSIT.ipynb
This is the main script of ML-AMPSIT. It is the central component of the sensitivity analysis. When first loaded, it will present the default graphical interface shown in the figure, where the user can select the type of analysis to be performed.

![Interface of ML-AMPSIT](ML-AMPSITdisplay.png).

The user can specify which variable to analyze over which region of the domain, which regression method to use, and much more.
Many of these options are read from the configAMPSIT.json configuration file, which will be discussed later.

## Loop suite: ML-AMPSITloop.ipynb
The ML-AMPSITloop.ipynb is a modified version of the ML-AMPSIT.ipynb to automate the execution of multiple analyses within a loop without having to manually set the graphical user interface each time. The specifications for the loop must be set in the loopconfig.json, which will be discussed later.

## Postprocessing

### ConvergenceAnlys.ipynb
This script aims to aggregate the data from the ML-AMPSIT or ML-AMPSITloop output and automatically perform convergence analysis on the results, generating a plot and heatmap from which the user can evaluate if the output is consistent across methods, number of simulations, domain coordinates, and variables. (It may still be under development for public use)



## Main Configuration File
The configuration file configAMPSIT.json allows the user to fill in all the details needed for the tool to work.


### Variables used to create folder runs and populate MPTABLE.TBL

The following variables are used to generate new realizations based on a main folder (which must be already existent) containing the reference simulation files (namelist, MPTABLE, etc.), and populating the MPTABLES by modifying the specified parameters.

	"folder": "foldersim",
	"vegtype": 5,
	"totalsim": 100,
	"parameter_names": [ "DLEAF", "HVT", "Z0MVT", "RHOL_NIR"],

The following variable is used to specify a reference value and a perturbation percentage for each parameters, in the format [middle point, percentage of perturbation]. The reference values will be used by sobol.ipynb to generate one sobol sequence for each parameter within the constrained range [reference - perturbation, reference + perturbation]. The ensemble of sobol sequences will be saved in the file X.txt which must be put on the same folder level as the main folder and the autofill.sh script.

	"MATRIX": [
	[0.040, 30.000],
	[16.0, 30.000],
	[0.80, 30.000],
	[0.45, 30.000]
	],     

Taking the above specifications as an example, once these variables are correctly filled, autofill.sh will generate new folders in the range {foldersim_1 , ... , foldersim_100} containing copies of the reference simulation files, each one with a modified MPTABLE.TBL accordingly with the new parameter values inside X.txt. 

Now that the folders are created, the user can run all the simulations to obtain the perturbed ensemble to be analyzed.


### Variables used by WRFload.ipynb, ML-AMPSIT.ipynb and convergenceAnlys.ipynb

Once the ensemble is obtained, each folder will contain the same output file (e.g. wrfout_d01_2015-03-20_12_00_00). Each file must be aggregated in a dedicated folder with the suffix _1, _2, _3, ..., _100. Once everything is ready, WRFload.ipynb will extract from each of the ncfile only the essential data needed for the sensitivity analysis, generating files in a format readable by ML-AMPSIT. To do this it needs to know: 

the variables to extract:

	"variables": ["V","T","LH","HFX"],

the variable dimensions (as of now it accepts only 3d or 2d variables, to be specified in "is_3d" in boolean format (1:yes,0:no)):

	"is_3d": [1,1,0,0],

the starting date and simulation duration:

	"startTime": "2015-03-20 12:00:00",
	"totalhours": 24,  

the ncfile format to recognize the wrfout files: 

	"ncfile_format":"wrfout_d01_2015-03-20_12_00_00",  

the horizontal domain extension and the interested domain points coordinate:

	"ymax": 120,
	"xmax": 55,
	"y1": 90,
	"x1": 30,
	"y2": 90,
	"x2": 26,
	"y3": 90,
	"x3": 20,    
	"y4": 20,
	"x4": 30

labels for the chosen domain points:

	"regions": ["valley", "slope", "ridge", "plain"],
  
number of vertical level to consider:

	"verticalmax": 10,
  
where to find the simulations and where to save the files to be read by ML-AMPSIT:  

	"input_pathname": "E:/X131223VALLEY/",
	"output_pathname": "C:/Users/username/Documents/XVAL131223/", 

the number of tuning iteration, if activated: 

	"tun_iter" : 10,

This will complete the configuration specifications needed from this file.


## Loop Suite Configuration File
To avoid the problem of manually changing the ML-AMPSIT options each time the user is interested in exploring each scenario and performing a large number of analyses, the loop suite ML-AMPSITloop.ipynb has been created. This script runs the regular ML-AMPSIT in a loop, exploring a different combination of settings each time. It needs its own configuration file, loopconfig.json, to be properly populated. The user can specify the following settings in loopconfig.json.


The numer of simulations to consider for each iteration of the loop (to perform a convergence analysis), e.g: 

    "NN": [
		10,
		20,
		30,
		40,
        50
    ],
	
The numer of vertical levels to consider for each iteration of the loop (to perform a vertical dependance analysis), e.g: 
	
    "vpointt": [
        2,
        3,
        4,
        5
    ],
	
The method to consider at each iteration of the loop (to perform an ensemble method analysis), considering that each method has its own identification number (rf:1, lasso:2, svm:3, brr:4, gpr:5, xgboost:6, cart:7), e.g: 
	
    "methh": [
        2,
        3,
        4,
        5

    ],
	
The variable to consider at each iteration of the loop, considering that each variable has its own identification number defined inside configAMPSIT.json. for example if the main configuration file contains "variables": ["V","T","LH","HFX"], to loop across V,T and LH the user should specify in loopconfig.json:  
	
    "varr": [
        1,
        2,
        3

],

The region to consider at each iteration of the loop, considering that each region has its own identification number defined inside configAMPSIT.json. for example if the main configuration file contains "regions": ["valley", "slope", "ridge", "plain"], to loop across valley, slope, and ridge, the user should specify in loopconfig.json:  

    "hpointt": [
        1,
        2,
        3
    ],

The tuning options, where 0: no tuning, 1: perform the tuning and save the best set of hyperparameters inside a file, 2: search for an already existent file with the best set of hyperparameters. For example:

    "tun": 0,

The number of hours must be specified for internal code reasons:

    "hour": 24,

The number of runs to be performed with the surrogate model (only brr and gpr at the moment can use the sobol method) to compute the sobol indices:

    "Nsobol": 1000



## Compatibility
ML-AMPSIT.ipynb relies on specific file formats, and as long as your data adheres to these formats, the tool can seamlessly accommodate any generic model and parameter set. If any of the script does not meet the user needs and therfore will not be used, the user must know that certain files must be created in order for ML-AMPSIT to work. The files to be created are the following:

### X.txt
This file must contain all the input values, in NxM matrix format where N is the number of simulations and M is the number of parameters. Here is an example of values for 8 simulations and 4 parameters, showing how X.txt must be filled:  

	0.03970 18.78297 0.83512 0.51916
	0.04016 15.37639 0.72693 0.44887
	0.04039 16.65808 0.95579 0.45687
	0.03987 13.25131 0.68650 0.37546
	0.03994 17.40794 0.77859 0.39695
	0.04022 14.00146 0.86913 0.47863
	0.04004 17.93301 0.65800 0.42686
	0.03978 16.19677 0.66596 0.48523


### VAR_R_levV_time.txt
This file must contain N values, one for each simulation, corresponding to the output of a single variable, a single region specified by the region label and the coordinates x and y, a single vertical level and a single timestep. For example, if we specify in configAMPSIT.json that we have variable V, region label valley (associated to domain coordinates x1:90 y1:30), vertical level 1, and timestep 3, we would need the following file to make it readable by ML-AMPSIT:

	V_valley_lev1_time3.txt

Assuming we have 8 simulations, the contents of the file must follow this format:
 
	1.3085256814956665,1.320160150527954,1.2770142555236816,1.1629282236099243,1.3877079486846924,0.8722674250602722,1.3292431831359863,1.4768025875091553

ML-AMPSIT needs this type of file for each variable, each region, each vertical level and each time. For instance, if we have 2 variables, 4 regions, 10 vertical levels and 24 times, a total of 2x4x10x24=1920 files.
	
These files, together with X.txt, configAMPSIT.json, and the optional loopconfig.json, guarantee that ML-AMPSIT will work properly even if the user did not follow the standard workflow.

## Warning
Currently, if the user needs to change parameters outside of the canopy related group in MPTABLE.TBL, autofill.sh cannot be used and the user must manually change the parameter values. If the user needs to implement ML-AMPSIT on a model other than Noah-MP or WRF, the user must follow the instructions in the Compatibility section to make it work.

##DOI Badge

[![DOI](https://zenodo.org/badge/768182093.svg)](https://zenodo.org/doi/10.5281/zenodo.10789929)
