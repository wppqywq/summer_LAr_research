# CERN Lar Energy Reconstruction Analysis - README

## Introduction
This repository contains the analysis of the energy reconstruction in the LAr using optimal filters and linear/non-linear Finite Impulse Response (FIR) filters. The analysis is divided into four main parts, each focusing on specific aspects of the data and filter development. Additionally, some of the filter development work has been based on the contributions of Professor Jonathan Le Roy Sievers.

### Main Repository Structure
The repository is organized as follows:

- [1_familiar_with_data.ipynb](./1_familiar_with_data.ipynb): the purpose of getting familiar with the dataset used in the analysis.

- [2_compare_wiener&OF.ipynb](./2_compare_wiener_OF.ipynb): a comparison is made between the outputs of the optimal filter and the Wiener filter using the available data.

- [3_implement_fir_compare_OF.ipynb](./3_implement_fir_compare_OF.ipynb): we develop Finite Impulse Response (FIR) filters with linear, quadratic, and cubic characteristics and compare their performance against the optimal filter.

- [4_root_data_behavior.ipynb](./4_root_data_behavior.ipynb): introduces five large datasets with diverse characteristics to evaluate the performance of our filters.

### Additional Filter Development
The following files contribute to the development of some of the filters used in this analysis:

- [sievers_of_deco_bayes.ipynb](./sievers_of_deco_bayes.ipynb): This notebook contains the development of the Optimal Filter (OF), deconvolution, and Bayesian Reconstruction, based on the work of Professor Jonathan Le Roy Sievers.

- [nlfir.py](./nlfir.py): To optimize the processing speed for the large dataset, the FIR filter implemented using matrix methods has been converted from the time domain to the frequency domain. This significant enhancement can be found in this file, and further details can be explored in the repository at https://github.com/sievers/nlfir.

## Data Extraction and Visualization
To work with the five large datasets, the following notebooks are included:

- [extract_root.ipynb](./extract_root.ipynb): This notebook extracts arrays from root files to prepare the data for analysis.

- [show_data.ipynb](./show_data.ipynb): Use this notebook to visualize the structure and content of the large datasets.
