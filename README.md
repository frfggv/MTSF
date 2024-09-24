# MTSF

The main code can be found in master.

The task of multivariate time series forecasting (MTSF) has become the focus of numerous studies.However, there are still significant research gaps in capturing the correlations between variables in different time patterns (trends and seasons) of time series. To address this oversight, we propose a Clustering Convolutional Network based on Mamba Representation (MambaCCN) for MTSF.

Our main contributions are as follows:

1. We propose a Clustering Convolutional Network based on Mamba representation (MambaCCN), realizing accurate multivariate prediction on the moving average decomposition architecture.
2. We introduce a Mamba-based Deep Representation Module (MDR-Module) in the network, relieving the distribution shift between the look-back and horizon.
3. We introduce the Clustering Convolution Module (CC-Module) in the network, aggregating relevant variables to capture patterns of change between and within variables.
