# MTSF

The main code can be found in master.

Multivariate Time Series Forecasting (MTSF) poses a challenge in effectively capturing complex dependencies between and within variables. Specifically, the problem of distribution shift in time series makes it difficult to predict. In addition, existing convolution methods ignore the variable correlation differences, leading to reduced accuracy.

Our main contributions are as follows:

1. We propose a Clustering Convolutional Network based on Mamba representation (MambaCCN), realizing accurate multivariate prediction on the moving average decomposition architecture.
2. We introduce a Mamba-based Deep Representation Module (MDR-Module) in the network, relieving the distribution shift between the look-back and horizon.
3. We introduce the Clustering Convolution Module (CC-Module) in the network, aggregating relevant variables to capture patterns of change between and within variables.

## 模型结构图
![image](https://github.com/frfggv/MTSF/blob/master/pictures/MambaCCN2.png)

## 部分基线对比实验结果
![image](https://github.com/frfggv/MTSF/blob/master/pictures/69d6afdd3fd2471e1aa216bea1f78be.png)
## 部分消融实验结果
![image](https://github.com/frfggv/MTSF/blob/master/pictures/消融.png)
## 模型效率与参数
![image](https://github.com/frfggv/MTSF/blob/master/pictures/model_efficiency_memory_usage.png)
