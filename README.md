# MTSF

The main code can be found in master.

The task of multivariate time series forecasting (MTSF) has become the focus of numerous studies.However, there are still significant research gaps in capturing the correlations between variables in different time patterns (trends and seasons) of time series. To address this oversight, we propose a Mamba clustering and representation convolutional network (MCRCN) on the moving average decomposition architecture for MTSF.

Our main contributions are as follows:

1.We recognize the complex correlation between variables and have designed MCRCN to enhance predictive performance in MTSF by capturing these dependencies as they manifest in trend and seasonal patterns.\n
2. To search for similar variables and minimize interference from irrelevant ones, we propose a Mamba-based Deep Clustering and Representation (MDCR) module, utilizing the K-means algorithm (Kingma & Welling, 2013) and three distinct Mamba blocks to cluster and represent multiple time series variables.
3. To effectively leverage the advantages of CD and simulate inter- and intra-variable dynamics, we introduce a variate-time convolution (V-T-Conv) block with positional memory, capable of simultaneously capturing two-dimensional changes in time and variables within the same cluster.
