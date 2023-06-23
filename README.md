# Probabilistic Safety for BNNs

### For the journal extension code (with significant experimental updates) see the following repository:
### https://github.com/matthewwicker/AdversarialRobustnessCertificationForBNNs/tree/main

This is code intended to reproduce the figures and experiments for the paper: Probabilistic Safety for Bayesian Neural Networks


![alt text](https://raw.githubusercontent.com/matthewwicker/ProbabilisticSafetyforBNNs/master/SimpleSafetyExample/Example.png)


NB: Some files are lablled 'ProbablisticReachability' these compute probablistic safety and will soon be refactored.

### Repository Organization:

SimpleSafetyExample: Run 'SimpleReachability' notebook to train and get first figure in paper. Run ProblemInsights to get figure two. Further exploration in ProcessInsights


VCAS: Code for VCAS experiments. Compile the dataset first before running verification or training script


MNIST: Code for MNIST experiments. This has seperation between 1 and 2 layer networks.
