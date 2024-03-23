# iAmyP
iAmyP: a multi-view learning for amyloidogenic hexapeptides identification based on sequential least squares programming

## Introduction
This paper proposes a starch-like hexapeptide prediction model called iAmyP, which consists of three main steps:\
(1) Dataset acquisition and composition of positive and negative samples.\
(2) Feature processing: Four types of features (AAC, DDE, Morgan fingerprint, BLOUSUM) are used, including three perspectives. Recursive feature elimination is then employed for feature selection, followed by feature fusion using attention mechanisms.\
(3) Model construction: The Weighted Sequence Least Squares Algorithm (SLSQP) is utilized to minimize the logarithmic loss function, allocating weights to nine machine learning classifiers for integrated prediction.

## Related Files

#### iAmyP

| FILE NAME         | DESCRIPTION                                                                             |
|:------------------|:----------------------------------------------------------------------------------------|
| data              | Training dataset、Testing dataset、additional dataset
| Feature           |Four feature representation methods： Morgan fingerprint、DDE、BLOUSUM、AAC               |
| model             |Feature selection、SHAP (SHapley Additive exPlanations) value analysis and train model    |

## Installation
- Requirement

  OS：
  - `Windows` ：Windows10 or later

  Python：
  - `Python` >= 3.6
