# Using Machine Learning to Diagnose Alzheimer's

## Abstract
Early detection of Alzheimer’s disease (AD) remains a significant clinical challenge, as the changes associated with cognitive decline are often subtle and difficult to identify through visual assessment alone. This study investigates modern machine learning methodologies to improve the prediction of cognitive impairment using volumetric MRI–derived region-of-interest (ROI) features. We constructed three binary classifiers [NC vs. AD, MCI vs. AD, and NC vs. MCI] and evaluated various algorithms, including logistic regression, random forests, neural networks, and support vector machines (SVMs). Using measurements generated from eight anatomical brain templates, our models learned patterns indicative of normal cognition, mild cognitive impairment, and Alzheimer’s disease. Among all tested approaches, the radial basis function (RBF) SVM consistently achieved the highest performance, reaching accuracies of approximately 70–80% depending on the classification task. We discuss the implications of this model’s dominance for future clinical applications and the continued development of machine learning–driven diagnostic tools.

## Instructions to Test
If you would like to test our model, download our most current [driver](Driver%(V11).py).
You will need to download our [dataset](Datasets/ADNI-Oasis-AIBL_dataset.csv) as well. Make sure it is in the same directory as the driver. You may need to change files pathways in the code.

 The [spm_auto.m](Data%Processing/spm_auto.m) script was used to process the raw MRI images and map them onto each [brain template](Data%Processing/brain_template).

## Thank You
Thank you for visiting our GitHub!
Don't forget to [visit our website](https://sites.google.com/view/ml-alzheimers/home) to learn more about our model and the creators!
