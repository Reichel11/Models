

# How to create a new model?

# Description

How to create a new model?

To create and train a new model, the model configurations of model_group.py can be used, adjusted and executed. There you can find the model configuration for the original dataset with an augmentation layer as well as the configuration with an already augmented dataset. These configurations worked best for the respective dataset. 

Recommendation: For the beginning you should work with the original dataset and the augmentation layer configuration, because it is very simple and does not need so much computing power and still could achieve an equivalent result. (see Master_Thesis.pdf)

What are the actual models able to predict?

- The models can predict open and closed MCC with high accuracy.
- The models are trained to additionally detect scenes that do not meet the criteria of open and closed low-level scenes (noMCC).
- However, the accuracy in the noMCC case is lower and the models are therefore not capable of predicting noMCC scenes, but the noMCC classification is used as not open or closed MCC classification.
### Statistics of Models 
<img src="statistics.png" width="800">


Both models have similar precision, recall, and F1-score. Thus, both models can be used to predict open and closed MCC with an overall accuracy of 80.6%. While the Model1 "Model1_layer.hdf5" with the automatic data augmentation has a higher precision in open MCC, the Model2 "Model2_manuel.hdf5" with the manual data augmentation has a higher recall of open MCC. This means that while the Model1 has predicted fewer open MCC correctly as open MCC, it has categorized also fewer other categories as open MCC, leading to a higher precision of open MCC with a lower recall (see also Fig. 4.7 in *Master_Thesis.pdf*). In contrast, Model2 has predicted a higher number of open MCC correctly as open, but also more noMCC cases are predicted as open MCC, leading to a lower precision and a higher recall of open MCC (see also Fig. 4.7 in *Master_Thesis.pdf*). However, most noMCC scenes, that are identified as open, could also by eye be identified as open, but might not match the 70% criteria of the visual inspection (see Fig. 4.9 in *Master_Thesis.pdf*). Therefore, we trust both models to predict open and closed MCC with high accuracy, in case the precision of open and closed MCC is particularly important we recommend using the Model1.
      

# Configurations

1. What to do to create a new model?
  - training data (one array with the data itself and one array for the categories)
  - configure training data (fillvalue,...
  - set model configurations


# Model settings
1. What to change and what do they do?
 - Filter/Kernel size, activation function, Dense-layer, learning-rate, loss function,optimizer
# How to avoid Overfitting

1. What is overfitting?
2. What to use against overfitting?
  - Regularizations:Dropout, batchnorm, data augmentation, 

# Improvement Ideas

# Technical Requirements
Python (version 3.7) is required to use the models. Furthermore the installation of the following packages is necessary:
- Tensorflow – 2.7.0
- Keras – 2.6.0
- Pickleshare – 0.7.5
- Numpy – 1.19.5
- Matplotlib – 3.1.3
