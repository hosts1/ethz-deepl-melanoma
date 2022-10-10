# Classification of a Melanoma with Deep Learning
## 447-6265-00L Deep Learning: Ein probabilistischer Ansatz HS2022
In this group work we applied different Deep Learning methods on a public dataset containing different melanomas. The goal is to identify harmful melanomas, which cause cancer.

## Table of Contents
* [General Info](#general-information)
* [Conclusion](#conclusion)
* [Setup](#setup)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)


## General Information
- Deep Learning implementation in R
- Dataset is public on Kaggle [1]: https://www.kaggle.com/datasets/wanderdust/skin-lesion-analysis-toward-melanoma-detection
- Train a Random Forest model as a baseline model
- Train a Neuronal Net model from scratch
- Evaluating the performance
- Identify the images which are predicted badly

## Conclusion
* Performance of Deep Learning
  * Our CNN models are performing worse than the Random forest ones -> possibly due to too many parameters which could been opimized
  * Are the current neuronal networks fitting the local minima? Maybe the learning rate is too big.
  * We could tried to use more epochs for getting smaller learning rate.
  * Maybe we should have added more hidden layers to receive more neurons.
  * Maybe we should have used a different optimizer than "adam" to compile the model? "adam" is an optimized algorithm instead of classical  SGD as stocahstic gradient descent, that can work with non-convex optimization problems (thus local minima instead of global minima?)
  
* Main issues
  * Image quality not satisfying
    * Different image resolutions
    * Different depth of color
    * Some images have artificial markups (from the physicians marking the melanoma), which could influence our model
  * Imbalanced data
    * The amount of image of the different labels (nevus, melanoma and seborrheic keratosis) is different
    * We can compensate this with a weighted or balanced random forest, or when using Tensorflow with class weighting or oversampling
  * Overfitting of CNN when training from scratch
    * Data augmentation was not possible with our R implementation
    * `could not find function "layer_random_flip"` this is most probably due to a version conflict

## Setup
### Technologies Used
- Kaggle notebooks & compute
- R
- Tensorflow (Version 2.6.5)
- Keras (Version 2.6)

### Run on Kaggle
1) The notebook `dl-melanona-detection-dataset.ipynb` can be downloaded and imported into Kaggle, or you can choose to import the notebook an Kaggle)
2) Set Language to `R` on Kaggle
3) Import the dataset into your Kaggle notebook
  * With "Add Data button"
  * Add the URL `https://www.kaggle.com/datasets/wanderdust/skin-lesion-analysis-toward-melanoma-detection`
4) Run the notebook cell by cell, or all at once


## Acknowledgements
Used and inspired by the following resources
* https://www.kaggle.com/datasets/wanderdust/skin-lesion-analysis-toward-melanoma-detection/code. 
* https://tensorchiefs.github.io/dl_rcourse_2022/.
* https://tensorflow.rstudio.com/guides/keras/preprocessing_layers.
* https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-21c14f7a716e

## Contact
Urs Mayr (mayru@student.ethz.ch)
Vera Jäggi (vjaeggi@student.ethz.ch)
Silvan Hostettler (shostettle@student.ethz.ch)
