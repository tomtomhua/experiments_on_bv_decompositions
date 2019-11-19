# Readme
HI there!<br>
This is a file about my undergraduate graduation project.
I change a lot of parameters and other related influences (train/test data size, etc.) to figure out what kinds of factors will have a significant changes on bias and variance of a classifier.
Here I generally use two relatively new machine learning model as my classifier: lightgbm and extratrees, and use PMLB dataset(which contains 166 different kinds of datasets, with various sample sizes and feature numbers).

Some interesting results:

* Under the extreme randomized tree algorithm, the sample rate has significant influence on bias, variance and loss function; the robustness of bias is good; the best parameter method can essentially decrease the variance; and for most cases changing one parameter has a significant influence on bias, variance and loss function.
* Under LightGBM algorithm, the sample rate has significant influence on bias, variance and loss function; the robustness of variance is good; the best parameter method can essentially decrease the bias; and no other parameters except min_sample_per_leaf has a significant influence on bias, variance and loss function.
