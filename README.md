# Kernel Methods Project

This is the code produced by LALANNE Clément and WILLEMS Lucas for the Kaggle challenge of the ["Kernel Methods for Machine Learning"](https://www.kaggle.com/c/kernel-methods-for-machine-learning-2018-2019) course.

## To reproduce our submission

Execute:

```
python3 main.py
```

## Project structure

-```data_handling.py```: A few procedures to load data, train classifiers and write test predictions.

-```svm.py```: A svm solver using CVXOPT.

-```spectrum.py```: The code for spectrum kernels.

-```kernel_combin.py```: A way to combine kernels additively. 

-```main.py```: The script to run to produce the test fine.

**Note:** the results are written in ```data/Yte.csv```.
