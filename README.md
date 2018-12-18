# Mcue
Mcue-surface-defect-inspection

## Introduction
This reporsitory stored part of the code for [Surface defect saliency of magnetic tile](https://link.springer.com/article/10.1007%2Fs00371-018-1588-5). The following code include the:
* MCueSal2.py - The saliency algorithms mentioned in papers, and MCue
* model.py - The U-Net like neural network built with keras
* Train.py - The main code for model training. Including saving the best model for during every validation.
* Test.py - The main code for testing. Generate the output for our model, and export the image to local disks. 
