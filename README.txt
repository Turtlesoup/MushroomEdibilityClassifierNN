***************
**   About   **
***************

A simple NN implementation using TensorFlow to build a classifier to classify mushrooms into either edible or poisonous based on various properties.

This dataset was originally donated to the UCI Machine Learning repository on 27 April 1987. Modifications have been made to the original data to enumerate/seperate the properties and classes.

This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like "leaflets three, let it be'' for Poisonous Oak and Ivy.

**************
**   Data   **
**************

Number of training samples: 7000
Number of test samples: 1124

Training/Test data:

enumerated values corresponding to the different possible types for each of the following properties:

[cap-shape,cap-surface,cap-color,bruises,odor,gill-attachment,gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,stalk-surface-above-ring,stalk-surface-below-ring,stalk-color-above-ring,stalk-color-below-ring,veil-type,veil-color,ring-number,ring-type,spore-print-color,population,habitat]

Training/Test classes:

one-hot vector corresponding to the class of each data set in the training/test data respectively:

[edible,poisonous]

*******************
**    Results    **
*******************

Classifier was able to achieve an accuracy of 0.985765 when trained/tested on all 7000/1124 samples. Not terrible, but could certainly be improved by changing the NN parameters and increasing the number of training samples.