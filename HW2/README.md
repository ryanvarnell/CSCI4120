# Team Members

Ryan Varnell\
varnellr18@students.ecu.edu

# Quick Start

As long as your environment meets the following dependencies, you should be able to run it just fine from its location
using:
> python3 hw2.py

### Dependencies:

Python >= 3.10\
SKLearn >= 0.0 (This is what pip tells me. I don't believe it's correct.)\
MatPlotLib >= 3.6.0\
YellowBrick >= 1.5\
NumPy >= 1.22.0\
SciPy >= 1.9.1\
Seaborn >= 0.12.0

# Which K Works the Best?

The elbow method determines that 3 is the "best" K-value in this instance. This is debatable.\
To the naked eye, 4 seems to be the obvious choice for the K-value. The difference in accuracies will be discussed in 
the next portion.

# The Best K Accuracy

The elbow-method-suggested K-value of 3 reliably lands at 75% accuracy. A K-value of 4, however, reliably lands at 100% 
accuracy.

![K Value 4 Confusion Matrix](/HW2/assets/confusionmatrix.png?raw=true "Best K Accuracy Confusion Matrix")