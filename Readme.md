MyML
==============

**My (*Yet Another*) Machine Learning library** in Python by Javier "Analytic Bastard"
Arriero-Pais

The primary intention of this library is improving my understanding of algorithms
that learn from data and my Python skills.

Secondary goals of this project are integration of the techniques implemented
here into the broadly-used, well-developed Sklearn and having a collection of
not-so-standard methods in ML such as unsupervised method, with special
emphasis on Kernel methods.


Currently implemented features:
--------------

- Gradient Kernel-based Dimension Reduction

  [1] K. Fukumizu, C. Leng - Gradient-based kernel method for feature 
      extraction and variable selection. NIPS 2012.
       
  
- Gradient descent with fixed learning rate

  [1] Stanford CS 229 Lecture Notes (Logistic Regression), Andrew Ng
  

- Ordinary Least squares

  [1] Stanford CS 229 Lecture Notes (Logistic Regression), Andrew Ng


- Logistic Regression

  [1] Stanford CS 229 Lecture Notes (Logistic Regression), Andrew Ng
  

- Non-negative least squares

  [1] Lawson C., Hanson R.J., Solving Least Squares Problems, SIAM. 1987


- Sparse Non-negative matrix factorization

  [1] Nonnegative Matrix Factorization Based on Alternating Nonnegativity
      Constrained Least Squares and Active Set Method. Hyunsoo Kim and
      Haesun Park. SIAM Journal on Matrix Analysis and Applications, 30-2,
      2008
      


Usage
--------------
Currently no installer is included. This means that you will have to put it
under your current working directory or under the Python path yourself.

from myml.supervised import regression

ols  = regression.OLS()
gkdr = gkdr.GKDR()


from myml.supervised import classification

lr   = classification.LogisticRegression()




Watch comments on
--------------

[http://machinomics.blogspot.com](http://machinomics.blogspot.com)