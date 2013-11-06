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

*Supervised methods*


- Naive Bayes binary classifier for *discrete* input data implemented with MapReduce

  [1] Author notes
  
- Gradient descent and stochastic gradient descent

  [1] Stanford CS 229 Lecture Notes (Logistic Regression), Andrew Ng
  

- Ordinary Least squares

  [1] Stanford CS 229 Lecture Notes (Logistic Regression), Andrew Ng


- Logistic Regression

  [1] Stanford CS 229 Lecture Notes (Logistic Regression), Andrew Ng
  

- Non-negative least squares

  [1] Lawson C., Hanson R.J., Solving Least Squares Problems, SIAM. 1987


- Gradient Kernel-based Dimension Reduction

  [1] K. Fukumizu, C. Leng - Gradient-based kernel method for feature 
      extraction and variable selection. NIPS 2012.
       

*Unsupervised methods*

- KMeans

  [1] Wikipedia: [http://en.wikipedia.org/wiki/K-means_clustering](http://en.wikipedia.org/wiki/K-means_clustering)
  

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




Planned features
--------------

- Gaussianization

  [1] Chen, Scott Shaobing, and Ramesh A. Gopinath. "Gaussianization." (2000).

- Sparse CCA

  [1] Sparse Canonical Correlation Analysis
	  David R. Hardoon and John Shawe-Taylor, Machine Learning Journal,
	  Volume 83 (3), Pages 331-353, 2011

- Probabilistic PCA

  [1] Probabilistic principal component analysis
      ME Tipping, CM Bishop. Journal of the Royal Statistical Society, 1999

- Sparse Filtering

  [1] Ngiam, Jiquan, et al. "Sparse filtering."
      Advances in Neural Information Processing Systems. 2011.

- Natural Gradient Descent

  [1] Amari, Shun-ichi, Andrzej Cichocki, and Howard Hua Yang.
      "A new learning algorithm for blind signal separation."
      Advances in neural information processing systems (1996): 757-763.
      
  [2] Amari, Shun-Ichi. "Natural gradient works efficiently in learning."
      Neural computation 10.2 (1998): 251-276.



Watch comments on
--------------

[http://machinomics.blogspot.com](http://machinomics.blogspot.com)
