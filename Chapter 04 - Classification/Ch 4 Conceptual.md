# Chapter 4 Conceptual

## Problem 1

**Proof** that the logistic function representation and logit representation for the logistic regression model are the same.

$$P(x) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}}$$

Note that $$1 - P(x) = \frac{1}{1 + e^{\beta_0 + \beta_1 X}}$$

Multiplying both sides of $P(x)$ by $\frac{1}{1 - P(x)}$ yields the desired result

$$\frac{P(x)}{1-P(x)} = e^{\beta_0 + \beta_1 X}$$

## Problem 2

Linear discriminant analysis for $p=1$. Assume observations in the $k^{th}$ class are taken from a normal distribution $N(\mu_k, \sigma^2)$, the Bayes classifier assigns an observation to the class which maximizes the discriminant function.

In other words maximizing:

$$p_k(x) = \frac{\pi_k\frac{1}{\sqrt{2\pi}\sigma}e^{\frac{-1}{2\sigma^2}}(x - \mu_k)^2}{\sum\limits_{l=1}^{K} \pi_l \frac{1}{\sqrt{2\pi}\sigma}e^{\frac{-1}{2\sigma^2}}(x - \mu_l)^2}$$

is equivalent to maximizing:

$$\delta_k(x) = x\cdot\frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + \log\pi_k$$

**Proof**

Taking the natural logarithm of both sides of the first equation yeilds:

$$\log p_k(x) = \frac{\log\pi_k + \log\frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{2\sigma^2}(x - \mu_k)^2}{\log(\sum\limits_{l=1}^{K} \pi_l \frac{1}{\sqrt{2\pi}\sigma}e^{\frac{-1}{2\sigma^2}}(x - \mu_l)^2)}$$

If $x$ is fixed, we simply vary $k$ and choose the maximum. This allows us to drop all terms that do not involve $k$. In particular the denominator goes away, and the $\log$ constant term in the numerator goes away. Expanding the quadratic term in the numerator and rearranging yeilds the desired result.

## Problem 3

Recall the quadratic discriminant analysis model (QDA) assumes the observations in each class come from a normal distribution with class specific mean vector and covariance matrix. Take the case $p=1$ (one predictor) and $K$ classes where observations from the $k$th class come from a one-dimensional normal distribution, i.e. $X \sim N(\mu_k, \sigma_k^2)$. The density function for the one-dimensional normal distribution is given by:

$$f_k(x) = \frac{1}{\sqrt{2\pi}\sigma_k}e^{\frac{-1}{2\sigma_k^2}}(x - \mu_k)^2$$

We prove the Bayes classifier in this case is quadratic in $x$

**Proof**

The Bayes classifier assigns the test observation to the class with the highest conditional probability, given by:

$$p_k(x) = \frac{\pi_k f_k(x)}{\sum\limits_{l=1}^{K} \pi_l f_l(x)}$$

Substituting in $f_k$ we get:

$$p_k(x) = \frac{\pi_k\frac{1}{\sqrt{2\pi}\sigma_k}e^{\frac{-1}{2\sigma_k^2}}(x - \mu_k)^2}{\sum\limits_{l=1}^{K} \pi_l \frac{1}{\sqrt{2\pi}\sigma_l}e^{\frac{-1}{2\sigma_l^2}}(x - \mu_l)^2}$$

Maximizing this is equivalent to maximizing the logarithm of both sides:

$$\log p_k(x) = \log \pi_k + \log \frac{1}{\sqrt{2\pi}\sigma_k} - \frac{1}{2\sigma_k^2}(x - \mu_k)^2 - \log(\sum\limits_{l=1}^{K} \pi_l \frac{1}{\sqrt{2\pi}\sigma_l}e^{\frac{-1}{2\sigma_l^2}}(x - \mu_l)^2)$$

The logarithm at the end with the summation can be dropped as it is a constant, for fixed $x$ we only care about terms dependent on $k$.

Expanding the right hand side we arrive at our discriminant function:

$$\delta_k(x) = \frac{-1}{2\sigma_k^2}x^2 + \frac{\mu_k}{\sigma_k^2}x - \frac{\mu_k^2}{2\sigma_k^2} - \frac{1}{2}\log\sigma_k^2 + \log\pi_k$$

Therefore the Bayes Classifier is qudratic in $x$.

## Problem 4

For $p$ predictors large we illustrate how local approaches to prediction such as KNN deteriorate. This is known as the curse of dimensionality.

**(a)** Let $X$ ($p =1$) be a dataset uniformly distributed on $[0,1]$. Using a local approach to prediction, say we want to use observations within $10$% of the range of $X$ closest to an observation. On average we will use $$\frac{1}{10}$$ of the available observations to make a prediction.

**(b)** Let $p=2$, $X_1, X_2$ uniformly distributed on $[0,1]\times[0,1]$. Say we want to make predictions using values within $10$% of the range of $X_1$ *and* within $10$% of the range of $X_2$ closest to a test observation. On average we will use $$\frac{1}{10^2} = \frac{1}{100}$$ of the available observations to make a prediction.

**(c)** Let $p=100$ features with the same setup as above. On average we will use $$\frac{1}{10^{100}}$$ of the available observations to make a prediction.

**(d)** For KNN, we see as $p$ grows, the fraction of training observations near any given test observation gets smaller and smaller.

**(e)** Say we wish to make a prediction for a test observation by creating a $p$-dimensional hypercube centered around the observation which contains $10$% of the training observations. We explore the side length of the hypercube for $p=1,2,100$.

Let $x$ be the side length.

- $p=1$: $x = 0.1$
- $p=2$: $x^2 = 0.1$ so $x = \sqrt{0.1} \approx 0.32$
- $p=100$: $x^{100} = 0.1$ so $x = \sqrt[100]{0.1} \approx 0.98$

We see as the dimension grows, in order to capture just $10$% of the training observations with a hypercube centered around our test point, the side length of the hypercube stretches almost to the boundary in every direction.

## Problem 5

Consider Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA).

**(a)** If the Bayes decision boundary is linear we expect QDA to perform better on the training set since it is more flexible of a fit. On the test set we expect LDA to perform better in general since QDA may overfit the noise in the training set.

**(b)** If the Bayes decision boundary is non-linear we expect QDA to perform better on the training set since it is more flexible of a fit. QDA will also perform better on the test set since it can more naturally fit the curvature present in the population.

**(c)** In general as the sample size $n$ increases we expect test prediction accuracy of QDA to improve relative to LDA.

**(d)** False. If the Bayes decision boundary is linear, even though QDA performs better on training data, it may overfit noise and thus perform worse on the test data.

## Problem 6

Take a fictional class of students. Let $X_1 =$ hours studied, $X_2 =$ undergrad GPA, and $Y =$ receive an A. Suppose we fit a model using logistic regression and produce $\hat{\beta_0} = -6$, $\hat{\beta_1} = 0.05$, $\hat{\beta_2} = 1$.

**(a)** Probability a student receives an A given 40 hours of study and undergrad GPA 3.5

$$P(40, 3.5) = \frac{e^{-6 + 0.05(40) + 3.5}}{1 + e^{-6 + 0.05(40) + 3.5}} \approx 0.3775$$

There is a 37.75% chance.

**(b)** Given an undergrad GPA of 3.5, the number of hours needed to study to have a 50% chance of receiving an A is:

$$\frac{e^{-6 + 0.05X_1 + 3.5}}{1 + e^{-6 + 0.05X_1 + 3.5}} = 0.5$$

Dividing the numerator and denominator of the left hand side by $e^{-6 + 0.05X_1 + 3.5}$ yields:

$$\frac{1}{\frac{1}{e^{-6 + 0.05X_1 + 3.5}} + 1} = 0.5$$

Dividing through both sides by 0.5 and multiplying both sides by the denominator of the left hand side, and simplifying yeilds:

$$e^{-6 + 0.05X_1 + 3.5} = 1$$

This implies $-6 + 0.05X_1 + 3.5 = 0$ giving us $X_1 = 50$ hours of study required.

## Problem 7