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

Recall the quadratic discriminant model (QDA) assumes the observations in each class come from a normal distribution with class specific mean vector and covariance matrix. Take the case $p=1$ (one predictor) and $K$ classes where observations from the $k$th class come from a one-dimensional normal distribution, i.e. $X \sim N(\mu_k, \sigma_k^2)$. The density function for the one-dimensional normal distribution is given by:

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