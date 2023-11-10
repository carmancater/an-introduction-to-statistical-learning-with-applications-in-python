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

Taking the natural logarithm of both sides of the first equation yields:

$$\log p_k(x) = \frac{\log\pi_k + \log\frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{2\sigma^2}(x - \mu_k)^2}{\log(\sum\limits_{l=1}^{K} \pi_l \frac{1}{\sqrt{2\pi}\sigma}e^{\frac{-1}{2\sigma^2}}(x - \mu_l)^2)}$$

If $x$ is fixed, we simply vary $k$ and choose the maximum. This allows us to drop all terms that do not involve $k$. In particular the denominator goes away, and the $\log$ constant term in the numerator goes away. Expanding the quadratic term in the numerator and rearranging yields the desired result.

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

Dividing through both sides by 0.5 and multiplying both sides by the denominator of the left hand side, and simplifying yields:

$$e^{-6 + 0.05X_1 + 3.5} = 1$$

This implies $-6 + 0.05X_1 + 3.5 = 0$ giving us $X_1 = 50$ hours of study required.

## Problem 7

We predict whether a given stock will issue a dividend ("Yes", "No") given last year's percent profit $X$. Suppose $\bar{X_{Yes}} = 10$, $\bar{X_{No}} = 0$, $\hat{\sigma^2_{Yes}} = \hat{\sigma^2_{No}} = 36$ and $80$% of the companies issued dividends (i.e. $\pi_{Yes} = 0.8$, $\pi_{No} = 0.2$). Assuming $X$ follows a normal distribution, predict whether a company will issue a dividend given last years percent profit was $X=4$.

**Solution**

We will use LDA classifier since the observations in each class come from a Gaussian (normal) distribution with class specific mean and common variance.

Recall the density function for a normal random variable is

$$f(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2 / 2\sigma^2}$$ 

Combining this with Bayes' Theorem

$$Pr(Y=k | X=x) = \frac{\pi_k f_k(x)}{\sum\limits_{l=1}^{K} \pi_l f_l(x)}$$

yields

$$P_{Yes}(4) = \frac{(0.8)\frac{1}{\sqrt{2\pi}(6)}e^{\frac{-1}{2(36)}(4-10)^2}}{(0.8)\frac{1}{\sqrt{2\pi}(6)}e^{\frac{-1}{2(36)}(4-10)^2} + (0.2)\frac{1}{\sqrt{2\pi}(6)}e^{\frac{-1}{2(36)}(4-0)^2}} \approx 0.7519$$

Thus we predict there is a $75.19$% chance of this company issuing a dividend.

## Problem 8

Take a data set, divide it into equally-sized training and test sets. Suppose we fit a logistic regression model with training error rate $20$%, test error rate $30$%. Now fit a KNN model with $K=1$ and suppose the average error rate (over both train and test sets) is $18$%. Which method should we prefer to use for classification of new data?

**Solution** For $K=1$, the KNN training error rate will be $0$% since for each training observation, we simply assign it to the same class given for that observation. Since in the problem the average error rate is $18$%, this implies the test error rate was $36$%. Since we want to use the model with the smaller test error rate, we choose to go with the logistic regression model.

## Problem 9

Recall that

$$odds = \frac{P(X)}{1 - P(X)}$$ 

giving us

$$P(X) = \frac{odds}{1 + odds}$$

**(a)** On average, the fraction of people with an odds of $0.37$ of defaulting on their credit card payment that will in fact default is:

$$ P(X) = \frac{0.37}{1 + 0.37} = \frac{37}{137}$$

**(b)** The odds of daulting of an individual who has a $16$% chance of defaulting on their credit card payment is:

$$odds = \frac{0.16}{1-0.16} = \frac{4}{21}$$

## Problem 10

We derive an expression for 

$$\log\big(\frac{Pr(Y=k | X=x)}{Pr(Y=K | X=x)}\big)$$

in the case $p = 1$ for LDA.

**Solution**

$$\log\big(\frac{Pr(Y=k | X=x)}{Pr(Y=K | X=x)}\big) = \log\big(\frac{\pi_k f_k(x)}{\pi_K f_K(x)}\big)$$

with $f_k, f_K$ Gaussian density functions.

$$= \log\big(\frac{\pi_k e^{\frac{-1}{2\sigma^2}(x-\mu_k)^2}}{\pi_K e^{\frac{-1}{2\sigma^2}(x-\mu_K)^2}}\big) = \log\big(\frac{\pi_k}{\pi_K}\big) + \frac{1}{2\sigma^2}\big(-2x\mu_K + \mu_K^2 + 2x\mu_k - \mu_k^2\big)$$

$$= \log\big(\frac{\pi_k}{\pi_K}\big) + \frac{\mu_K^2 - \mu_k^2}{2\sigma^2} + \frac{\mu_k - \mu_K}{\sigma^2}x = a_k + b_k x$$

where $a_k = \log\big(\frac{\pi_k}{\pi_K}\big) + \frac{\mu_K^2 - \mu_k^2}{2\sigma^2}$ and $b_k =  \frac{\mu_k - \mu_K}{\sigma^2}$ 

## Problem 11

We work out the detailed form of the log odds of the posterior probabilities for QDA.

**Solution**

Recall for the $k^{th}$ class the multivariate Gaussian density function is defined as

$$f_k(x) = \frac{1}{(2\pi)^{\frac{p}{2}} |\sum_k|^{\frac{1}{2}}}e^{\frac{-1}{2}(x-\mu_k)^T {\sum_k}^{-1}(x-\mu_k)}$$

Plugging this into the log odds:

$$\log\big(\frac{Pr(Y=k | X=x)}{Pr(Y=K | X=x)}\big) = \log\big(\frac{\pi_k f_k(x)}{\pi_K f_K(x)}\big)$$


