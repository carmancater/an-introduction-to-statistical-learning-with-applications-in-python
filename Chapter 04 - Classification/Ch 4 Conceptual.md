# Chapter 4 Conceptual

## Problem 1

Proof that the logistic function representation and logit representation for the logistic regression model are the same.

$$P(x) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}}$$

Note that $$1 - P(x) = \frac{1}{1 + e^{\beta_0 + \beta_1 X}}$$

Multiplying both sides of $P(x)$ by $\frac{1}{1 - P(x)}$ yields the desired result

$$\frac{P(x)}{1-P(x)} = e^{\beta_0 + \beta_1 X}$$

## Problem 2

Linear discriminant analysis for $p=1$. Assume observations in the $k^{th}$ class are taken from a normal distribution $N(\mu_k, \sigma^2)$, the Bayes classifier assigns an observation to the class which maximizes the discriminant function.

In other words maximizing:

$$p_k(x) = \frac{\pi_k\frac{1}{\sqrt{2\pi}\sigma}e^{\frac{-1}{2\sigma^2}}(x - \mu_k)^2}{\sum\limits_{l=1}^{K} \pi_l \frac{1}{\sqrt{2\pi}\sigma}e^{\frac{-1}{2\sigma^2}}(x - \mu_l)^2}$$

is equivalent to maximizing:

$$\delta_k(x) = x\cdot\frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + \log(\pi_k)$$

Proof

