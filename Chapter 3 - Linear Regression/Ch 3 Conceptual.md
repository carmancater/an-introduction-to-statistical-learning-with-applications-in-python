# Chapter 3 Conceptual

## Problem 1

|             | Coefficient | Standard Error | t-statistic | p-value  |
| ----------- | ----------- | -------------- | ----------- | -------- |
| Intercept   | 2.939       | 0.3119         | 9.42        | < 0.0001 | 
| TV          | 0.046       | 0.0014         | 32.81       | < 0.0001 |
| Radio       | 0.189       | 0.0086         | 21.89       | < 0.0001 |
| Newspaper   | -0.001      | 0.0059         | -0.18       | 0.8599   |

Suppose the estimated regression equation is of the form: $\hat{y} = \hat{\beta_0} + \hat{\beta_1} \cdot TV + \hat{\beta_2} \cdot Radio + \hat{\beta_3} \cdot Newspaper$

- **Intercept**
    - $H_0: \beta_0 = 0$ (Expected value of sales is zero when no money is spent on advertising.)
    - $H_a: \beta_0 \neq 0$ (Expected value of sales is non-zero when no money is spent on advertising.)
    - Since $p < 0.0001$ is significant we reject the null hypothesis and conclude when no money is spent on advertising the expected sales is non-zero.
    
- **TV**
    - $H_0: \beta_1 = 0$ (There is no relationship between sales and TV.)
    - $H_a: \beta_1 \neq 0$ (There is some relationship between sales and TV.)
    - Since $p < 0.0001$ is significant we reject the null hypothesis and conclude in the precense of radio and newspaper advertising there is a relationship between sales and TV advertising.)
    
- **Radio**
    - $H_0: \beta_2 = 0$ (There is no relationship between sales and radio.)
    - $H_a: \beta_2 \neq 0$ (There is some relationship between sales and radio.)
    - Since $p < 0.0001$ is significant we reject the null hypothesis and conclude in the precense of TV and newspaper advertising there is a relationship between sales and radio advertising.
    
- **Newspaper**
    - $H_0: \beta_3 = 0$ (There is no relationship between sales and newspaper.)
    - $H_a: \beta_3 \neq 0$ (There is some relationship between sales and newspaper.)
    - Since $p = 0.8599$ is not significant in the precense of radio and TV advertising we do not reject the null hypothesis. There is not enough evidence to conclude there is a relationship between sales and newspaper advertising.)
    
## Problem 2
- **K-Nearest Neighbors (KNN)**
    - **KNN Regression**
        - Take a set of $n$ observation pairs { $(x_1, y_1), (x_2, y_2), \ldots ,(x_n, y_n)$ } with $y_i$ quantitative.  Given $K \in \mathbb Z_{> 0}$ and predicitor $x_0$, identify the $K$ nearest training observations (e.g., using Euclidean distance) represented by $\mathcal N_{0}$. The predicted response value is given by: $$\hat{f}(x_0) := \frac{1}{K}\sum_{i \in \mathcal N_{0}}y_i$$

    - **KNN Classifier**
        - Take a set of $n$ observation pairs { $(x_1, y_1), (x_2, y_2), \ldots ,(x_n, y_n)$ } with $y_i$ qualitative. Given $K \in \mathbb Z_{> 0}$ and predicitor $x_0$, identify the $K$ nearest training observations (e.g., using Euclidean distance) represented by $\mathcal N_{0}$. Estimate the conditional probability for class $j$ as the fraction of points in  $\mathcal N_{0}$ whose observed response values equal $j$. This is denoted by $$Pr(Y=j | X = x_0) := \frac{1}{K} \sum_{i \in \mathcal N_{0}}I(y_i = j)$$ where $I(y_i = j) :=0 \text{ if } y_i \neq j$ and $I(y_i = j) := 1 \text{ if } y_i = j$. KNN classifier classifies $x_0$ to the class with the largest probability.
        
## Problem 3
Let $X_1 = GPA$, $X_2 = IQ$, $X_3 = Level$ (1 for College and 0 for High School), $X_4 = GPA \cdot IQ$, and $X_5 = GPA \cdot Level$. Note $X_4$ and $X_5$ are interaction terms. Response variable is starting salary after graduation (in thousands of dollars). After performing a least squares fit we get $\hat{\beta_0} = 50$, $\hat{\beta_1} = 20$, $\hat{\beta_2} =0.07$, $\hat{\beta_3} = 35$, $\hat{\beta_4} = 0.01$, $\hat{\beta_5} = -10$.

$$\hat{y} = 50 + 20 \cdot GPA + 0.07 \cdot IQ + 35 \cdot Level + 0.01 \cdot (GPA \cdot IQ) - 10 \cdot (GPA \cdot Level)$$

**(a)** For a fixed value of $IQ$ and $GPA$, high school graduates earn more, on average, than college graduates provided that the $GPA$ is high enough. This can be seen by comparing:

$$\hat{y}_{HS} = 50 + 20 \cdot GPA + 0.07 \cdot IQ + 0.01 \cdot (GPA \cdot IQ)$$

$$\hat{y}_{Col} = 50 + 20 \cdot GPA + 0.07 \cdot IQ + 35 + 0.01 \cdot (GPA \cdot IQ) - 10 \cdot GPA$$

and noticing if $GPA > 3.5$ then $\hat{y_{Col}} < \hat{y_{HS}}$.

**(b)** For a college graduate with an IQ of 110 and GPA of 4.0 we predict their salary to be <span>$</span>137,100 $$\hat{y} = 50 + 20(4) + 0.07(110) + 35(1) + 0.01(4)(110) - 10(4) = 137.1$$
    
**(c)** False. Even though the coefficient for the GPA/IQ interaction term is very small it could still be significant with a small standard error. 
    
## Problem 4
Take a set of data with $n=100$ observations containing a single predictor and quantitative response. Suppose a linear regression model is fit as well as a seperate cubic regression, i.e., $Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \beta_3 X^3 + \epsilon$.
    
**(a)** If the true relationship between $X$ and $Y$ is linear we would expect the training residual sum of squares (RSS) for the cubic regression to be smaller since the curve will better fit the training data.
    
**(b)** If the true relationship between $X$ and $Y$ is linear we would expect the test RSS to be smaller for the linear regression model since the cubic model will likely overfit the data.
    
**(c)** If the true relationship between $X$ and $Y$ is not linear we would expect the training RSS for the cubic model to be lower than the linear model since it will better fit the curvature of the observations.
    
**(d)** If the true relationship between $X$ and $Y$ is not linear there is not enough information to tell whether the test RSS will be higher or lower for the linear model versus cubic model.
    
## Problem 5
Suppose we perform simple linear regression without an intercept and translate our data set so $\bar{x} = \bar{y} = 0$. Our resulting equation will have the form $\hat{y_i} = x_i \hat{\beta}$ with $\hat{\beta}$ given by $$\hat{\beta} = \frac{\sum\limits_{i=1}^{n} x_i y_i}{\sum\limits_{i'=1}^{n} x_{i'}^2}$$

We show we can write $\hat{y_i} = \sum\limits_{i'=1}^n a_{i'}y_{i'}$ with $a_{i'}$ given below.
    
$$\hat{y_i} = x_i\hat{\beta} = \frac{\sum\limits_{i'=1}^n x_i x_{i'} y_{i'}}{\sum\limits_{i'=1}^n x_{i'}^2} = \sum\limits_{i'=1}^n a_{i'}y_{i'} \text{ with } a_{i'} = \frac{x_i x_{i'}}{\sum\limits_{j=1}^n x_{j}^2}$$
    
## Problem 6
In the case of simple linear regression the least squares line always passes through the point $(\bar{x}, \bar{y})$. Take the least squares line to be $\hat{y} = \hat{\beta_0} + \hat{\beta_1}x$ with the known formulas for $\hat{\beta_0}$ and $\hat{\beta_1}$:
$$\hat{\beta_1} = \frac{\sum\limits_{i=1}^{n} (x_i - \bar{x}) (y_i - \bar{y})}{\sum\limits_{i=1}^{n} (x_{i} - \bar{x})^2} \text{ , } \hat{\beta_0} = \bar{y} - \hat{\beta_1}\bar{x}$$

Substituting $\bar{x}$ into the regression equation:
$$\hat{\beta_0} + \hat{\beta_1}\bar{x} = \bar{y} - \hat{\beta_1}\bar{x} + \hat{\beta_1}\bar{x} = \bar{y}$$

This completes the proof.

## Problem 7
In the case of simple linear regression of Y onto X, the $R^2$ statistic is equal to the square of the correlation between X and Y. For simplicity one may assum $\bar{x} = \bar{y} = 0$.

$$Cor(X, Y) = \frac{\sum\limits_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum\limits_{i=1}^n (x_i - \bar{x})^2}\sqrt{\sum\limits_{i=1}^n (y_i - \bar{y})^2}}\bigg\vert_{\bar{x} = \bar{y} = 0} = \frac{\sum\limits_{i=1}^n x_i y_i}{\sqrt{\sum\limits_{i=1}^n x_i^2}\sqrt{\sum\limits_{i=1}^n y_i^2}}$$
    
From the equations in Problem 6 we see
    
$$\hat{\beta_1}\bigg\vert_{\bar{x} = \bar{y} = 0} = \frac{\sum\limits_{i=1}^{n} x_i y_i}{\sum\limits_{i=1}^{n} x_{i}^2} \text{ ,and } \hat{\beta_0}\bigg\vert_{\bar{x} = \bar{y} = 0} = 0$$
    
giving us 
    
$$\hat{y_i} = \hat{\beta_1}x_i = \frac{\sum\limits_{j=1}^{n} x_j y_j}{\sum\limits_{j=1}^{n} x_{j}^2} x_i$$
    
Plugging this in to $R^2$ we see
    
$$R^2 = \frac{TSS - RSS}{TSS} = 1 - \frac{\sum\limits_{i=1}^n(y_i - \hat{y_i})^2}{\sum\limits_{i=1}^n(y_i - \bar{y})^2}\bigg\vert_{\bar{x} = \bar{y} = 0} = \cdots = \frac{2\sum\limits_{i=1}^n y_i \hat{y_i} - \sum\limits_{i=1}^n \hat{y_i}^2}{\sum\limits_{i=1}^n y_i^2}$$
    
which has been rearranged by expanding the square and performing the subtraction. Substituting in for $\hat{y_i}$ and factoring out some terms we see
    
$$R^2 = \frac{\big(\sum\limits_{j=1}^n x_j y_j\big)^2}{\sum\limits_{j=1}^n x_j^2 \sum\limits_{i=1}^n y_i^2}\Bigg[2 \frac{\sum\limits_{i=1}^{n} x_i y_i}{\sum\limits_{j=1}^{n} x_j y_j} - \frac{\sum\limits_{i=1}^n x_i^2}{\sum\limits_{j=1}^n x_j^2}\Bigg] = \frac{\big(\sum\limits_{j=1}^n x_j y_j\big)^2}{\sum\limits_{j=1}^n x_j^2 \sum\limits_{i=1}^n y_i^2} = Cor(X, Y)^2$$
    
This completes the proof.


