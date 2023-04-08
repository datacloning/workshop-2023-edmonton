Statistical and computational preliminaries
================

## Introduction

In the first part of the course, we will go over some statistical
preliminaries and corresponding computational aspects. We will learn:

1.  To write down a likelihood function
2.  Meaning of the likelihood function
3.  Meaning of the Maximum likelihood estimator, difference between a
    parameter, an estimator and an estimate
4.  Good properties of an estimator: Consistency, Asymptotic normality,
    Unbiasedness, low Mean squared error
5.  Frequentist paradigm and quantification of uncertainty
6.  How to use Fisher information for an approximate quantification of
    uncertainty
7.  Motivation for the Bayesian paradigm
8.  Meaning of the prior distribution
9.  Derivation and meaning of the posterior distribution
10. Interpretation of a credible interval and a confidence interval
11. Scope of inference: Should I specify the hypothetical experiment or
    should I specify the prior distribution? Each one comes with its own
    scope of inference.

## Occupancy studies

Let us start with an occupancy study. Suppose we have a site that is
being considered for development. There is a species of interest that
might get affected by this development. Hence we need to study what
proportion of the area is occupied by the species of interest. If this
proportion is not very large, we may go ahead with the development.

Suppose we divide the site in several equal-area cells. Suppose all
cells have similar habitats (*identical*). Further we assume that
occupancy of one cell does not affect occupancy of other quadrats
(*independence*). Let ![N](https://latex.codecogs.com/svg.image?N "N")
be the total number of cells.

Let ![Y_i](https://latex.codecogs.com/svg.image?Y_i "Y_i") be the
occupancy status of the *i*-th quadrat. This is unknown and hence is a
random variable. It takes values in (0,1), 0 meaning unoccupied and 1
meaning occupied. This is a *Bernoulli* random variable. We denote this
by
![Y\_{i}\sim Bernoulli(\phi)](https://latex.codecogs.com/svg.image?Y_%7Bi%7D%5Csim%20Bernoulli%28%5Cphi%29 "Y_{i}\sim Bernoulli(\phi)").
The random variable ![Y](https://latex.codecogs.com/svg.image?Y "Y")
takes value 1 with probability
![\phi](https://latex.codecogs.com/svg.image?%5Cphi "\phi"). This is the
probability of occupancy. The value of
![\phi](https://latex.codecogs.com/svg.image?%5Cphi "\phi") is unknown.
This is the parameter of the distribution.

Suppose we visit ![n](https://latex.codecogs.com/svg.image?n "n"), a
subset, of these cells. These are selected using simple random sampling
without replacement. The observations are denoted by
![y_1,y_2,y_3,...,y_n](https://latex.codecogs.com/svg.image?y_1%2Cy_2%2Cy_3%2C...%2Cy_n "y_1,y_2,y_3,...,y_n").
We can use these to infer about the unknown parameter
![\phi](https://latex.codecogs.com/svg.image?%5Cphi "\phi"). The main
tool for such inductive inference (data to population and *not*
hypothesis to prediction) is the *likelihood function*.

### The likelihood function

Suppose the observed data are (0,1,1,0,0). Then we can compute the
probabiity of observing these data under various values of the parameter
![\phi](https://latex.codecogs.com/svg.image?%5Cphi "\phi") (assuming
independent, identically distributed Bernoulli random variables). It can
be written as:
![L(\phi;y\_{1},y\_{2},...,y\_{n})={\prod}P(y\_{i};\phi)={\prod}\phi^{y\_{i}}(1-\phi)^{1-y\_{i}}](https://latex.codecogs.com/svg.image?L%28%5Cphi%3By_%7B1%7D%2Cy_%7B2%7D%2C...%2Cy_%7Bn%7D%29%3D%7B%5Cprod%7DP%28y_%7Bi%7D%3B%5Cphi%29%3D%7B%5Cprod%7D%5Cphi%5E%7By_%7Bi%7D%7D%281-%5Cphi%29%5E%7B1-y_%7Bi%7D%7D "L(\phi;y_{1},y_{2},...,y_{n})={\prod}P(y_{i};\phi)={\prod}\phi^{y_{i}}(1-\phi)^{1-y_{i}}")

Notice that this is a function of the parameter
![\phi](https://latex.codecogs.com/svg.image?%5Cphi "\phi") and the data
are fixed. The likelihood function or equivalently the log-likelihood
function quantifies the *relative* support for different values of the
parameters. Hence only the likelihood ratio function is meaningful.

### Maximum likelihood estimator

A natural approach to estimation (inference) of
![\phi](https://latex.codecogs.com/svg.image?%5Cphi "\phi") is to choose
the value that is better supported than any other value in the parameter
space
![(0,1)](https://latex.codecogs.com/svg.image?%280%2C1%29 "(0,1)"). This
is called the maximum likelihood estimator. We can show that this turns
out to be:
![\hat{\phi}=\frac{1}{n}\sum y\_{i}](https://latex.codecogs.com/svg.image?%5Chat%7B%5Cphi%7D%3D%5Cfrac%7B1%7D%7Bn%7D%5Csum%20y_%7Bi%7D "\hat{\phi}=\frac{1}{n}\sum y_{i}")
This is called an *estimate*. This is a fixed quantity because the data
are observed and hence not random.

### Quantification of uncertainty

As scientists, would you stop at reporting this? I suspect not. If this
estimate is large, say 0.85, the developer is going to say ‘you just got
lucky (or, worse, you cheated) with your particular sample’. A natural
question to ask then is ‘how different this estimate would have been if
someone else had conducted the experiment?’. In this case, the
‘experiment to be repeated’ is fairly uncontroversial. We take another
simple random sample without replacement from the study area. However,
that is not always the case as we will see when we deal with the
regression model.

The sampling distribution is the distribution of the estimates that one
would have obtained had one conducted these replicate experiments. It is
possible to get an approximation to this sampling distribution in a very
general fashion if we use the method of maximum likelihood estimator. In
many situations, it can be shown that the sampling distribution is
![\hat{\phi}\sim N(\phi,\frac{1}{n}I^{-1}(\phi))](https://latex.codecogs.com/svg.image?%5Chat%7B%5Cphi%7D%5Csim%20N%28%5Cphi%2C%5Cfrac%7B1%7D%7Bn%7DI%5E%7B-1%7D%28%5Cphi%29%29 "\hat{\phi}\sim N(\phi,\frac{1}{n}I^{-1}(\phi))")
where
![I(\phi)=-\frac{1}{n}\sum\frac{d^2}{d^2\phi}logL(\phi;{y})](https://latex.codecogs.com/svg.image?I%28%5Cphi%29%3D-%5Cfrac%7B1%7D%7Bn%7D%5Csum%5Cfrac%7Bd%5E2%7D%7Bd%5E2%5Cphi%7DlogL%28%5Cphi%3B%7By%7D%29 "I(\phi)=-\frac{1}{n}\sum\frac{d^2}{d^2\phi}logL(\phi;{y})")

This is also called the Hessian matrix or the curvature matrix of the
log-likelihood function. The higher the curvature, the less variable are
the estimates from one experiment to other. Hence the resultant
‘estimate’ is considered highly reliable.

### 95% Confidence interval

This is just a set of values that covers the estimates from 95% of the
experiments. The experiments are not actually replicated and hence this
simply tells us what the various outcomes could be. Our decisions could
be based on this variation *as long as we all agree on the experiment
that could be replicated*. We are simply covering our bases against the
various outcomes and protect ourselves from future challenges. If we use
the maximum likelihood estimator, we can obtain this as:
![\hat{\phi}-\frac{1.96}{n}\sqrt{I^{-1}(\hat{\phi})},\hat{\phi}+\frac{1.96}{n}\sqrt{I^{-1}(\hat{\phi})}](https://latex.codecogs.com/svg.image?%5Chat%7B%5Cphi%7D-%5Cfrac%7B1.96%7D%7Bn%7D%5Csqrt%7BI%5E%7B-1%7D%28%5Chat%7B%5Cphi%7D%29%7D%2C%5Chat%7B%5Cphi%7D%2B%5Cfrac%7B1.96%7D%7Bn%7D%5Csqrt%7BI%5E%7B-1%7D%28%5Chat%7B%5Cphi%7D%29%7D "\hat{\phi}-\frac{1.96}{n}\sqrt{I^{-1}(\hat{\phi})},\hat{\phi}+\frac{1.96}{n}\sqrt{I^{-1}(\hat{\phi})}")
You will notice that as we increase the sample size, the width of this
interval converges to zero. That is, as we increase the sample size, the
MLE converges to the true parameter value. This is called the
‘consistency’ of an estimator. This is an essential property of any
statistical inferential procedure.

## Bayesian paradigm

All the above statements seem logical but fake at the same time! No one
repeats the same experiment (although replication consistency is an
essential scientific requirement). What if we have time series? We can
never replicate a time series. So then should we simply take the
estimated value *prima facie*? That also seems incorrect scientifically.
So where is the uncertainty in our mind coming from? According to the
Bayesian paradigm, it arises because of our ‘personal’ uncertainty about
the parameter values.

**Prior distribution**: Suppose we have some idea about what values of
occupancy are more likely than others *before* any data are collected.
This can be quantified as a probability distribution on the parameter
space (0,1). This distribution can be anything, unimodal or bimodal or
even multimodal! Let us denote this by
![\pi(\phi)](https://latex.codecogs.com/svg.image?%5Cpi%28%5Cphi%29 "\pi(\phi)").
How do we change this *after* we observe the data?

**Posterior distribution** This is the quantification of uncertainty
*after* we observe the data. Usually observing the data decreases our
uncertainty, although it is not guaranteed to be the case. The posterior
distribution is obtained by:
![\pi(\phi\|y)=\frac{L(\phi;y)\pi(\phi)}{\int L(\phi;d\phi y)\pi(\phi)}](https://latex.codecogs.com/svg.image?%5Cpi%28%5Cphi%7Cy%29%3D%5Cfrac%7BL%28%5Cphi%3By%29%5Cpi%28%5Cphi%29%7D%7B%5Cint%20L%28%5Cphi%3Bd%5Cphi%20y%29%5Cpi%28%5Cphi%29%7D "\pi(\phi|y)=\frac{L(\phi;y)\pi(\phi)}{\int L(\phi;d\phi y)\pi(\phi)}")

### Credible interval

This is obtained by using the percentiles of the posterior distribution.

Notice a few things here:

1.  This involves an integral in the denominator. Depending on how many
    parameters (unknowns) are in the model, this can be a large
    dimensional integral. Imagine a regression model with 5 covariates.
    This integral will be 6 dimensional (add one for the variance).
2.  Data are fixed. We do not need to replicate the experiment. The
    uncertainty is completely in the mind of the researcher.
3.  Different researchers might have different prior uncertainties. This
    will lead to different posterior uncertainties. Hence this is a
    subjective or personal quantification of uncertainty. It is not
    transferable from one researcher to another.

An interesting result follows. As we increase the samples size, the
Bayesian posterior, for ANY prior, converges to the distribution that
looks very much like the frequentist sampling distribution of the MLE.
That is,
![\pi(\phi\|y)\thickapprox N(\hat{\phi},\frac{1}{n}I^{-1}(\hat{\phi}))](https://latex.codecogs.com/svg.image?%5Cpi%28%5Cphi%7Cy%29%5Cthickapprox%20N%28%5Chat%7B%5Cphi%7D%2C%5Cfrac%7B1%7D%7Bn%7DI%5E%7B-1%7D%28%5Chat%7B%5Cphi%7D%29%29 "\pi(\phi|y)\thickapprox N(\hat{\phi},\frac{1}{n}I^{-1}(\hat{\phi}))")
There are subtle differences that we are going to ignore here.
Qualitatively, what this says is that for large sample size:

1.  Posterior mean and the MLE are similar
2.  Posterior variance is similar to the inverse of the Hessian matrix.

Hence credible interval and confidence intervals will be
indistinguishable for large sample size. Effect of the choice of the
prior distribution vanishes. How large a sample size should be for this
to happen? It depends on the number of parameters in the model and how
strong the prior distribution is.

## No math please!

> Bayesian and ML inference using MCMC and data cloning

We now show how one can compute the posterior distribution for any
choice of the prior distribution without analytically calculating the
integral in the denominator. We will generate the data under the
Bernoulli model. You can change the parameters as you wish when you run
the code.

Simulate a simple data set with 30 observations:

``` r
library(dclone)
```

    ## Loading required package: coda

    ## Loading required package: parallel

    ## Loading required package: Matrix

    ## dclone 2.3-1      2023-04-07

``` r
phi.true = 0.3
n = 30
Y = rbinom(n,1,phi.true)

table(Y)
```

    ## Y
    ##  0  1 
    ## 20 10

Analytical MLE:

``` r
(MLE.est = sum(Y)/n)
```

    ## [1] 0.3333333

### Bayesian inference

We will use the [JAGS](https://mcmc-jags.sourceforge.io/) program and
the [dclone](https://CRAN.R-project.org/package=dclone) R package.

First, we need to define the model function. This is the critical
component.

``` r
Occ.model = function(){
  # Likelihood 
  for (i in 1:n){
    Y[i] ~ dbin(phi_occ, 1)
  }
  # Prior
  phi_occ ~ dbeta(1, 1)
}
```

Second, we need to provide the data to the model and generate random
numbers from the posterior. We will discuss different options later.

``` r
dat = list(Y=Y, n=n)
Occ.Bayes = jags.fit(data=dat, params="phi_occ", model=Occ.model)
```

    ## Registered S3 method overwritten by 'R2WinBUGS':
    ##   method            from  
    ##   as.mcmc.list.bugs dclone

    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 30
    ##    Unobserved stochastic nodes: 1
    ##    Total graph size: 33
    ## 
    ## Initializing model

``` r
summary(Occ.Bayes)
```

    ## 
    ## Iterations = 2001:7000
    ## Thinning interval = 1 
    ## Number of chains = 3 
    ## Sample size per chain = 5000 
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##           Mean             SD       Naive SE Time-series SE 
    ##      0.3451933      0.0835788      0.0006824      0.0008805 
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##   2.5%    25%    50%    75%  97.5% 
    ## 0.1917 0.2856 0.3423 0.4011 0.5176

``` r
plot(Occ.Bayes)
```

![](README_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

The summary describes the posterior distribution: its mean, standard
deviation, and quantiles.

This was quite easy. Now we use data cloning to compute the MLE and its
variance using MCMC.

## Data cloning in a nutshell

As you all know, at least in this simple situation, we can write down
the likelihood function analytically. We can also use calculus and/or
numerical optimization such as the `optim()` function in R to get the
location of the maximum and its Hessian matrix. But suppose we do not
want to go through all of that and instead want to use the MCMC
algorithm. Why? Because it is easy and can be generalized to more
complex hierarchical models.

Earlier we noted that as we increase the sample size, the Bayesian
posterior converges to the sampling distribution of the MLE. We,
obviously, cannot increase the sample size. The data are given to us.
Data cloning conducts a computational trick to increase the sample size.
We clone the data!

Imagine a sequence of *K* independent researchers.

- Step 1: First researcher has data
  ![y_1,y_2,...,y_n](https://latex.codecogs.com/svg.image?y_1%2Cy_2%2C...%2Cy_n "y_1,y_2,...,y_n").
  They use their own prior and obtain the posterior distribution.
- Step 2: Second researcher goes out and gets their own data. It just so
  happens that they observed the same exact locations as the first
  researcher. Being a good Bayesian, they use the posterior of the first
  researcher as their prior (knowledge accumulation).
- Step K: The K-th researcher also obtains the same data but uses the
  posterior at the (K-1) step as their prior.

What is happening with these sequential posterior distributions?

**The posterior distribution is converging to a single point; a
degenerate distribution. This is identical to the MLE!**

1.  As we increase the number of clones, the mean of the posterior
    distributions converges to the MLE.
2.  The variance of the posterior distribution converges to 0.
3.  If we scale the posterior distribution with the number of clones
    (that is, multiply the posterior variance by the number of clones),
    it is identical to the inverse of the Fisher information matrix.

You can play with the number of clones and see the effect on the
posterior distribution using this [Shiny
app](https://psolymos.shinyapps.io/dcapps/) ([R code for the
app](../app/))

We do not need to implement this procedure sequentially. The matrix of
these K datasets is of dimension (*n*,*K*) with identical columns.

![\left\[\begin{array}{cccccccccc} y\_{1} & y\_{1} & y\_{1} & y\_{1} & y\_{1} & y\_{1} & y\_{1} & y\_{1} & y\_{1} & y\_{1}\\ y\_{2} & y\_{2} & y\_{2} & y\_{2} & y\_{2} & y\_{2} & y\_{2} & y\_{2} & y\_{2} & y\_{2}\\ y\_{3} & y\_{3} & y\_{3} & y\_{3} & y\_{3} & y\_{3} & y\_{3} & y\_{3} & y\_{3} & y\_{3}\\ y\_{4} & y\_{4} & y\_{4} & y\_{4} & y\_{4} & y\_{4} & y\_{4} & y\_{4} & y\_{4} & y\_{4}\\ y\_{5} & y\_{5} & y\_{5} & y\_{5} & y\_{5} & y\_{5} & y\_{5} & y\_{5} & y\_{5} & y\_{5} \end{array}\right\]](https://latex.codecogs.com/svg.image?%5Cleft%5B%5Cbegin%7Barray%7D%7Bcccccccccc%7D%20y_%7B1%7D%20%26%20y_%7B1%7D%20%26%20y_%7B1%7D%20%26%20y_%7B1%7D%20%26%20y_%7B1%7D%20%26%20y_%7B1%7D%20%26%20y_%7B1%7D%20%26%20y_%7B1%7D%20%26%20y_%7B1%7D%20%26%20y_%7B1%7D%5C%5C%20y_%7B2%7D%20%26%20y_%7B2%7D%20%26%20y_%7B2%7D%20%26%20y_%7B2%7D%20%26%20y_%7B2%7D%20%26%20y_%7B2%7D%20%26%20y_%7B2%7D%20%26%20y_%7B2%7D%20%26%20y_%7B2%7D%20%26%20y_%7B2%7D%5C%5C%20y_%7B3%7D%20%26%20y_%7B3%7D%20%26%20y_%7B3%7D%20%26%20y_%7B3%7D%20%26%20y_%7B3%7D%20%26%20y_%7B3%7D%20%26%20y_%7B3%7D%20%26%20y_%7B3%7D%20%26%20y_%7B3%7D%20%26%20y_%7B3%7D%5C%5C%20y_%7B4%7D%20%26%20y_%7B4%7D%20%26%20y_%7B4%7D%20%26%20y_%7B4%7D%20%26%20y_%7B4%7D%20%26%20y_%7B4%7D%20%26%20y_%7B4%7D%20%26%20y_%7B4%7D%20%26%20y_%7B4%7D%20%26%20y_%7B4%7D%5C%5C%20y_%7B5%7D%20%26%20y_%7B5%7D%20%26%20y_%7B5%7D%20%26%20y_%7B5%7D%20%26%20y_%7B5%7D%20%26%20y_%7B5%7D%20%26%20y_%7B5%7D%20%26%20y_%7B5%7D%20%26%20y_%7B5%7D%20%26%20y_%7B5%7D%20%5Cend%7Barray%7D%5Cright%5D "\left[\begin{array}{cccccccccc} y_{1} & y_{1} & y_{1} & y_{1} & y_{1} & y_{1} & y_{1} & y_{1} & y_{1} & y_{1}\\ y_{2} & y_{2} & y_{2} & y_{2} & y_{2} & y_{2} & y_{2} & y_{2} & y_{2} & y_{2}\\ y_{3} & y_{3} & y_{3} & y_{3} & y_{3} & y_{3} & y_{3} & y_{3} & y_{3} & y_{3}\\ y_{4} & y_{4} & y_{4} & y_{4} & y_{4} & y_{4} & y_{4} & y_{4} & y_{4} & y_{4}\\ y_{5} & y_{5} & y_{5} & y_{5} & y_{5} & y_{5} & y_{5} & y_{5} & y_{5} & y_{5} \end{array}\right]")

We use the Bayesian procedure to analyze these data. The model function
used previously can be used with a minor modification to do this.

``` r
Occ.model.dc = function(){
  # Likelihood 
  for(k in 1:ncl){
    for (i in 1:n){
      Y[i,k] ~ dbin(phi_occ, 1)
    }
  }
  # Prior
  phi_occ ~ dbeta(1, 1)
}
```

To match this change in the model, we need to turn the original data
into an array.

``` r
Y = array(Y, dim=c(n, 1))
Y = dcdim(Y)
```

When defining the data, we need to add another index `ncl` for the
cloned dimension. It gets multiplied by the number of clones.

``` r
dat = list(Y=Y, n=n, ncl=1)
# 2 clones of the Y array
dclone(Y, 2)
```

    ##       clone.1 clone.2
    ##  [1,]       1       1
    ##  [2,]       0       0
    ##  [3,]       0       0
    ##  [4,]       0       0
    ##  [5,]       1       1
    ##  [6,]       0       0
    ##  [7,]       1       1
    ##  [8,]       1       1
    ##  [9,]       0       0
    ## [10,]       0       0
    ## [11,]       0       0
    ## [12,]       0       0
    ## [13,]       0       0
    ## [14,]       0       0
    ## [15,]       0       0
    ## [16,]       1       1
    ## [17,]       0       0
    ## [18,]       1       1
    ## [19,]       1       1
    ## [20,]       0       0
    ## [21,]       1       1
    ## [22,]       1       1
    ## [23,]       0       0
    ## [24,]       0       0
    ## [25,]       0       0
    ## [26,]       0       0
    ## [27,]       0       0
    ## [28,]       0       0
    ## [29,]       0       0
    ## [30,]       1       1
    ## attr(,"n.clones")
    ## [1] 2
    ## attr(,"n.clones")attr(,"method")
    ## [1] "dim"
    ## attr(,"n.clones")attr(,"method")attr(,"drop")
    ## [1] TRUE

``` r
# 2 clones of the data list - this is not what we want
dclone(dat, 2)
```

    ## $Y
    ##       clone.1 clone.2
    ##  [1,]       1       1
    ##  [2,]       0       0
    ##  [3,]       0       0
    ##  [4,]       0       0
    ##  [5,]       1       1
    ##  [6,]       0       0
    ##  [7,]       1       1
    ##  [8,]       1       1
    ##  [9,]       0       0
    ## [10,]       0       0
    ## [11,]       0       0
    ## [12,]       0       0
    ## [13,]       0       0
    ## [14,]       0       0
    ## [15,]       0       0
    ## [16,]       1       1
    ## [17,]       0       0
    ## [18,]       1       1
    ## [19,]       1       1
    ## [20,]       0       0
    ## [21,]       1       1
    ## [22,]       1       1
    ## [23,]       0       0
    ## [24,]       0       0
    ## [25,]       0       0
    ## [26,]       0       0
    ## [27,]       0       0
    ## [28,]       0       0
    ## [29,]       0       0
    ## [30,]       1       1
    ## attr(,"n.clones")
    ## [1] 2
    ## attr(,"n.clones")attr(,"method")
    ## [1] "dim"
    ## attr(,"n.clones")attr(,"method")attr(,"drop")
    ## [1] TRUE
    ## 
    ## $n
    ## [1] 30 30
    ## attr(,"n.clones")
    ## [1] 2
    ## attr(,"n.clones")attr(,"method")
    ## [1] "rep"
    ## 
    ## $ncl
    ## [1] 1 1
    ## attr(,"n.clones")
    ## [1] 2
    ## attr(,"n.clones")attr(,"method")
    ## [1] "rep"

Notice that this changes `n` also. We do not want that, we want to keep
`n` unchanged.

``` r
dclone(dat, 2, unchanged="n", multiply="ncl")
```

    ## $Y
    ##       clone.1 clone.2
    ##  [1,]       1       1
    ##  [2,]       0       0
    ##  [3,]       0       0
    ##  [4,]       0       0
    ##  [5,]       1       1
    ##  [6,]       0       0
    ##  [7,]       1       1
    ##  [8,]       1       1
    ##  [9,]       0       0
    ## [10,]       0       0
    ## [11,]       0       0
    ## [12,]       0       0
    ## [13,]       0       0
    ## [14,]       0       0
    ## [15,]       0       0
    ## [16,]       1       1
    ## [17,]       0       0
    ## [18,]       1       1
    ## [19,]       1       1
    ## [20,]       0       0
    ## [21,]       1       1
    ## [22,]       1       1
    ## [23,]       0       0
    ## [24,]       0       0
    ## [25,]       0       0
    ## [26,]       0       0
    ## [27,]       0       0
    ## [28,]       0       0
    ## [29,]       0       0
    ## [30,]       1       1
    ## attr(,"n.clones")
    ## [1] 2
    ## attr(,"n.clones")attr(,"method")
    ## [1] "dim"
    ## attr(,"n.clones")attr(,"method")attr(,"drop")
    ## [1] TRUE
    ## 
    ## $n
    ## [1] 30
    ## 
    ## $ncl
    ## [1] 2
    ## attr(,"n.clones")
    ## [1] 2
    ## attr(,"n.clones")attr(,"method")
    ## [1] "multi"

The `dc.fit` function takes the familiar arguments to determine how to
clone the data list.

``` r
Occ.DC = dc.fit(data=dat, params="phi_occ", model=Occ.model.dc,
  n.clones=c(1, 2, 5), unchanged="n", multiply="ncl")
```

    ## 
    ## Fitting model with 1 clone 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 30
    ##    Unobserved stochastic nodes: 1
    ##    Total graph size: 34
    ## 
    ## Initializing model
    ## 
    ## 
    ## Fitting model with 2 clones 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 60
    ##    Unobserved stochastic nodes: 1
    ##    Total graph size: 64
    ## 
    ## Initializing model
    ## 
    ## 
    ## Fitting model with 5 clones 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 150
    ##    Unobserved stochastic nodes: 1
    ##    Total graph size: 154
    ## 
    ## Initializing model

``` r
summary(Occ.DC)
```

    ## 
    ## Iterations = 2001:7000
    ## Thinning interval = 1 
    ## Number of chains = 3 
    ## Sample size per chain = 5000 
    ## Number of clones = 5
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##           Mean      SD   DC SD  Naive SE Time-series SE R hat
    ## phi_occ 0.3361 0.03847 0.08601 0.0003141      0.0003953     1
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##   2.5%    25%    50%    75%  97.5% 
    ## 0.2632 0.3096 0.3354 0.3613 0.4149

``` r
plot(Occ.DC)
```

![](README_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

Notice the `Mean`, `DC SD`, and `R hat` columns in the summary. These
refer to the maximum likelihood estimate, the asymptotic standatd error
(SD of the posterior times square root of *K*), and the Gelman-Rubin
disgnostic:

``` r
coef(Occ.DC) # MLE
```

    ##   phi_occ 
    ## 0.3360527

``` r
dcsd(Occ.DC) # SE=SD*sqrt(K)
```

    ##    phi_occ 
    ## 0.08601376 
    ## attr(,"method")
    ##       Y       n     ncl 
    ##   "dim"      NA "multi"

``` r
gelman.diag(Occ.DC) # R hat
```

    ## Potential scale reduction factors:
    ## 
    ##         Point est. Upper C.I.
    ## phi_occ          1          1

### Summaries for the clones

Summaries of the posterior distributions for the different numbers of
clones are saved and we can print these out with the `dctable()`
command. We can visualize these with the `plot` function.

``` r
dctable(Occ.DC)
```

    ## $phi_occ
    ##   n.clones      mean         sd      2.5%       25%       50%       75%
    ## 1        1 0.3442208 0.08236735 0.1919951 0.2857756 0.3413784 0.3993429
    ## 2        2 0.3379790 0.05914627 0.2259625 0.2966980 0.3372281 0.3769647
    ## 3        5 0.3360527 0.03846652 0.2631959 0.3095706 0.3353931 0.3612651
    ##       97.5%    r.hat
    ## 1 0.5124188 1.000029
    ## 2 0.4575750 1.000337
    ## 3 0.4149269 1.000453
    ## 
    ## attr(,"class")
    ## [1] "dctable"

``` r
dctable(Occ.DC) |> plot()
```

![](README_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

Some data cloning related diagnostics are printed with the `dcdiag()`
function. We will discuss these statistics in detail. The most important
thing to ckeck is that the solid line follows the scattered line for
`lambda.max`, i.e. decreases with the number of clones.

``` r
dcdiag(Occ.DC)
```

    ##   n.clones  lambda.max    ms.error    r.squared    r.hat
    ## 1        1 0.006784381 0.005060674 0.0016489249 1.000029
    ## 2        2 0.003498281 0.001121359 0.0005475793 1.000337
    ## 3        5 0.001479673 0.001580203 0.0007886910 1.000453

``` r
dcdiag(Occ.DC) |> plot()
```

![](README_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

## Regression models

Now we will generalize these models to account for covariates. We will
consider Logistic regression but also comment on how to change it to
Probit regression easily. Similarly we show how this basic prototype can
be modified to do linear and non-linear regression, Poisson regression
etc.

``` r
n = 30 # sample size
X1 = rnorm(n) # a covariate
X = model.matrix(~X1)
beta.true = c(0.5, 1)
link_mu = X %*% beta.true # logit scale
```

### Logistic regression model

``` r
phi_occ = plogis(link_mu) # prob scale
Y = rbinom(n, 1, phi_occ)
```

Maximum likelihood estimate using `glm()`:

``` r
MLE.est = glm(Y ~ X1, family="binomial")
```

Bayesian analysis

``` r
Occ.model = function(){
  # Likelihood 
  for (i in 1:n){
    phi_occ[i] <- ilogit(X[i,] %*% beta)
    Y[i] ~ dbin(phi_occ[i], 1)
  }
  # Prior
  beta[1] ~ dnorm(0, 1)
  beta[2] ~ dnorm(0, 1)
}
```

Now we need to provide the data to the model and generate random numbers
from the posterior. We will discuss different options later.

``` r
dat = list(Y=Y, X=X, n=n)
Occ.Bayes = jags.fit(data=dat, params="beta", model=Occ.model)
```

    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 30
    ##    Unobserved stochastic nodes: 2
    ##    Total graph size: 186
    ## 
    ## Initializing model

``` r
summary(Occ.Bayes)
```

    ## 
    ## Iterations = 2001:7000
    ## Thinning interval = 1 
    ## Number of chains = 3 
    ## Sample size per chain = 5000 
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##           Mean     SD Naive SE Time-series SE
    ## beta[1] -0.132 0.3795 0.003098       0.003949
    ## beta[2]  1.133 0.4942 0.004035       0.005254
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##            2.5%     25%     50%    75%  97.5%
    ## beta[1] -0.8772 -0.3898 -0.1317 0.1261 0.6035
    ## beta[2]  0.2111  0.7937  1.1200 1.4529 2.1602

``` r
plot(Occ.Bayes)
```

![](README_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

``` r
pairs(Occ.Bayes)
```

![](README_files/figure-gfm/unnamed-chunk-18-2.png)<!-- -->

Now we modify this to get the MLE using data cloning.

``` r
Occ.model_dc = function(){
  # Likelihood 
  for (k in 1:ncl){
    for (i in 1:n){
      phi_occ[i,k] <- ilogit(X[i,,k] %*% beta)
      Y[i,k] ~ dbin(phi_occ[i,k],1)
    }
  }
  # Prior
  beta[1] ~ dnorm(0, 1)
  beta[2] ~ dnorm(0, 1)
}
```

Now we need to provide the data to the model and generate random numbers
from the posterior. We will discuss different options later.

``` r
Y = array(Y, dim=c(n, 1))
X = array(X, dim=c(dim(X), 1))
# clone the objects
Y = dcdim(Y)
X = dcdim(X)
```

Data cloning with `dc.fit()`:

``` r
dat = list(Y=Y, X=X, n=n, ncl=1)
Occ.DC = dc.fit(data=dat, params="beta", model=Occ.model_dc,
  n.clones=c(1, 2, 5), unchanged="n", multiply="ncl")
```

    ## 
    ## Fitting model with 1 clone 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 30
    ##    Unobserved stochastic nodes: 2
    ##    Total graph size: 187
    ## 
    ## Initializing model
    ## 
    ## 
    ## Fitting model with 2 clones 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 60
    ##    Unobserved stochastic nodes: 2
    ##    Total graph size: 307
    ## 
    ## Initializing model
    ## 
    ## 
    ## Fitting model with 5 clones 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 150
    ##    Unobserved stochastic nodes: 2
    ##    Total graph size: 667
    ## 
    ## Initializing model

``` r
summary(Occ.DC)
```

    ## 
    ## Iterations = 2001:7000
    ## Thinning interval = 1 
    ## Number of chains = 3 
    ## Sample size per chain = 5000 
    ## Number of clones = 5
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##            Mean     SD  DC SD Naive SE Time-series SE  R hat
    ## beta[1] -0.1537 0.1824 0.4079 0.001489       0.001871 0.9999
    ## beta[2]  1.3555 0.2560 0.5724 0.002090       0.002633 1.0003
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##            2.5%     25%     50%     75%  97.5%
    ## beta[1] -0.5140 -0.2755 -0.1525 -0.0289 0.1962
    ## beta[2]  0.8778  1.1802  1.3473  1.5218 1.8813

``` r
plot(Occ.DC)
```

![](README_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->

``` r
pairs(Occ.DC)
```

![](README_files/figure-gfm/unnamed-chunk-21-2.png)<!-- -->

These are the familiar functions for the asymptotic ML inference
(already accounted for the number of clones):

``` r
coef(Occ.DC) # MLE
```

    ##   beta[1]   beta[2] 
    ## -0.153658  1.355483

``` r
dcsd(Occ.DC) # SE
```

    ##   beta[1]   beta[2] 
    ## 0.4078653 0.5723581

``` r
vcov(Occ.DC) # asymptotic VCV
```

    ##              beta[1]      beta[2]
    ## beta[1]  0.166354106 -0.003761021
    ## beta[2] -0.003761021  0.327593796

``` r
confint(Occ.DC, level=0.95) # 95% CI
```

    ##              2.5 %    97.5 %
    ## beta[1] -0.9530593 0.6457433
    ## beta[2]  0.2336817 2.4772842

Let’s check the posterior summaries and the DC diagnostics:

``` r
dctable(Occ.DC)
```

    ## $`beta[1]`
    ##   n.clones       mean        sd       2.5%        25%        50%        75%
    ## 1        1 -0.1363533 0.3766352 -0.8782684 -0.3871472 -0.1346313  0.1167166
    ## 2        2 -0.1440196 0.2733913 -0.6833644 -0.3234490 -0.1429613  0.0399442
    ## 3        5 -0.1536580 0.1824029 -0.5139933 -0.2754806 -0.1524508 -0.0289017
    ##       97.5%     r.hat
    ## 1 0.6044465 1.0017379
    ## 2 0.3828652 1.0002462
    ## 3 0.1962173 0.9999155
    ## 
    ## $`beta[2]`
    ##   n.clones     mean        sd      2.5%       25%      50%      75%    97.5%
    ## 1        1 1.138760 0.4991882 0.1910324 0.7967937 1.123490 1.470346 2.148675
    ## 2        2 1.255492 0.3800814 0.5468632 0.9971681 1.242364 1.497280 2.044164
    ## 3        5 1.355483 0.2559663 0.8778271 1.1801514 1.347302 1.521779 1.881280
    ##       r.hat
    ## 1 0.9999858
    ## 2 0.9999722
    ## 3 1.0003475
    ## 
    ## attr(,"class")
    ## [1] "dctable"

``` r
dctable(Occ.DC) |> plot()
```

![](README_files/figure-gfm/unnamed-chunk-23-1.png)<!-- -->

``` r
dcdiag(Occ.DC)
```

    ##   n.clones lambda.max    ms.error    r.squared    r.hat
    ## 1        1  0.2491952 0.023644992 0.0045901096 1.000227
    ## 2        2  0.1445526 0.024192668 0.0042162055 1.000163
    ## 3        5  0.0655363 0.001286623 0.0002434335 1.000035

``` r
dcdiag(Occ.DC) |> plot()
```

![](README_files/figure-gfm/unnamed-chunk-23-2.png)<!-- -->

We hope you can see the pattern in how we are changing the prototype
model function and the data function. If we want to do a Normal linear
regression and Poisson regression we can modify the regression program
above easily.

### Linear regression

The following section issustrates Gaussian linear regression.

``` r
n = 30
X1 = rnorm(n)
X = model.matrix(~X1)
beta.true = c(0.5, 1)
link_mu = X %*% beta.true

# Linear regression model
mu = link_mu
sigma.e = 1
Y = rnorm(n,mean=mu,sd=sigma.e)

# MLE
MLE.est = glm(Y ~ X1, family="gaussian")

# Bayesian analysis
Normal.model = function(){
  # Likelihood 
  for (i in 1:n){
    mu[i] <- X[i,] %*% beta
    Y[i] ~ dnorm(mu[i],prec.e)
  }
  # Prior
  beta[1] ~ dnorm(0, 1)
  beta[2] ~ dnorm(0, 1)
  prec.e ~ dlnorm(0, 1)
}
dat = list(Y=Y, X=X, n=n)
Normal.Bayes = jags.fit(data=dat, params=c("beta","prec.e"), model=Normal.model)
```

    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 30
    ##    Unobserved stochastic nodes: 3
    ##    Total graph size: 157
    ## 
    ## Initializing model

``` r
summary(Normal.Bayes)
```

    ## 
    ## Iterations = 2001:7000
    ## Thinning interval = 1 
    ## Number of chains = 3 
    ## Sample size per chain = 5000 
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##           Mean     SD Naive SE Time-series SE
    ## beta[1] 0.5150 0.1730 0.001412       0.001400
    ## beta[2] 0.8126 0.2388 0.001949       0.001949
    ## prec.e  1.1733 0.3022 0.002468       0.003445
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%    25%    50%    75%  97.5%
    ## beta[1] 0.1745 0.3998 0.5156 0.6301 0.8516
    ## beta[2] 0.3363 0.6571 0.8115 0.9726 1.2762
    ## prec.e  0.6625 0.9533 1.1480 1.3611 1.8428

``` r
plot(Normal.Bayes)
```

![](README_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->

``` r
pairs(Normal.Bayes)
```

![](README_files/figure-gfm/unnamed-chunk-24-2.png)<!-- -->

``` r
# MLE using data cloning.
Normal.model_dc = function(){
  # Likelihood 
  for (k in 1:ncl){
    for (i in 1:n){
      mu[i,k] <- X[i,,k] %*% beta
      Y[i,k] ~ dnorm(mu[i,k],prec.e)
    }
  }
  # Prior
  beta[1] ~ dnorm(0, 1)
  beta[2] ~ dnorm(0, 1)
  prec.e ~ dlnorm(0, 1)
}
Y = array(Y, dim=c(n, 1))
X = array(X, dim=c(dim(X), 1))
Y = dcdim(Y)
X = dcdim(X)
dat = list(Y=Y, X=X, n=n, ncl=1)
Normal.DC = dc.fit(data=dat, params=c("beta","prec.e"), model=Normal.model_dc,
  n.clones=c(1, 2, 5), unchanged="n", multiply="ncl")
```

    ## 
    ## Fitting model with 1 clone 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 30
    ##    Unobserved stochastic nodes: 3
    ##    Total graph size: 158
    ## 
    ## Initializing model
    ## 
    ## 
    ## Fitting model with 2 clones 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 60
    ##    Unobserved stochastic nodes: 3
    ##    Total graph size: 278
    ## 
    ## Initializing model
    ## 
    ## 
    ## Fitting model with 5 clones 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 150
    ##    Unobserved stochastic nodes: 3
    ##    Total graph size: 638
    ## 
    ## Initializing model

``` r
summary(Normal.DC)
```

    ## 
    ## Iterations = 2001:7000
    ## Thinning interval = 1 
    ## Number of chains = 3 
    ## Sample size per chain = 5000 
    ## Number of clones = 5
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##           Mean      SD  DC SD  Naive SE Time-series SE  R hat
    ## beta[1] 0.5282 0.07352 0.1644 0.0006003      0.0006105 1.0001
    ## beta[2] 0.8481 0.10339 0.2312 0.0008442      0.0008305 1.0000
    ## prec.e  1.2465 0.14478 0.3237 0.0011822      0.0015873 0.9999
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%    25%    50%    75%  97.5%
    ## beta[1] 0.3847 0.4783 0.5281 0.5772 0.6707
    ## beta[2] 0.6473 0.7784 0.8482 0.9180 1.0528
    ## prec.e  0.9826 1.1458 1.2407 1.3418 1.5504

``` r
plot(Normal.DC)
```

![](README_files/figure-gfm/unnamed-chunk-24-3.png)<!-- -->

``` r
pairs(Normal.DC)
```

![](README_files/figure-gfm/unnamed-chunk-24-4.png)<!-- -->

### Poisson log-link regression

We will now modify the code to conduct count data regression using the
Poisson distribution and log-link.

``` r
n = 30
X1 = rnorm(n)
X = model.matrix(~X1)
beta.true = c(0.5, 1)
link_mu = X %*% beta.true

# Log-linear regression model
mu = exp(link_mu)
Y = rpois(n, mu)

# MLE
MLE.est = glm(Y ~ X1, family="poisson")

# Bayesian analysis
Poisson.model = function(){
  # Likelihood 
  for (i in 1:n){
    mu[i] <- exp(X[i,] %*% beta)
    Y[i] ~ dpois(mu[i])
  }
  # Prior
  beta[1] ~ dnorm(0, 1)
  beta[2] ~ dnorm(0, 1)
}
dat = list(Y=Y, X=X, n=n)
Poisson.Bayes = jags.fit(data=dat, params="beta", model=Poisson.model)
```

    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 30
    ##    Unobserved stochastic nodes: 2
    ##    Total graph size: 186
    ## 
    ## Initializing model

``` r
summary(Poisson.Bayes)
```

    ## 
    ## Iterations = 2001:7000
    ## Thinning interval = 1 
    ## Number of chains = 3 
    ## Sample size per chain = 5000 
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##           Mean     SD Naive SE Time-series SE
    ## beta[1] 0.6142 0.1465 0.001196       0.002057
    ## beta[2] 1.0089 0.1322 0.001079       0.001835
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%    25%   50%    75%  97.5%
    ## beta[1] 0.3174 0.5161 0.619 0.7162 0.8852
    ## beta[2] 0.7521 0.9193 1.008 1.0962 1.2719

``` r
plot(Poisson.Bayes)
```

![](README_files/figure-gfm/unnamed-chunk-25-1.png)<!-- -->

``` r
pairs(Poisson.Bayes)
```

![](README_files/figure-gfm/unnamed-chunk-25-2.png)<!-- -->

``` r
# MLE using data cloning
Poisson.model_dc = function(){
  # Likelihood 
  for (k in 1:ncl){
    for (i in 1:n){
      mu[i,k] <- exp(X[i,,k] %*% beta)
      Y[i,k] ~ dpois(mu[i,k])
    }
  }
  # Prior
  beta[1] ~ dnorm(0, 1)
  beta[2] ~ dnorm(0, 1)
}
Y = array(Y, dim=c(n, 1))
X = array(X, dim=c(dim(X), 1))
Y = dcdim(Y)
X = dcdim(X)
dat = list(Y=Y, X=X, n=n, ncl=1)
Poisson.DC = dc.fit(data=dat, params="beta", model=Poisson.model_dc,
  n.clones=c(1, 2, 5), unchanged="n", multiply="ncl")
```

    ## 
    ## Fitting model with 1 clone 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 30
    ##    Unobserved stochastic nodes: 2
    ##    Total graph size: 187
    ## 
    ## Initializing model
    ## 
    ## 
    ## Fitting model with 2 clones 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 60
    ##    Unobserved stochastic nodes: 2
    ##    Total graph size: 307
    ## 
    ## Initializing model
    ## 
    ## 
    ## Fitting model with 5 clones 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 150
    ##    Unobserved stochastic nodes: 2
    ##    Total graph size: 667
    ## 
    ## Initializing model

``` r
summary(Poisson.DC)
```

    ## 
    ## Iterations = 2001:7000
    ## Thinning interval = 1 
    ## Number of chains = 3 
    ## Sample size per chain = 5000 
    ## Number of clones = 5
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##          Mean      SD  DC SD  Naive SE Time-series SE R hat
    ## beta[1] 0.627 0.06671 0.1492 0.0005447      0.0009843 1.001
    ## beta[2] 1.018 0.06014 0.1345 0.0004910      0.0008706 1.001
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%    25%    50%   75%  97.5%
    ## beta[1] 0.4929 0.5831 0.6289 0.672 0.7537
    ## beta[2] 0.9021 0.9768 1.0172 1.057 1.1368

``` r
plot(Poisson.DC)
```

![](README_files/figure-gfm/unnamed-chunk-25-3.png)<!-- -->

``` r
pairs(Poisson.DC)
```

![](README_files/figure-gfm/unnamed-chunk-25-4.png)<!-- -->

## Why use MCMC based Bayesian and data cloning?

1.  Writing the model function is much more intuitive than writing the
    likelihood function, prior, etc.
2.  Do not need to do numerical integration or numerical optimization.
3.  Data cloning overcomes multimodality of the likelihood function.
    Entire prior distribution essentially works as a set of starting
    values. In the usual optimization, starting values can be quite
    important when the function is not well behaved. By using data
    cloning, except for the global maximum, all the local maxima tend to
    0.
4.  Asymptotic variance of the MLE is simple to obtain. It is also more
    stable than computing the inverse of the second derivative of the
    log-likelihood function numerically.
