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

    ## dclone 2.3-1      2022-07-11

``` r
phi.true = 0.3
n = 30
Y = rbinom(n,1,phi.true)

table(Y)
```

    ## Y
    ##  0  1 
    ## 14 16

Analytical MLE:

``` r
MLE.est = sum(Y)/n
```

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
    ##      0.5314316      0.0864074      0.0007055      0.0008578 
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##   2.5%    25%    50%    75%  97.5% 
    ## 0.3633 0.4711 0.5317 0.5933 0.6954

``` r
plot(Occ.Bayes)
```

![](README_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

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
    ##  [1,]       0       0
    ##  [2,]       1       1
    ##  [3,]       1       1
    ##  [4,]       1       1
    ##  [5,]       1       1
    ##  [6,]       0       0
    ##  [7,]       1       1
    ##  [8,]       1       1
    ##  [9,]       0       0
    ## [10,]       0       0
    ## [11,]       1       1
    ## [12,]       0       0
    ## [13,]       0       0
    ## [14,]       0       0
    ## [15,]       1       1
    ## [16,]       0       0
    ## [17,]       1       1
    ## [18,]       1       1
    ## [19,]       1       1
    ## [20,]       0       0
    ## [21,]       1       1
    ## [22,]       0       0
    ## [23,]       1       1
    ## [24,]       1       1
    ## [25,]       0       0
    ## [26,]       0       0
    ## [27,]       0       0
    ## [28,]       0       0
    ## [29,]       1       1
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
    ##  [1,]       0       0
    ##  [2,]       1       1
    ##  [3,]       1       1
    ##  [4,]       1       1
    ##  [5,]       1       1
    ##  [6,]       0       0
    ##  [7,]       1       1
    ##  [8,]       1       1
    ##  [9,]       0       0
    ## [10,]       0       0
    ## [11,]       1       1
    ## [12,]       0       0
    ## [13,]       0       0
    ## [14,]       0       0
    ## [15,]       1       1
    ## [16,]       0       0
    ## [17,]       1       1
    ## [18,]       1       1
    ## [19,]       1       1
    ## [20,]       0       0
    ## [21,]       1       1
    ## [22,]       0       0
    ## [23,]       1       1
    ## [24,]       1       1
    ## [25,]       0       0
    ## [26,]       0       0
    ## [27,]       0       0
    ## [28,]       0       0
    ## [29,]       1       1
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
    ##  [1,]       0       0
    ##  [2,]       1       1
    ##  [3,]       1       1
    ##  [4,]       1       1
    ##  [5,]       1       1
    ##  [6,]       0       0
    ##  [7,]       1       1
    ##  [8,]       1       1
    ##  [9,]       0       0
    ## [10,]       0       0
    ## [11,]       1       1
    ## [12,]       0       0
    ## [13,]       0       0
    ## [14,]       0       0
    ## [15,]       1       1
    ## [16,]       0       0
    ## [17,]       1       1
    ## [18,]       1       1
    ## [19,]       1       1
    ## [20,]       0       0
    ## [21,]       1       1
    ## [22,]       0       0
    ## [23,]       1       1
    ## [24,]       1       1
    ## [25,]       0       0
    ## [26,]       0       0
    ## [27,]       0       0
    ## [28,]       0       0
    ## [29,]       1       1
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
    ##           Mean      SD DC SD.phi_occ  Naive SE Time-series SE R hat
    ## phi_occ 0.5322 0.03993       0.08929 0.0003261      0.0004144     1
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##   2.5%    25%    50%    75%  97.5% 
    ## 0.4534 0.5053 0.5323 0.5595 0.6101

``` r
plot(Occ.DC)
```

![](README_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

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
    ## beta[1] 0.4221 0.3659 0.002988       0.003936
    ## beta[2] 0.2809 0.3507 0.002864       0.003712
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##            2.5%     25%    50%    75%  97.5%
    ## beta[1] -0.2853 0.17484 0.4205 0.6647 1.1433
    ## beta[2] -0.4070 0.04671 0.2749 0.5116 0.9752

``` r
plot(Occ.Bayes)
```

![](README_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

``` r
pairs(Occ.Bayes)
```

    ## Warning in par(usr): argument 1 does not name a graphical parameter

    ## Warning in par(usr): argument 1 does not name a graphical parameter

![](README_files/figure-gfm/unnamed-chunk-14-2.png)<!-- -->

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
    ##           Mean     SD  DC SD Naive SE Time-series SE R hat
    ## beta[1] 0.4572 0.1697 0.3794 0.001385       0.001820 1.001
    ## beta[2] 0.3011 0.1609 0.3599 0.001314       0.001778 1.001
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##             2.5%    25%    50%    75%  97.5%
    ## beta[1]  0.12756 0.3413 0.4570 0.5715 0.7896
    ## beta[2] -0.01494 0.1931 0.3009 0.4089 0.6178

``` r
plot(Occ.DC)
```

![](README_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

``` r
pairs(Occ.DC)
```

    ## Warning in par(usr): argument 1 does not name a graphical parameter

    ## Warning in par(usr): argument 1 does not name a graphical parameter

![](README_files/figure-gfm/unnamed-chunk-17-2.png)<!-- -->

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
    ## beta[1] 0.4639 0.1667 0.001361       0.001423
    ## beta[2] 0.9969 0.1329 0.001085       0.001159
    ## prec.e  1.3256 0.3368 0.002750       0.003718
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%    25%    50%    75%  97.5%
    ## beta[1] 0.1319 0.3546 0.4659 0.5725 0.7953
    ## beta[2] 0.7307 0.9097 0.9985 1.0844 1.2609
    ## prec.e  0.7546 1.0838 1.2987 1.5348 2.0707

``` r
plot(Normal.Bayes)
```

![](README_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

``` r
pairs(Normal.Bayes)
```

    ## Warning in par(usr): argument 1 does not name a graphical parameter

    ## Warning in par(usr): argument 1 does not name a graphical parameter

    ## Warning in par(usr): argument 1 does not name a graphical parameter

![](README_files/figure-gfm/unnamed-chunk-18-2.png)<!-- -->

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
    ##           Mean      SD  DC SD  Naive SE Time-series SE R hat
    ## beta[1] 0.4717 0.07067 0.1580 0.0005770      0.0006027 1.000
    ## beta[2] 1.0083 0.05618 0.1256 0.0004587      0.0004850 1.000
    ## prec.e  1.4205 0.16529 0.3696 0.0013496      0.0017728 1.001
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%    25%    50%    75%  97.5%
    ## beta[1] 0.3334 0.4242 0.4713 0.5191 0.6106
    ## beta[2] 0.8983 0.9706 1.0087 1.0462 1.1182
    ## prec.e  1.1137 1.3040 1.4145 1.5292 1.7607

``` r
plot(Normal.DC)
```

![](README_files/figure-gfm/unnamed-chunk-18-3.png)<!-- -->

``` r
pairs(Normal.DC)
```

    ## Warning in par(usr): argument 1 does not name a graphical parameter

    ## Warning in par(usr): argument 1 does not name a graphical parameter

    ## Warning in par(usr): argument 1 does not name a graphical parameter

![](README_files/figure-gfm/unnamed-chunk-18-4.png)<!-- -->

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
    ##           Mean     SD  Naive SE Time-series SE
    ## beta[1] 0.6994 0.1411 0.0011518       0.002495
    ## beta[2] 1.0276 0.1085 0.0008861       0.001880
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%    25%    50%    75%  97.5%
    ## beta[1] 0.4126 0.6047 0.7026 0.7986 0.9634
    ## beta[2] 0.8190 0.9528 1.0267 1.1013 1.2421

``` r
plot(Poisson.Bayes)
```

![](README_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

``` r
pairs(Poisson.Bayes)
```

    ## Warning in par(usr): argument 1 does not name a graphical parameter

    ## Warning in par(usr): argument 1 does not name a graphical parameter

![](README_files/figure-gfm/unnamed-chunk-19-2.png)<!-- -->

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
    ##           Mean      SD  DC SD  Naive SE Time-series SE R hat
    ## beta[1] 0.7106 0.06465 0.1446 0.0005279      0.0012126 1.001
    ## beta[2] 1.0300 0.04967 0.1111 0.0004055      0.0009202 1.001
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%    25%   50%    75%  97.5%
    ## beta[1] 0.5804 0.6677 0.711 0.7543 0.8352
    ## beta[2] 0.9325 0.9966 1.030 1.0634 1.1268

``` r
plot(Poisson.DC)
```

![](README_files/figure-gfm/unnamed-chunk-19-3.png)<!-- -->

``` r
pairs(Poisson.DC)
```

    ## Warning in par(usr): argument 1 does not name a graphical parameter

    ## Warning in par(usr): argument 1 does not name a graphical parameter

![](README_files/figure-gfm/unnamed-chunk-19-4.png)<!-- -->

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
