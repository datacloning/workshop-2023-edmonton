Random effects models and data cloning
================

## Introduction

In the previous part, we reviewed the basic statistical concepts behind
the likelihood inference and the Bayesian inference. We looked at how to
write a JAGS model function for some linear and generalized linear
regression models and use it in the package ‘dclone’ to get the Bayesian
credible intervals as well as the frequentist confidence intervals based
on the asymptotic normal distribution. We also discussed some of the
reasons to use the MCMC approach to conducting statistical inference,
either Bayesian or frequentist.

We will now generalize the models to make them relevant to some complex
practical situations. These are some of the situations where the
analytical approaches to Bayesian and likelihood inference are difficult
to impossible to implement. The question of estimability of the
parameters becomes much more relevant but difficult to diagnose. The
method of data cloning is particularly useful for diagnosing
estimability of the parameters. You will notice that, although the
models are much more complex, the coding component does not increase in
complexity. We will also discuss prediction of missing data.

## Detection error in occupancy studies (latent variables)

Let us revisit the occupancy model again. In practice, the assumption
that you observe the occupancy status correctly is somewhat suspect. For
example, if we are looking for a bird species, if the bird never sings
or gives some sort of a cue, it is extremely difficult to know if they
are there. Hence, even if the species is present, we may make an error
and note that it is not present. This is called ‘detection error’. How
can we model this?

Let ![W_i](https://latex.codecogs.com/svg.image?W_i "W_i") denote the
true status of the *i*-th cell. So in our previous notation, now
![P(W\_{i}=1)=\phi](https://latex.codecogs.com/svg.image?P%28W_%7Bi%7D%3D1%29%3D%5Cphi "P(W_{i}=1)=\phi").
The observed value, generally denoted by
![Y_i](https://latex.codecogs.com/svg.image?Y_i "Y_i") could be 1 or 0
depending on the true status. If we assume that the species are never
misidentified, then we can write
![P(Y_i=1\|W\_{i}=1)=p](https://latex.codecogs.com/svg.image?P%28Y_i%3D1%7CW_%7Bi%7D%3D1%29%3Dp "P(Y_i=1|W_{i}=1)=p")
and
![P(Y_i=0\|W\_{i}=1)=1-p](https://latex.codecogs.com/svg.image?P%28Y_i%3D0%7CW_%7Bi%7D%3D1%29%3D1-p "P(Y_i=0|W_{i}=1)=1-p").
Moreover,
![P(Y_i=0\|W\_{i}=0)=1](https://latex.codecogs.com/svg.image?P%28Y_i%3D0%7CW_%7Bi%7D%3D0%29%3D1 "P(Y_i=0|W_{i}=0)=1").
In principle, we can also have misidentification. For example, a coyote
could be mistaken for a wolf. But we will not discuss it here. It is a
fairly simple extension of this model. Probability of detection is
![p](https://latex.codecogs.com/svg.image?p "p") and probability of
occupancy is
![\phi](https://latex.codecogs.com/svg.image?%5Cphi "\phi"). How can we
infer about these given the data?

Notice that we only observe
![Y_i](https://latex.codecogs.com/svg.image?Y_i "Y_i")’s and not the
![W_i](https://latex.codecogs.com/svg.image?W_i "W_i")’s. *The
unobserved variable
![W_i](https://latex.codecogs.com/svg.image?W_i "W_i") is called a
latent variable.*

To write down the likelihood function, we need to compute the
distribution of the observed data
![Y_i](https://latex.codecogs.com/svg.image?Y_i "Y_i"). We can see that
![P(Y\_{i}=1)=p \phi](https://latex.codecogs.com/svg.image?P%28Y_%7Bi%7D%3D1%29%3Dp%20%5Cphi "P(Y_{i}=1)=p \phi")
and
![P(Y\_{i}=0)=(1-p) \phi](https://latex.codecogs.com/svg.image?P%28Y_%7Bi%7D%3D0%29%3D%281-p%29%20%5Cphi "P(Y_{i}=0)=(1-p) \phi").
We can write down the likelihood based on this. However, we are going to
write this as a hierarchical model.

Let us modify the previous code to see how this inference proceeds.

``` r
library(dclone)
```

    ## Loading required package: coda

    ## Loading required package: parallel

    ## Loading required package: Matrix

    ## dclone 2.3-1      2023-04-07

``` r
phi.true = 0.3 # occupancy
p.true = 0.7   # detectability
n = 30         # sample size
W = rbinom(n, 1 ,phi.true)   # true status
Y = rbinom(n, 1, W * p.true) # detections
```

### Bayesian inference using JAGS and dclone

Step 1: WE need to define the model function. This is the critical
component.

``` r
Occ.model = function(){
  # Likelihood: Latent variables are random variables.
  for (i in 1:n){
    W[i] ~ dbin(phi_occ, 1)
    Y[i] ~ dbin(W[i] * p_det, 1)
  }
  # Priors
  phi_occ ~ dbeta(1, 1)
  p_det ~ dbeta(1, 1)
}
```

Now we need to provide the data to the model and generate random numbers
from the posterior. We will discuss different options later.

``` r
dat = list(Y=Y, n=n)
```

Following command will not work. But try it by removing the comments
hash to see what happens.

``` r
Occ.Bayes = jags.fit(data=dat, params=c("phi_occ","p_det"), model=Occ.model)
```

    ## Registered S3 method overwritten by 'R2WinBUGS':
    ##   method            from  
    ##   as.mcmc.list.bugs dclone

    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 30
    ##    Unobserved stochastic nodes: 32
    ##    Total graph size: 94
    ## 
    ## Initializing model
    ## Deleting model

    ## Error in rjags::jags.model(model, data, n.chains = n.chains, n.adapt = n.adapt, : Error in node Y[1]
    ## Node inconsistent with parents

This did not quite work. When there are latent variables, many times, we
have to start the process at the appropriate initial values.

``` r
ini = list(W=rep(1, n)) 
Occ.Bayes = jags.fit(data=dat, params=c("phi_occ","p_det"), model = Occ.model,
    inits=ini)
```

    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 30
    ##    Unobserved stochastic nodes: 32
    ##    Total graph size: 94
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
    ## p_det   0.5354 0.2260 0.001845       0.008623
    ## phi_occ 0.5075 0.2217 0.001810       0.010029
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%    25%    50%    75%  97.5%
    ## p_det   0.1816 0.3479 0.5057 0.7151 0.9642
    ## phi_occ 0.1723 0.3285 0.4689 0.6715 0.9579

``` r
plot(Occ.Bayes)
```

![](README_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

This seems to work quite well! But our answers are quite weird (We know
the truth!). Let us plot the two dimensional (joint) distribution of the
parameters.

``` r
pairs(Occ.Bayes)
```

![](README_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

This suggests that the posterior distribution is banana shaped. But just
looking at these plots, we cannot say for sure if our answers are
correct or not.

Should we, then, accept the answers? Not so fast. Let us look at the
model again. It is clear that we can estimate the product
![p \phi](https://latex.codecogs.com/svg.image?p%20%5Cphi "p \phi")
given the data. But decomposing this product in
![p](https://latex.codecogs.com/svg.image?p "p") and
![\phi](https://latex.codecogs.com/svg.image?%5Cphi "\phi") is
impossible. This is called ‘non-estimability’.

In this case, this also is non-identifiability. There are several
combinations of ![p](https://latex.codecogs.com/svg.image?p "p") and
![\phi](https://latex.codecogs.com/svg.image?%5Cphi "\phi") that lead to
the same
![p \phi](https://latex.codecogs.com/svg.image?p%20%5Cphi "p \phi") and
hence the same distribution of the observed data. Such situations are
not uncommon when dealing with the hiearchical models in general, and
measurement error models in particular.

We should not make any inferences about the probability of occupancy
based on these data. You can change the priors and see what happens to
the posteriors. You might find it interesting and educational.

**Non-estimability: If there are two or more values in the parameter
space that lead to identical likelihood value, such values are called
‘non-estimable’.**

Note: You may recall from the linear regression that if the covariates
are perfectly correlated with each other, the regression coefficients
are non-estimable. If covariate
![X_1](https://latex.codecogs.com/svg.image?X_1 "X_1") is perfectly
correlated with ![X_2](https://latex.codecogs.com/svg.image?X_2 "X_2"),
these covariates separately give no additional information.

### Bayesian result and its data cloning version

If the posterior distribution converges to a non-degenerate distribution
as the sample size increases, it implies that set of parameters is
non-estimable.

If the posterior distribution converges to a non-degenerate distribution
as the number of clones increases, it implies that set of parameters is
non-estimable.

An immediate consequence of this result is that the variance of the
posterior distribution does not converge to 0 (instead it converges to
some positive number).

Let us modify our data cloning code to see what happens.

``` r
Occ.model.dc = function(){
  # Likelihood 
  for(k in 1:ncl){
    for (i in 1:n){
        W[i,k] ~ dbin(p_det, 1)
        Y[i,k] ~ dbin(phi_occ, 1)
    }
  }
  # Prior
  phi_occ ~ dbeta(1, 1)
  p_det ~ dbeta(1, 1)
}
```

We need to turn the original data into an array. And we need to add
another index `ncl` for the cloned dimension. It gets multiplied by the
number of clones.

``` r
Y = array(Y, dim=c(n, 1))
Y = dcdim(Y)
dat = list(Y=Y, n=n, ncl=1)
```

As previously, we need to initiate the `W`’s.

``` r
ini = list(W=array(rep(1, n), dim=c(n, 1)))
```

We need to clone these initial values as well. You should always check
if this is doing the right job.

``` r
initfn = function(model, n.clones){
  W=array(rep(1, n), dim=c(n, 1))
  list(W=dclone(dcdim(W), n.clones))
}
initfn(n.clones=2)
```

    ## $W
    ##       clone.1 clone.2
    ##  [1,]       1       1
    ##  [2,]       1       1
    ##  [3,]       1       1
    ##  [4,]       1       1
    ##  [5,]       1       1
    ##  [6,]       1       1
    ##  [7,]       1       1
    ##  [8,]       1       1
    ##  [9,]       1       1
    ## [10,]       1       1
    ## [11,]       1       1
    ## [12,]       1       1
    ## [13,]       1       1
    ## [14,]       1       1
    ## [15,]       1       1
    ## [16,]       1       1
    ## [17,]       1       1
    ## [18,]       1       1
    ## [19,]       1       1
    ## [20,]       1       1
    ## [21,]       1       1
    ## [22,]       1       1
    ## [23,]       1       1
    ## [24,]       1       1
    ## [25,]       1       1
    ## [26,]       1       1
    ## [27,]       1       1
    ## [28,]       1       1
    ## [29,]       1       1
    ## [30,]       1       1
    ## attr(,"n.clones")
    ## [1] 2
    ## attr(,"n.clones")attr(,"method")
    ## [1] "dim"
    ## attr(,"n.clones")attr(,"method")attr(,"drop")
    ## [1] TRUE

Let’s run data cloning.

``` r
Occ.DC = dc.fit(data=dat, params=c("phi_occ","p_det"), model=Occ.model.dc,
    n.clones=c(1, 2, 5), unchanged="n", multiply="ncl",
    inits=ini, initsfun=initfn)
```

    ## 
    ## Fitting model with 1 clone 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 30
    ##    Unobserved stochastic nodes: 32
    ##    Total graph size: 65
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
    ##    Unobserved stochastic nodes: 62
    ##    Total graph size: 125
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
    ##    Unobserved stochastic nodes: 152
    ##    Total graph size: 305
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
    ## p_det   0.4976 0.28992 0.64828 0.0023672      0.0023891     1
    ## phi_occ 0.2372 0.03467 0.07753 0.0002831      0.0003587     1
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%    25%    50%    75%  97.5%
    ## p_det   0.0219 0.2437 0.4996 0.7508 0.9741
    ## phi_occ 0.1716 0.2136 0.2364 0.2598 0.3077

``` r
plot(Occ.DC)
```

![](README_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

There are a couple of diagnostic tools available in ‘dclone’ for the
non-estimability issue.

``` r
dcdiag(Occ.DC)
```

    ##   n.clones lambda.max  ms.error  r.squared    r.hat
    ## 1        1 0.08331270 0.1545713 0.01613750 1.000252
    ## 2        2 0.08272574 0.1548808 0.01353781 1.000144
    ## 3        5 0.08405302 0.1435564 0.01205429 1.000359

``` r
dcdiag(Occ.DC) |> plot()
```

![](README_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

1.  Check if the `lambda.max` is converging to zero. The rate at which
    this converges to zero should be approximately 1/*K*.
2.  If the variance is converging to 0 but the rate is different than
    1/*K*, it implies the asymptotics is of a different order. If you
    find such an example in your work, please let me know.
3.  Check the pairs plot to see if the likelihood function is converging
    to a manifold instead of a single point.

**Availability of the estimability diagnostics is one of the most
important features of data cloning. It will warn you if your scientific
inferences could be misleading.**

If the parameters are non-estimable, the only recourse one has is to
change the model (add assumptions or collect different kind of data).
There is always a possibility that, although the full parameter space
may not be estimable, a function of the parameter might be estimable. If
such a function is also of scientific interest, we can safely conduct
scientific inferences based on estimates of such a function of the
parameters.

### Replicate surveys

Suppose we visit the same cell several times. Assume that the visits are
independent of each other and the true occupancy status remains the same
throughout these surveys, then we can estimate the parameters. The model
can be written as a hierarchical model:

- Hierarchy 1:
  ![W\_{i}\sim Bernoulli(\phi)](https://latex.codecogs.com/svg.image?W_%7Bi%7D%5Csim%20Bernoulli%28%5Cphi%29 "W_{i}\sim Bernoulli(\phi)")
- Hierarchy 2:
  ![Y\_{ij}\|W\_{i}=w\_{i}\sim Bernoulli(p w\_{i})](https://latex.codecogs.com/svg.image?Y_%7Bij%7D%7CW_%7Bi%7D%3Dw_%7Bi%7D%5Csim%20Bernoulli%28p%20w_%7Bi%7D%29 "Y_{ij}|W_{i}=w_{i}\sim Bernoulli(p w_{i})")

Notice that hierarchy 2 depends on hierarchy 1 result.

We can easily modify the earlier code to allow for multiple surveys. We
will do such a modification with two surveys for each cell.

``` r
phi.true = 0.3
p.true = 0.7
n = 30
v = 2 # number of visits
W = rbinom(n, 1, phi.true)
Y = NULL
for (j in 1:v)
    Y <- cbind(Y, rbinom(n, 1, W * p.true))
```

### Bayesian inference using JAGS and dclone

Step 1: we need to define the model function.

``` r
Occ.model = function(){
  # Likelihood: Latent variables are random variables.
  for (i in 1:n){
    W[i] ~ dbin(phi_occ, 1)
    for (j in 1:v){
        Y[i,j] ~ dbin(W[i] * p_det, 1)}
  }
  # Priors
  phi_occ ~ dbeta(1, 1)
  p_det ~ dbeta(1, 1)
}
```

Now we need to provide the data to the model and generate random numbers
from the posterior. We will discuss different options later.

``` r
dat = list(Y=Y, n=n, v=v)
ini = list(W=rep(1, n))

Occ.Bayes = jags.fit(data=dat, params=c("phi_occ","p_det"), model=Occ.model,
    inits=ini)
```

    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 60
    ##    Unobserved stochastic nodes: 32
    ##    Total graph size: 125
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
    ## p_det   0.4897 0.1882 0.001537       0.005513
    ## phi_occ 0.2951 0.1613 0.001317       0.006542
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##            2.5%    25%    50%    75%  97.5%
    ## p_det   0.14242 0.3481 0.4878 0.6337 0.8350
    ## phi_occ 0.09779 0.1874 0.2564 0.3548 0.7631

``` r
plot(Occ.Bayes)
```

![](README_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

``` r
pairs(Occ.Bayes)
```

![](README_files/figure-gfm/unnamed-chunk-16-2.png)<!-- -->

These are nice unimodal posterior distributions.

We can modify the data cloning code to check if the parameters are, in
fact, estimable.

``` r
Occ.model.dc = function(){
  # Likelihood
  for(k in 1:ncl){
    for (i in 1:n){
      W[i,k] ~ dbin(phi_occ, 1)
      for (j in 1:v){
        Y[i,j,k] ~ dbin(p_det * W[i,k], 1)
      }
    }
  }
  # Prior
  phi_occ ~ dbeta(1, 1)
  p_det ~ dbeta(1, 1)
}
```

We need to turn the original data into an array. And we need to add
another index `ncl` for the cloned dimension. It gets multiplied by the
number of clones.

``` r
Y = array(Y, dim=c(n,v,1))
Y = dcdim(Y)
dat = list(Y=Y, n=n, v=v, ncl=1)

# As previously, we need to initiate the W's. 
ini = list(W=array(rep(1, n), dim=c(n, 1)))
# We need to clone these initial values as well. You should always check if this is doing the right job.
initfn =function(model, n.clones){
  W=array(rep(1,n), dim=c(n,1))
  list(W=dclone(dcdim(W), n.clones))
}
Occ.DC = dc.fit(data=dat, params=c("phi_occ","p_det"), model=Occ.model.dc,
    n.clones=c(1, 2, 5),
    unchanged=c("n","v"), multiply="ncl",
    inits=ini, initsfun=initfn)
```

    ## 
    ## Fitting model with 1 clone 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 60
    ##    Unobserved stochastic nodes: 32
    ##    Total graph size: 126
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
    ##    Observed stochastic nodes: 120
    ##    Unobserved stochastic nodes: 62
    ##    Total graph size: 246
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
    ##    Observed stochastic nodes: 300
    ##    Unobserved stochastic nodes: 152
    ##    Total graph size: 606
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
    ##           Mean      SD  DC SD  Naive SE Time-series SE R hat
    ## p_det   0.5503 0.09675 0.2163 0.0007900       0.002250 1.003
    ## phi_occ 0.2205 0.05039 0.1127 0.0004114       0.001235 1.005
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%    25%    50%    75%  97.5%
    ## p_det   0.3511 0.4855 0.5540 0.6176 0.7315
    ## phi_occ 0.1389 0.1857 0.2149 0.2486 0.3350

``` r
plot(Occ.DC)
```

![](README_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

``` r
dcdiag(Occ.DC)
```

    ##   n.clones lambda.max  ms.error r.squared    r.hat
    ## 1        1 0.05348012 0.3339641 0.0425344 1.001282
    ## 2        2 0.02618914 1.2255274 0.1315938 1.014639
    ## 3        5 0.01034708 0.9043499 0.1184760 1.003717

``` r
plot(dcdiag(Occ.DC))
```

![](README_files/figure-gfm/unnamed-chunk-18-2.png)<!-- -->

## Random effects in regression: Why and when?

We have shown how hierarchical models can be used to deal with
measurement error. Now we will look at a few examples where we use them
to combine data across several studies.

We will start with a simple (but extremely important) example that
started the entire field of mixture as well as hierarchical models
(Neyman and Scott, 1949).

Researchers in animal husbandary wanted to know how to improve the stock
of animals such as milk cows and bulls. This played an important role in
the ‘white revolution’ that lead to improving the nutrition in many
countries. Following is a somewhat made up and highly simplified
situation.

Suppose we have *n* cows. We want to know which cows have good genetic
potential that can be passed on to the next generation. Each cow might
have only a few calves. We measure the amount of milk by each calf.

Let ![Y_ij](https://latex.codecogs.com/svg.image?Y_ij "Y_ij") be the
amount of milk produced by the j-th calf of the i-th cow. We can
consider a linear model that uses the ‘cow effect’ (genetic) and
‘environmental effect’ to explain the amount of milk. This is same as
one way ANOVA model.

![Y\_{ij}=\mu+\alpha\_{i}+\epsilon\_{ij}](https://latex.codecogs.com/svg.image?Y_%7Bij%7D%3D%5Cmu%2B%5Calpha_%7Bi%7D%2B%5Cepsilon_%7Bij%7D "Y_{ij}=\mu+\alpha_{i}+\epsilon_{ij}")

Under the usual Gaussian error structure, we know that

![Y\_{ij}\sim N(\mu\_{i},\sigma^{2})](https://latex.codecogs.com/svg.image?Y_%7Bij%7D%5Csim%20N%28%5Cmu_%7Bi%7D%2C%5Csigma%5E%7B2%7D%29 "Y_{ij}\sim N(\mu_{i},\sigma^{2})")
where
![i=1,2,...,n](https://latex.codecogs.com/svg.image?i%3D1%2C2%2C...%2Cn "i=1,2,...,n")
and ![j=1,2](https://latex.codecogs.com/svg.image?j%3D1%2C2 "j=1,2")
(![\mu\_{i}=\mu + \alpha_i](https://latex.codecogs.com/svg.image?%5Cmu_%7Bi%7D%3D%5Cmu%20%2B%20%5Calpha_i "\mu_{i}=\mu + \alpha_i")).

There are ![2 n](https://latex.codecogs.com/svg.image?2%20n "2 n")
observations and
![n+1](https://latex.codecogs.com/svg.image?n%2B1 "n+1") parameters. The
number of parameters increases at the same rate as the number of
observations. Note that the ratio of parameters to observations
converges to 0.5. Roughly speaking, for the MLE to work, this ratio
needs to go to 0. Generally the number of parameters is fixed and hence
this condition is satisfied.

We simply do not have much information about each
![\mu_i](https://latex.codecogs.com/svg.image?%5Cmu_i "\mu_i") as there
are only two observations corresponding to it. It also turns out that
the ML estimator of
![\sigma^2](https://latex.codecogs.com/svg.image?%5Csigma%5E2 "\sigma^2")
converges to
![0.5\*\sigma^2](https://latex.codecogs.com/svg.image?0.5%2A%5Csigma%5E2 "0.5*\sigma^2").
Hence it is not consistent even though the number of observations
corresponding to it do converge to infinity. This was a major blow to
the theory of maximum likelihood. Although it turns out Fisher had
implicitly answered it a decade before this paper.

How can we reduce the number of parameters?

1.  If there are covariates such as weight of the mother, mother’s milk
    production are available, we can model
    ![\mu\_{i}=X\_{i}\beta](https://latex.codecogs.com/svg.image?%5Cmu_%7Bi%7D%3DX_%7Bi%7D%5Cbeta "\mu_{i}=X_{i}\beta").
2.  Suppose covariates are not available or difficult to assess what to
    use as a covariate. If we assume that cows are kind of similar to
    each other, then we can assume that they come from a population of
    cows such that
    ![\mu\_{i}\sim N(\mu,\tau^{2})](https://latex.codecogs.com/svg.image?%5Cmu_%7Bi%7D%5Csim%20N%28%5Cmu%2C%5Ctau%5E%7B2%7D%29 "\mu_{i}\sim N(\mu,\tau^{2})").
    It turns out, under such an assumption, we can estimate the
    parameters
    ![(\mu,\sigma^2,\tau^2)](https://latex.codecogs.com/svg.image?%28%5Cmu%2C%5Csigma%5E2%2C%5Ctau%5E2%29 "(\mu,\sigma^2,\tau^2)")
    consistently.

This model is a hierarchical model.

- Hierarchy 1:
  ![Y\_{ij}\|\mu_i\sim N(\mu\_{i},\sigma^{2})](https://latex.codecogs.com/svg.image?Y_%7Bij%7D%7C%5Cmu_i%5Csim%20N%28%5Cmu_%7Bi%7D%2C%5Csigma%5E%7B2%7D%29 "Y_{ij}|\mu_i\sim N(\mu_{i},\sigma^{2})")
- Hierarchy 2:
  ![\mu\_{i}\sim N(\mu,\tau^{2})](https://latex.codecogs.com/svg.image?%5Cmu_%7Bi%7D%5Csim%20N%28%5Cmu%2C%5Ctau%5E%7B2%7D%29 "\mu_{i}\sim N(\mu,\tau^{2})")

For a Bayesian approach, we put priors on the three parameters. This
forms the third hierarchy.

This simple model can be used in many different situations.

1.  Measurement error in covariates in regression
2.  Random intercept regression model to account for missing covariates
3.  Kalman filter models for time series with measurement error are of
    the same kind with a bit more complexity as we will see the third
    part.

Let us see what makes it a difficult model to analyze using the
likelihood approach.

In order to write down the likelihood function, we need to compute the
marginal distribution of the observations. Remember
![\mu_i](https://latex.codecogs.com/svg.image?%5Cmu_i "\mu_i") are not
observed. Hence we have to integrate over them.

![f(y\_{ij};\mu,\sigma^{2},\tau^{2})=\int f(y\_{ij}\|\mu\_{i})g(\mu\_{i})d\mu\_{i}](https://latex.codecogs.com/svg.image?f%28y_%7Bij%7D%3B%5Cmu%2C%5Csigma%5E%7B2%7D%2C%5Ctau%5E%7B2%7D%29%3D%5Cint%20f%28y_%7Bij%7D%7C%5Cmu_%7Bi%7D%29g%28%5Cmu_%7Bi%7D%29d%5Cmu_%7Bi%7D "f(y_{ij};\mu,\sigma^{2},\tau^{2})=\int f(y_{ij}|\mu_{i})g(\mu_{i})d\mu_{i}")

Again, this is not a precise statement but it gives you the idea. This
integral is one dimensional and hence can be computed analytically and
also numerically. However, in many cases the dimension of the integral
is quite large and hence neither of these solutions are available. In
some cases, one can obtain Laplace approximation to this integral (INLA
and related methods rely on this approximation). The most general
approach is based on the MCMC algorithm.

``` r
n = 30
mu_true = 2
tau_true = 0.5
sigma_true = 1

mu = rnorm(n, mu_true, tau_true)
Y = cbind(
    rnorm(n, mu, sigma_true),
    rnorm(n, mu, sigma_true))
```

We will write the Bayesian model first. Remember the normal distribution
is defined in terms of the precision (inverse of the variance).

``` r
LM_Bayes = function(){
  # Likelihood
  for (i in 1:n){
    mu[i] ~ dnorm(mu.t, prec_tau)
    for (j in 1:2){
      Y[i,j] ~ dnorm(mu[i], prec_sigma)
    }
  }
  # Priors
  mu.t ~ dnorm(0, 0.01)
  prec_tau ~ dgamma(0.1, 0.1)
  prec_sigma ~ dgamma(0.1, 0.1)
  # Parameters on the natural scale
  tau <- sqrt(1/prec_tau)
  sigma <- sqrt(1/prec_sigma)
}
```

Data in array form for data cloning purpose

``` r
Y = as.array(Y, dim=c(n,2,1))
Y = dcdim(Y)

dat = list(Y=Y, n=n)
LM_Bayes_fit = jags.fit(data=dat, params=c("mu.t","tau","sigma"), model=LM_Bayes)
```

    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 60
    ##    Unobserved stochastic nodes: 33
    ##    Total graph size: 102
    ## 
    ## Initializing model

``` r
summary(LM_Bayes_fit)
```

    ## 
    ## Iterations = 1001:6000
    ## Thinning interval = 1 
    ## Number of chains = 3 
    ## Sample size per chain = 5000 
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##         Mean     SD  Naive SE Time-series SE
    ## mu.t  1.9332 0.1495 0.0012206       0.002868
    ## sigma 0.9273 0.1041 0.0008503       0.001514
    ## tau   0.4556 0.1518 0.0012392       0.004392
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##         2.5%    25%    50%    75%  97.5%
    ## mu.t  1.6409 1.8343 1.9322 2.0335 2.2285
    ## sigma 0.7450 0.8550 0.9195 0.9912 1.1525
    ## tau   0.2051 0.3421 0.4430 0.5535 0.7864

We will modify it to do data cloning. This is useful to assure us that
the parameters are estimable. It is also important for making it
invariant to the choice of the priors and parameterization.

``` r
LM_DC = function(){
  # Likelihood
  for (k in 1:ncl){
    for (i in 1:n){
      mu[i,k] ~ dnorm(mu.t, prec_tau)
      for (j in 1:2){
        Y[i,j,k] ~ dnorm(mu[i,k], prec_sigma)
      }
    }
  }
  # Priors
  mu.t ~ dnorm(0, 0.01)
  prec_tau ~ dgamma(0.1, 0.1)
  prec_sigma ~ dgamma(0.1, 0.1)
  # Parameters on the natural scale
  tau <- sqrt(1/prec_tau)
  sigma <- sqrt(1/prec_sigma)
}
```

Data in array form for data cloning purpose.

``` r
Y = array(Y, dim=c(n,2,1))
Y = dcdim(Y)

dat = list(Y=Y, n=n, ncl=1)
LM_DC_fit = dc.fit(data=dat, params=c("mu.t","tau","sigma"), model=LM_DC,
    n.clones=c(1, 2, 5), unchanged="n", multiply="ncl")
```

    ## 
    ## Fitting model with 1 clone 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 60
    ##    Unobserved stochastic nodes: 33
    ##    Total graph size: 103
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
    ##    Observed stochastic nodes: 120
    ##    Unobserved stochastic nodes: 63
    ##    Total graph size: 193
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
    ##    Observed stochastic nodes: 300
    ##    Unobserved stochastic nodes: 153
    ##    Total graph size: 463
    ## 
    ## Initializing model

``` r
summary(LM_DC_fit)
```

    ## 
    ## Iterations = 1001:6000
    ## Thinning interval = 1 
    ## Number of chains = 3 
    ## Sample size per chain = 5000 
    ## Number of clones = 5
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##         Mean      SD  DC SD  Naive SE Time-series SE R hat
    ## mu.t  1.9369 0.06302 0.1409 0.0005145       0.001334 1.003
    ## sigma 0.9251 0.04848 0.1084 0.0003958       0.001242 1.002
    ## tau   0.3865 0.08627 0.1929 0.0007044       0.003845 1.007
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##         2.5%    25%    50%    75%  97.5%
    ## mu.t  1.8126 1.8939 1.9375 1.9802 2.0602
    ## sigma 0.8330 0.8920 0.9237 0.9578 1.0224
    ## tau   0.2174 0.3274 0.3852 0.4476 0.5529

``` r
pairs(LM_DC_fit)
```

![](README_files/figure-gfm/unnamed-chunk-23-1.png)<!-- -->

``` r
dcdiag(LM_DC_fit)
```

    ##   n.clones  lambda.max   ms.error   r.squared    r.hat
    ## 1        1 0.024605910 0.22765240 0.023800795 1.002562
    ## 2        2 0.015053325 0.05660741 0.008518448 1.007700
    ## 3        5 0.008131706 0.00871139 0.001428311 1.007885

``` r
plot(dcdiag(LM_DC_fit))
```

![](README_files/figure-gfm/unnamed-chunk-23-2.png)<!-- -->

The best thing about the MCMC approach is that we can modify the
prototype program to do Generalized linear mixed models.

Let us see how we can change the program to do Poisson regression with
random intercepts. This is useful for accounting for missing covariates
in the usual Poisson regression. This can also be used to account for
site effect in abundance surveys.

Mathematically the model is:

- Hierarchy 1:
  ![log(\lambda\_{i})\sim N(log(\lambda),\tau^{2})](https://latex.codecogs.com/svg.image?log%28%5Clambda_%7Bi%7D%29%5Csim%20N%28log%28%5Clambda%29%2C%5Ctau%5E%7B2%7D%29 "log(\lambda_{i})\sim N(log(\lambda),\tau^{2})")
- Hierarchy 2:
  ![Y\_{i}\|\lambda\_{i}\sim Poisson(\lambda\_{i})](https://latex.codecogs.com/svg.image?Y_%7Bi%7D%7C%5Clambda_%7Bi%7D%5Csim%20Poisson%28%5Clambda_%7Bi%7D%29 "Y_{i}|\lambda_{i}\sim Poisson(\lambda_{i})")

``` r
n = 30
mu_true = 2
tau_true = 0.5
sigma_true = 1

mu = rnorm(n, mu_true, tau_true)
Y = rpois(n, exp(mu))
```

We will write the Bayesian model first. Remember the normal distribution
is defined in terms of the precision (inverse of the variance).

``` r
GLMM_Bayes = function(){
  # Likelihood
  for (i in 1:n){
    mu[i] ~ dnorm(mu.t, prec_tau)
    Y[i] ~ dpois(exp(mu[i]))
  }
  # Priors
  mu.t ~ dnorm(0, 0.01)
  prec_tau ~ dgamma(0.1, 0.1)
  # Parameters on the natural scale
  tau <- sqrt(1/prec_tau)
  lambda <- exp(mu.t)
}

dat = list(Y=Y, n=n)
GLMM_Bayes_fit = jags.fit(data=dat, params=c("lambda","tau"), model=GLMM_Bayes)
```

    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 30
    ##    Unobserved stochastic nodes: 32
    ##    Total graph size: 100
    ## 
    ## Initializing model

``` r
summary(GLMM_Bayes_fit)
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
    ##         Mean     SD  Naive SE Time-series SE
    ## lambda 7.111 0.8704 0.0071072        0.01179
    ## tau    0.532 0.1078 0.0008799        0.00203
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##          2.5%    25%    50%    75%  97.5%
    ## lambda 5.4940 6.5145 7.0784 7.6643 8.9063
    ## tau    0.3482 0.4561 0.5221 0.5992 0.7648

As usual, we can turn the crank for data cloning to check the
estimability of the parameters.

``` r
GLMM_DC = function(){
  # Likelihood
  for (k in 1:ncl){
    for (i in 1:n){
      mu[i,k] ~ dnorm(mu.t,prec_tau)
      Y[i,k] ~ dpois(exp(mu[i,k]))
    }
  }
  # Priors
  mu.t ~ dnorm(0, 0.01)
  prec_tau ~ dgamma(0.1, 0.1)
  # Parameters on the natural scale
  tau <- sqrt(1/prec_tau)
  lambda <- exp(mu.t)
}

# Data in array form for data cloning purpose

Y = array(Y, dim=c(n,1))
Y = dcdim(Y)

dat = list(Y=Y, n=n, ncl=1)
GLMM_DC_fit = dc.fit(data=dat, params=c("lambda","tau"), model=GLMM_DC,
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
    ##    Unobserved stochastic nodes: 32
    ##    Total graph size: 101
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
    ##    Unobserved stochastic nodes: 62
    ##    Total graph size: 191
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
    ##    Unobserved stochastic nodes: 152
    ##    Total graph size: 461
    ## 
    ## Initializing model

``` r
summary(GLMM_DC_fit)
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
    ## lambda 7.0842 0.38081 0.8515 0.0031093      0.0055526 1.001
    ## tau    0.5079 0.04625 0.1034 0.0003776      0.0008574 1.002
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##          2.5%    25%    50%    75%  97.5%
    ## lambda 6.3515 6.8223 7.0826 7.3433 7.8313
    ## tau    0.4236 0.4755 0.5061 0.5377 0.6033

``` r
exp(mu_true)
```

    ## [1] 7.389056

``` r
tau_true
```

    ## [1] 0.5

### Estimating the effect of a treatment from multi-center clinical trials

In clinical trials, we are interested in estimating the effect of the
treatment. One of the simplest forms of clinical trial is where we split
a group of patients in two groups randomly. One of the group gets the
treatment and the other gets a placebo.

We can then estimate the difference in the outcomes. This may be done
using a simple t-test if the outcome is a continuous measurement. If the
patients are quite different from each other in terms of say age, blood
pressure or some such physical characteristics that may affect the
outcome, we adjust them by using a regression approach. We include these
other covariates and the treatment/control indicator variable in the
model. The effect of the treatment after adjusting for other covariates
can be studied using such a regression model.

Often in practice, we may not have access to such covariates. Using
random intercept model in regression is one way out of such a situation.
In this case, we do consider the differences between patients but
without ascribing them to any specific, known values of the covariates.
The model one may consider is:

![Y\_{i}=\beta\_{0i}+\beta I\_{(Treatment)}+\epsilon\_{i}](https://latex.codecogs.com/svg.image?Y_%7Bi%7D%3D%5Cbeta_%7B0i%7D%2B%5Cbeta%20I_%7B%28Treatment%29%7D%2B%5Cepsilon_%7Bi%7D "Y_{i}=\beta_{0i}+\beta I_{(Treatment)}+\epsilon_{i}")

As we have seen in the previous example (Neyman-Scott problem), this
model is non-estimable. One way to make it estimable is by using a
hierarchical structure:

Hierarchy 2:
![\beta\_{0i}\sim N(\beta\_{0},\tau^{2})](https://latex.codecogs.com/svg.image?%5Cbeta_%7B0i%7D%5Csim%20N%28%5Cbeta_%7B0%7D%2C%5Ctau%5E%7B2%7D%29 "\beta_{0i}\sim N(\beta_{0},\tau^{2})")

Homework: Check the validity of the following statement without doing
any mathematics. You can use data cloning to do that.

This leads to estimability for
![\beta](https://latex.codecogs.com/svg.image?%5Cbeta "\beta"), the
parameter of interest. Although it does not lead to estimation of the
variances
![(\tau,\sigma)](https://latex.codecogs.com/svg.image?%28%5Ctau%2C%5Csigma%29 "(\tau,\sigma)").

> An approach not described in this course: We can use profile
> likelihood for the parameter
> ![\beta](https://latex.codecogs.com/svg.image?%5Cbeta "\beta"). This
> eliminates the ‘nuisance parameters’
> ![\beta_0,\tau,\sigma](https://latex.codecogs.com/svg.image?%5Cbeta_0%2C%5Ctau%2C%5Csigma "\beta_0,\tau,\sigma").
> Computing the profile likelihood and quantification of uncertainty for
> inferences based on it for hierarchical models can be tackled using
> data cloning. This could be another course!

Suppose the outcome is binary, survival for 5 years vs. failure before 5
years. In this case, a convenient model is a binary regression model
such as a Logistic regression model.

Hierarchy 1:

![P(Y\_{i}=1)=\frac{exp(\beta\_{0i}+\beta I\_{(Treatment)})}{1+exp(\beta\_{0i}+\beta I\_{(Treatment)})}](https://latex.codecogs.com/svg.image?P%28Y_%7Bi%7D%3D1%29%3D%5Cfrac%7Bexp%28%5Cbeta_%7B0i%7D%2B%5Cbeta%20I_%7B%28Treatment%29%7D%29%7D%7B1%2Bexp%28%5Cbeta_%7B0i%7D%2B%5Cbeta%20I_%7B%28Treatment%29%7D%29%7D "P(Y_{i}=1)=\frac{exp(\beta_{0i}+\beta I_{(Treatment)})}{1+exp(\beta_{0i}+\beta I_{(Treatment)})}")

Hierarchy 2:
![\beta\_{0i}\sim N(\beta\_{0},\tau^{2})](https://latex.codecogs.com/svg.image?%5Cbeta_%7B0i%7D%5Csim%20N%28%5Cbeta_%7B0%7D%2C%5Ctau%5E%7B2%7D%29 "\beta_{0i}\sim N(\beta_{0},\tau^{2})")

This is an example of a Generalized Linear Mixed Model. Fortunately, for
this model all parameters are estimable as we will see using data
cloning. The random intercept here could be accounting for differences
in the clinical centers (hospitals), assuming we have only two patients,
one in control and one in the treatment group. If we have multiple
patients in each group, we may include another random effect to account
for differences in the patients within each group. For example, we may
consider a model:

Hierarchy 1:

![P(Y\_{ij}=1)=\frac{exp(\beta\_{0i}+\beta I\_{(Treatment)}+\beta\_{1j})}{1+exp(\beta\_{0i}+\beta I\_{(Treatment)}+\beta\_{1j})}](https://latex.codecogs.com/svg.image?P%28Y_%7Bij%7D%3D1%29%3D%5Cfrac%7Bexp%28%5Cbeta_%7B0i%7D%2B%5Cbeta%20I_%7B%28Treatment%29%7D%2B%5Cbeta_%7B1j%7D%29%7D%7B1%2Bexp%28%5Cbeta_%7B0i%7D%2B%5Cbeta%20I_%7B%28Treatment%29%7D%2B%5Cbeta_%7B1j%7D%29%7D "P(Y_{ij}=1)=\frac{exp(\beta_{0i}+\beta I_{(Treatment)}+\beta_{1j})}{1+exp(\beta_{0i}+\beta I_{(Treatment)}+\beta_{1j})}")

Hierarchy 2:
![\beta\_{0i}\sim N(\beta\_{0},\tau^{2})](https://latex.codecogs.com/svg.image?%5Cbeta_%7B0i%7D%5Csim%20N%28%5Cbeta_%7B0%7D%2C%5Ctau%5E%7B2%7D%29 "\beta_{0i}\sim N(\beta_{0},\tau^{2})"),
![\beta\_{1j}\sim N(\beta\_{1},\tau^{2})](https://latex.codecogs.com/svg.image?%5Cbeta_%7B1j%7D%5Csim%20N%28%5Cbeta_%7B1%7D%2C%5Ctau%5E%7B2%7D%29 "\beta_{1j}\sim N(\beta_{1},\tau^{2})")

We can include interactions between random effects and so on. We will
not go into these complex models in this course.

Let us see how one can modify the prototype program to analyze the
random intercept Logistic regression model.

Each center has one control and one treatment patient.

``` r
n = 300      # Number of clinical centers
mu_true = 0
tau_true = 0.5
delta = 1   # Treatment effect
mu = rnorm(n, mu_true, tau_true)
Y = cbind(
    rbinom(n, 1, plogis(mu)),
    rbinom(n, 1, plogis(mu+delta)))
```

We will write the Bayesian model first. Remember the normal distribution
is defined in terms of the precision (inverse of the variance).

``` r
GLMM_Bayes = function(){
  # Likelihood
  for (i in 1:n){
    mu[i] ~ dnorm(mu.t,prec_tau)
    Y[i,1] ~ dbin(ilogit(mu[i]),1)
    Y[i,2] ~ dbin(ilogit(mu[i]+delta),1)
  }
  # Priors
  mu.t ~ dnorm(0, 0.01)
  prec_tau ~ dgamma(0.1, 0.1)
  delta ~ dnorm(0, 0.1)
  # Parameters on the natural scale
  tau <- sqrt(1/prec_tau)
}

dat = list(Y=Y, n=n)
GLMM_Bayes_fit = jags.fit(data=dat, params=c("delta","mu.t","tau"), model=GLMM_Bayes)
```

    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 600
    ##    Unobserved stochastic nodes: 303
    ##    Total graph size: 1810
    ## 
    ## Initializing model

``` r
summary(GLMM_Bayes_fit)
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
    ##          Mean     SD Naive SE Time-series SE
    ## delta 1.05477 0.1900 0.001551       0.006965
    ## mu.t  0.05857 0.1304 0.001065       0.006024
    ## tau   0.65582 0.2130 0.001739       0.023416
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##          2.5%      25%     50%    75%  97.5%
    ## delta  0.6869  0.92595 1.05174 1.1799 1.4319
    ## mu.t  -0.1927 -0.02852 0.05729 0.1439 0.3234
    ## tau    0.2760  0.50037 0.65182 0.7941 1.1043

We will modify this to get the MLE using data cloning.

``` r
GLMM_DC = function(){
  # Likelihood
  for (k in 1:ncl){
    for (i in 1:n){
      mu[i,k] ~ dnorm(mu.t,prec_tau)
      Y[i,1,k] ~ dbin(ilogit(mu[i,k]),1)
      Y[i,2,k] ~ dbin(ilogit(mu[i,k]+delta),1)
    }
  }
  # Priors
  mu.t ~ dnorm(0, 0.01)
  prec_tau ~ dgamma(0.1, 0.1)
  delta ~ dnorm(0, 0.1)
  # Parameters on the natural scale
  tau <- sqrt(1/prec_tau)
}

Y = array(Y, dim=c(dim(Y),1))
Y = dcdim(Y)
dat = list(Y=Y, n=n, ncl=1)

GLMM_DC_fit = dc.fit(data=dat, params=c("delta","mu.t","tau"), model=GLMM_DC,
    n.clones=c(1, 2, 5), unchanged="n", multiply="ncl")
```

    ## 
    ## Fitting model with 1 clone 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 600
    ##    Unobserved stochastic nodes: 303
    ##    Total graph size: 1811
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
    ##    Observed stochastic nodes: 1200
    ##    Unobserved stochastic nodes: 603
    ##    Total graph size: 3611
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
    ##    Observed stochastic nodes: 3000
    ##    Unobserved stochastic nodes: 1503
    ##    Total graph size: 9011
    ## 
    ## Initializing model

``` r
summary(GLMM_DC_fit)
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
    ## delta 1.03814 0.08686 0.1942 0.0007092       0.003383 1.004
    ## mu.t  0.06138 0.05546 0.1240 0.0004529       0.002425 1.003
    ## tau   0.62338 0.11858 0.2651 0.0009682       0.016359 1.036
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##           2.5%     25%    50%     75%  97.5%
    ## delta  0.87030 0.97888 1.0373 1.09674 1.2117
    ## mu.t  -0.04754 0.02392 0.0614 0.09916 0.1691
    ## tau    0.38629 0.54300 0.6204 0.70369 0.8555

``` r
pairs(GLMM_DC_fit)
```

![](README_files/figure-gfm/unnamed-chunk-29-1.png)<!-- -->

``` r
dcdiag(GLMM_DC_fit)
```

    ##   n.clones lambda.max    ms.error    r.squared    r.hat
    ## 1        1 0.06330392 0.008709029 0.0008659365 1.043179
    ## 2        2 0.03887969 0.021926101 0.0020470154 1.094722
    ## 3        5 0.01613287 0.004337083 0.0005956888 1.026414

``` r
plot(dcdiag(GLMM_DC_fit))
```

![](README_files/figure-gfm/unnamed-chunk-29-2.png)<!-- -->

Following is a real data set on multi-center clinical trials. Number of
patients in the treatment are `"nt"` and those who survived are `"rt"`.
Similarly `"nc"` are the number of patients in the control group and
`"rc"` are the ones that survived.

``` r
OR.data = list(
    "rt" =
        c(3, 7, 5, 102, 28, 4, 98, 60, 25, 138, 64, 45, 9, 57, 25, 33, 
        28, 8, 6, 32, 27, 22),
    "nt" =
        c(38, 114, 69, 1533, 355, 59, 945, 632, 278, 1916, 873, 263, 
        291, 858, 154, 207, 251, 151, 174, 209, 391, 680),
    "rc" =
        c(3, 14, 11, 127, 27, 6, 152, 48, 37, 188, 52, 47, 16, 45, 31, 
        38, 12, 6, 3, 40, 43, 39),
    "nc" =
        c(39, 116, 93, 1520, 365, 52, 939, 471, 282, 1921, 583, 266, 
        293, 883, 147, 213, 122, 154, 134, 218, 364, 674),
    "Num" =
        22)
```

The model functions for DD are as follows.

``` r
DC.MLE.fn = function() {
  for (k in 1:ncl){
    for (i in 1:Num) {
      rt[i,k] ~ dbin(pt[i,k], nt[i,k])
      rc[i,k] ~ dbin(pc[i,k], nc[i,k])
      logit(pc[i,k]) <- mu[i,k] 
      logit(pt[i,k]) <- mu[i,k] + delta
      mu[i,k] ~ dnorm(alpha,tau1)
    }
  }
  # Priors 
  parms ~ dmnorm(MuP, PrecP)
  delta <- parms[1]
  tau1 <- exp(parms[3])
  sigma1 <- 1/sqrt(tau1)
  alpha <- parms[2]
}

# Analysis
rt = dcdim(array(OR.data$rt, dim=c(22,1)))
nt = dcdim(array(OR.data$nt, dim=c(22,1)))
rc = dcdim(array(OR.data$rc, dim=c(22,1)))
nc = dcdim(array(OR.data$nt, dim=c(22,1)))
Num = OR.data$Num

dat = list(rt=rt, rc=rc, nt=nt, nc=nc, Num=Num, ncl=1, 
    MuP=rep(0, 3), PrecP=diag(0.01, 3, 3))

DC.MLE = dc.fit(data=dat, params="parms", model=DC.MLE.fn,
    n.clones=c(1, 4, 16),
    multiply="ncl", unchanged=c("Num","MuP","PrecP"),
    n.chains=5, n.update=1000, n.iter=5000, n.adapt=2000)
```

    ## 
    ## Fitting model with 1 clone 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 44
    ##    Unobserved stochastic nodes: 23
    ##    Total graph size: 200
    ## 
    ## Initializing model
    ## 
    ## 
    ## Fitting model with 4 clones 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 176
    ##    Unobserved stochastic nodes: 89
    ##    Total graph size: 728
    ## 
    ## Initializing model
    ## 
    ## 
    ## Fitting model with 16 clones 
    ## 
    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 704
    ##    Unobserved stochastic nodes: 353
    ##    Total graph size: 2840
    ## 
    ## Initializing model

``` r
# Check the convergence and MLE estimates etc.
summary(DC.MLE)
```

    ## 
    ## Iterations = 3001:8000
    ## Thinning interval = 1 
    ## Number of chains = 5 
    ## Sample size per chain = 5000 
    ## Number of clones = 16
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##             Mean      SD   DC SD  Naive SE Time-series SE R hat
    ## parms[1] -0.1955 0.01225 0.04899 7.746e-05      0.0002699 1.001
    ## parms[2] -2.2621 0.02869 0.11478 1.815e-04      0.0008476 1.002
    ## parms[3]  1.3643 0.08443 0.33773 5.340e-04      0.0060733 1.022
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##             2.5%     25%     50%     75%   97.5%
    ## parms[1] -0.2194 -0.2035 -0.1956 -0.1873 -0.1715
    ## parms[2] -2.3187 -2.2809 -2.2617 -2.2433 -2.2038
    ## parms[3]  1.1880  1.3103  1.3667  1.4197  1.5219

``` r
dctable(DC.MLE)
```

    ## $`parms[1]`
    ##   n.clones       mean         sd       2.5%        25%        50%        75%
    ## 1        1 -0.1962588 0.05060023 -0.2909296 -0.2302265 -0.1964363 -0.1612282
    ## 2        4 -0.1951291 0.02474131 -0.2443361 -0.2114888 -0.1953264 -0.1784273
    ## 3       16 -0.1955156 0.01224777 -0.2193745 -0.2035219 -0.1955766 -0.1873386
    ##         97.5%    r.hat
    ## 1 -0.09691651 1.001245
    ## 2 -0.14633269 1.000799
    ## 3 -0.17148523 1.000919
    ## 
    ## $`parms[2]`
    ##   n.clones      mean         sd      2.5%       25%       50%       75%
    ## 1        1 -2.265504 0.12071400 -2.508465 -2.345284 -2.266183 -2.186500
    ## 2        4 -2.263082 0.05873369 -2.378808 -2.300076 -2.262953 -2.224601
    ## 3       16 -2.262134 0.02869399 -2.318724 -2.280857 -2.261724 -2.243336
    ##       97.5%    r.hat
    ## 1 -2.029327 1.006588
    ## 2 -2.148635 1.006884
    ## 3 -2.203778 1.002123
    ## 
    ## $`parms[3]`
    ##   n.clones     mean        sd      2.5%      25%      50%      75%    97.5%
    ## 1        1 1.289238 0.3487657 0.5772857 1.055585 1.300952 1.522908 1.945893
    ## 2        4 1.359040 0.1722342 1.0018284 1.243065 1.364020 1.475816 1.693881
    ## 3       16 1.364287 0.0844322 1.1880357 1.310287 1.366748 1.419670 1.521893
    ##      r.hat
    ## 1 1.011778
    ## 2 1.021351
    ## 3 1.021884
    ## 
    ## attr(,"class")
    ## [1] "dctable"

``` r
dcdiag(DC.MLE)
```

    ##   n.clones  lambda.max    ms.error   r.squared    r.hat
    ## 1        1 0.121693755 0.021720949 0.003552573 1.011516
    ## 2        4 0.029688078 0.009894563 0.001291571 1.018007
    ## 3       16 0.007128992 0.017287583 0.001384011 1.017402

It should be clear by now that we can modify any Bayesian analysis to
get Maximum likelihood estimate quite easily by adding one dimension to
the data and a do loop over the clones.

In the next part, we will discuss how to analyzed time series data.
