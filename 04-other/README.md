Other considerations
================

## What is going on inside jags.fit

``` r
library(dclone)
```

    ## Loading required package: coda

    ## Loading required package: parallel

    ## Loading required package: Matrix

    ## dclone 2.3-1      2023-04-07

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

    ## Registered S3 method overwritten by 'R2WinBUGS':
    ##   method            from  
    ##   as.mcmc.list.bugs dclone

    ## Compiling model graph
    ##    Resolving undeclared variables
    ##    Allocating nodes
    ## Graph information:
    ##    Observed stochastic nodes: 30
    ##    Unobserved stochastic nodes: 3
    ##    Total graph size: 157
    ## 
    ## Initializing model

What just happened? `jags.fit` is a wrapper around some rjags functions.

``` r
m <- jagsModel(file = Normal.model, data=dat, n.chains=3)
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
m
```

    ## JAGS model:
    ## 
    ## model
    ## {
    ##     for (i in 1:n) {
    ##         mu[i] <- X[i, ] %*% beta
    ##         Y[i] ~ dnorm(mu[i], prec.e)
    ##     }
    ##     beta[1] ~ dnorm(0.00000E+00, 1)
    ##     beta[2] ~ dnorm(0.00000E+00, 1)
    ##     prec.e ~ dlnorm(0.00000E+00, 1)
    ## }
    ## Fully observed variables:
    ##  X Y n

``` r
str(m)
```

    ## List of 8
    ##  $ ptr      :function ()  
    ##  $ data     :function ()  
    ##  $ model    :function ()  
    ##  $ state    :function (internal = FALSE)  
    ##  $ nchain   :function ()  
    ##  $ iter     :function ()  
    ##  $ sync     :function ()  
    ##  $ recompile:function ()  
    ##  - attr(*, "class")= chr "jags"

``` r
str(m$state())
```

    ## List of 3
    ##  $ :List of 2
    ##   ..$ beta  : num [1:2] 0.263 0.972
    ##   ..$ prec.e: num 0.699
    ##  $ :List of 2
    ##   ..$ beta  : num [1:2] 0.174 0.791
    ##   ..$ prec.e: num 0.483
    ##  $ :List of 2
    ##   ..$ beta  : num [1:2] 0.444 1.011
    ##   ..$ prec.e: num 0.942

``` r
m$iter()
```

    ## [1] 1000

``` r
update(m, n.iter=1000)
str(m$state())
```

    ## List of 3
    ##  $ :List of 2
    ##   ..$ beta  : num [1:2] 0.439 0.746
    ##   ..$ prec.e: num 0.984
    ##  $ :List of 2
    ##   ..$ beta  : num [1:2] 0.227 0.958
    ##   ..$ prec.e: num 1.44
    ##  $ :List of 2
    ##   ..$ beta  : num [1:2] 0.384 1.029
    ##   ..$ prec.e: num 1.27

``` r
m$iter()
```

    ## [1] 2000

``` r
s = codaSamples(m, variable.names=c("beta","prec.e"), n.iter=5000)
str(s)
```

    ## List of 3
    ##  $ : 'mcmc' num [1:5000, 1:3] 0.5 0.395 0.62 0.573 0.616 ...
    ##   ..- attr(*, "dimnames")=List of 2
    ##   .. ..$ : NULL
    ##   .. ..$ : chr [1:3] "beta[1]" "beta[2]" "prec.e"
    ##   ..- attr(*, "mcpar")= num [1:3] 2001 7000 1
    ##  $ : 'mcmc' num [1:5000, 1:3] 0.356 0.318 0.441 0.505 0.54 ...
    ##   ..- attr(*, "dimnames")=List of 2
    ##   .. ..$ : NULL
    ##   .. ..$ : chr [1:3] "beta[1]" "beta[2]" "prec.e"
    ##   ..- attr(*, "mcpar")= num [1:3] 2001 7000 1
    ##  $ : 'mcmc' num [1:5000, 1:3] 0.291 0.66 0.54 0.578 0.543 ...
    ##   ..- attr(*, "dimnames")=List of 2
    ##   .. ..$ : NULL
    ##   .. ..$ : chr [1:3] "beta[1]" "beta[2]" "prec.e"
    ##   ..- attr(*, "mcpar")= num [1:3] 2001 7000 1
    ##  - attr(*, "class")= chr "mcmc.list"

``` r
head(s)
```

    ## [[1]]
    ## Markov Chain Monte Carlo (MCMC) output:
    ## Start = 2001 
    ## End = 2007 
    ## Thinning interval = 1 
    ##        beta[1]   beta[2]    prec.e
    ## [1,] 0.4999702 1.1724731 0.8450824
    ## [2,] 0.3949369 1.0562144 1.0063648
    ## [3,] 0.6198544 1.2625788 1.0068026
    ## [4,] 0.5727043 0.7912519 1.1818847
    ## [5,] 0.6163789 1.2179580 1.0801854
    ## [6,] 0.5630512 1.0102842 0.7010217
    ## [7,] 0.5646638 1.2335906 0.4164947
    ## 
    ## [[2]]
    ## Markov Chain Monte Carlo (MCMC) output:
    ## Start = 2001 
    ## End = 2007 
    ## Thinning interval = 1 
    ##        beta[1]   beta[2]    prec.e
    ## [1,] 0.3558273 0.9595604 1.0754907
    ## [2,] 0.3184937 0.8077153 1.3064703
    ## [3,] 0.4410759 1.1061628 1.0994428
    ## [4,] 0.5046203 0.9194636 0.8254754
    ## [5,] 0.5395639 1.1012889 0.8143211
    ## [6,] 0.6802857 1.2644899 0.7532393
    ## [7,] 0.3899092 0.6136866 0.8488133
    ## 
    ## [[3]]
    ## Markov Chain Monte Carlo (MCMC) output:
    ## Start = 2001 
    ## End = 2007 
    ## Thinning interval = 1 
    ##        beta[1]   beta[2]    prec.e
    ## [1,] 0.2905594 1.3513154 0.8892932
    ## [2,] 0.6601448 0.9939320 1.0118439
    ## [3,] 0.5398437 1.2293463 0.7318512
    ## [4,] 0.5781334 0.4120254 0.7631129
    ## [5,] 0.5427442 1.1709146 1.3170710
    ## [6,] 0.5047020 1.0234947 1.5600661
    ## [7,] 0.3930740 0.8976199 1.2267152

``` r
m$iter()
```

    ## [1] 7000

``` r
update(m, n.iter=1000)
m$iter()
```

    ## [1] 8000

``` r
s = codaSamples(m, variable.names=c("beta","prec.e"), n.iter=5000)
head(s)
```

    ## [[1]]
    ## Markov Chain Monte Carlo (MCMC) output:
    ## Start = 8001 
    ## End = 8007 
    ## Thinning interval = 1 
    ##        beta[1]   beta[2]    prec.e
    ## [1,] 0.3390798 0.2149953 0.3913921
    ## [2,] 0.4433881 0.9715009 0.6712223
    ## [3,] 0.2823311 0.9190776 1.2208282
    ## [4,] 0.1715156 1.0516969 0.5759351
    ## [5,] 0.3369730 1.2823903 0.8464790
    ## [6,] 0.3810078 0.8227776 0.9153048
    ## [7,] 0.4337844 1.0712739 0.7679734
    ## 
    ## [[2]]
    ## Markov Chain Monte Carlo (MCMC) output:
    ## Start = 8001 
    ## End = 8007 
    ## Thinning interval = 1 
    ##        beta[1]   beta[2]    prec.e
    ## [1,] 0.7041280 0.6777114 0.7371299
    ## [2,] 0.2879974 0.8439683 0.8478328
    ## [3,] 0.1394623 0.9368820 1.0423896
    ## [4,] 0.1387999 0.8677913 1.0611940
    ## [5,] 0.7583891 0.7647948 0.6352911
    ## [6,] 0.6280370 1.0915991 1.1247642
    ## [7,] 0.1986587 1.0989896 0.6778598
    ## 
    ## [[3]]
    ## Markov Chain Monte Carlo (MCMC) output:
    ## Start = 8001 
    ## End = 8007 
    ## Thinning interval = 1 
    ##        beta[1]   beta[2]    prec.e
    ## [1,] 0.4161634 1.0705933 0.8965556
    ## [2,] 0.7119730 1.1706593 0.9039597
    ## [3,] 0.5711140 0.9858748 0.7587962
    ## [4,] 0.4191208 1.2629605 0.7476215
    ## [5,] 0.6516293 1.0254174 0.6495790
    ## [6,] 0.1550864 0.7057860 0.7677134
    ## [7,] 0.1015106 0.8090942 0.8217568

``` r
m$iter()
```

    ## [1] 13000

Can we further update `Normal.Bayes`?

``` r
m2 <- updated.model(Normal.Bayes)
m2$iter()
```

    ## [1] 7000

``` r
update(m2, n.iter=1000)
m2$iter()
```

    ## [1] 8000

Couple of things to cover here:

``` r
str(formals(jags.fit))
```

    ## Dotted pair list of 11
    ##  $ data         : symbol 
    ##  $ params       : symbol 
    ##  $ model        : symbol 
    ##  $ inits        : NULL
    ##  $ n.chains     : num 3
    ##  $ n.adapt      : num 1000
    ##  $ n.update     : num 1000
    ##  $ thin         : num 1
    ##  $ n.iter       : num 5000
    ##  $ updated.model: logi TRUE
    ##  $ ...          : symbol

Implicitly, we do:

``` r
Normal.Bayes = jags.fit(
    data=dat, 
    params=c("beta","prec.e"),
    model=Normal.model,
    inits=NULL,
    n.chains=3,
    n.adapt=1000,
    n.update=1000,
    thin=1,
    n.iter=5000,
    updated.model=TRUE)
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

## Working with MCMC lists

``` r
mcmcapply(Normal.Bayes, sd)
```

    ##   beta[1]   beta[2]    prec.e 
    ## 0.1952770 0.2120569 0.2387348

``` r
mcmcapply(Normal.Bayes, quantile, c(0.05, 0.5, 0.95))
```

    ##       beta[1]   beta[2]    prec.e
    ## 5%  0.1360275 0.6189342 0.5668411
    ## 50% 0.4547150 0.9752258 0.8970265
    ## 95% 0.7734413 1.3148416 1.3378016

``` r
quantile(Normal.Bayes, c(0.05, 0.5, 0.95))
```

    ##       beta[1]   beta[2]    prec.e
    ## 5%  0.1360275 0.6189342 0.5668411
    ## 50% 0.4547150 0.9752258 0.8970265
    ## 95% 0.7734413 1.3148416 1.3378016

``` r
str(as.matrix(Normal.Bayes))
```

    ##  num [1:15000, 1:3] 0.515 0.141 0.179 0.396 0.706 ...
    ##  - attr(*, "dimnames")=List of 2
    ##   ..$ : NULL
    ##   ..$ : chr [1:3] "beta[1]" "beta[2]" "prec.e"

We need to talk about RNGs

``` r
library(rjags)
```

    ## Linked to JAGS 4.3.1

    ## Loaded modules: basemod,bugs

``` r
str(parallel.seeds("base::BaseRNG", 5))
```

    ## List of 5
    ##  $ :List of 2
    ##   ..$ .RNG.name : chr "base::Marsaglia-Multicarry"
    ##   ..$ .RNG.state: int [1:2] 1697000109 706257735
    ##  $ :List of 2
    ##   ..$ .RNG.name : chr "base::Super-Duper"
    ##   ..$ .RNG.state: int [1:2] 121285605 -1731591065
    ##  $ :List of 2
    ##   ..$ .RNG.name : chr "base::Mersenne-Twister"
    ##   ..$ .RNG.state: int [1:625] 1 1117868619 1001738153 1101463288 1170119206 1411584833 2031548627 1302051970 -385811788 -1242190281 ...
    ##  $ :List of 2
    ##   ..$ .RNG.name : chr "base::Wichmann-Hill"
    ##   ..$ .RNG.state: int [1:3] 10025 20690 12289
    ##  $ :List of 2
    ##   ..$ .RNG.name : chr "base::Marsaglia-Multicarry"
    ##   ..$ .RNG.state: int [1:2] 2106934083 407033467

``` r
## The lecuyer module provides the RngStream factory, which allows large
## numbers of independent parallel RNGs to be generated. 
load.module("lecuyer")
```

    ## module lecuyer loaded

``` r
list.factories(type="rng")
```

    ##              factory status
    ## 1 lecuyer::RngStream   TRUE
    ## 2      base::BaseRNG   TRUE

``` r
str(parallel.seeds("lecuyer::RngStream", 5))
```

    ## List of 5
    ##  $ :List of 2
    ##   ..$ .RNG.name : chr "lecuyer::RngStream"
    ##   ..$ .RNG.state: int [1:6] -1485395108 1504502298 743804168 21870502 640342388 -1077103118
    ##  $ :List of 2
    ##   ..$ .RNG.name : chr "lecuyer::RngStream"
    ##   ..$ .RNG.state: int [1:6] -2076215521 -363145817 -293725617 989721991 764139962 771567617
    ##  $ :List of 2
    ##   ..$ .RNG.name : chr "lecuyer::RngStream"
    ##   ..$ .RNG.state: int [1:6] -1857214677 -1733734863 -1367758950 -2120546901 -1817530441 323247534
    ##  $ :List of 2
    ##   ..$ .RNG.name : chr "lecuyer::RngStream"
    ##   ..$ .RNG.state: int [1:6] -1145481088 1016419490 -2030786848 1344650391 -559694556 697385543
    ##  $ :List of 2
    ##   ..$ .RNG.name : chr "lecuyer::RngStream"
    ##   ..$ .RNG.state: int [1:6] -162704308 -1292796771 -1271231871 -2060200158 -1708657971 1964613031

## Can we run chains in parallel?

``` r
system.time({
    Normal.Bayes = jags.fit(
        data=dat, params=c("beta","prec.e"), model=Normal.model,
        n.chains=4,
        n.update=10^6)
})
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

    ##    user  system elapsed 
    ##  18.688   0.047  18.742

``` r
system.time({
    Normal.Bayes = jags.parfit(
        cl=10,
        data=dat, params=c("beta","prec.e"), model=Normal.model,
        n.chains=4,
        n.update=10^6)
})
```

    ## 
    ## Parallel computation in progress

    ##    user  system elapsed 
    ##  19.446   0.079   4.914

## I like DC but I don’t have time …

Well, check `dc.parfit`

``` r
## determine the number of workers needed
clusterSize(1:5)
```

    ##   workers none load size both
    ## 1       1   15   15   15   15
    ## 2       2   12    9    8    8
    ## 3       3    9    7    5    5
    ## 4       4    7    6    5    5
    ## 5       5    5    5    5    5

``` r
## visually compare balancing options
opar <- par(mfrow=c(2, 2))
plotClusterSize(2,1:5, "none")
plotClusterSize(2,1:5, "load")
plotClusterSize(2,1:5, "size")
plotClusterSize(2,1:5, "both")
```

![](README_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
par(opar)
```

Parallel chains, size balancing, both.
