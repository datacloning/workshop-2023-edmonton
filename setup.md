# Install software for the workshop

## Install JAGS

Follow instructions from <https://mcmc-jags.sourceforge.io/>.

```bash
# Ubuntu Linux
sudo apt update -qq && sudo apt install --yes --no-install-recommends jags

# Mac
brew install jags
```

Windows: download the installed from [here](https://sourceforge.net/projects/mcmc-jags/files/JAGS/4.x/Windows/).

## Install R packages

From R:

```R
# CRAN packages
install.packages(c("rjags", "shiny", "R2WinBUGS"))

# latest dclone version from R-universe
install.packages("dclone", repos = 'https://datacloning.r-universe.dev')
```

## Test your installation

If you can run this example without error, you are all set:

```R
example("jags.fit", package = "dclone", run.dontrun = TRUE)
```
