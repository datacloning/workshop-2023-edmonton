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
install.packages(c('rjags', 'dclone', 'shiny', 'ggplot2', 'mgcv', 'R2WinBUGS'))
```

## Test your installation

If you can run this example without error, you are all set:

```R
example("jags.fit", "dclone", run.dontrun = TRUE)
```
