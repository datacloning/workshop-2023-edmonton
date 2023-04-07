pkgs <- c(
    "shiny",
    "shinydashboard")

inst <- rownames(installed.packages())
needed_pkgs <- setdiff(pkgs, inst)
if (length(needed_pkgs) > 0L)
    install.packages(needed_pkgs, repos="https://cloud.r-project.org/")
for (i in pkgs)
    library(i, character.only=TRUE)
