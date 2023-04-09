# workshop-2023-edmonton
> Data Cloning Workshop 2023 - Edmonton

**CalgaryR & YEGRUG Meetup: Data Cloning - Hierarchical Models Made Easy**

## Attend

Jointly organized by the [Edmonton R User Group](https://yegrug.github.io/) & [CalgaryR](https://imstatsbee.github.io/calgaryr/)

This is a 4-hour long hybrid (in-person and online) event:

- Date: April 13, 2023
- Time: 2-6 pm
- Location: University of Alberta, Dept. Biological Sciences, room G-113 ([map](https://www.ualberta.ca/maps.html?l=53.52898,-113.526374&z=17&campus=north_campus&b=bs))

## Instructors

- [Subhash Lele](https://scholar.google.ca/citations?hl=en&user=1CNJm5UAAAAJ­), Professor Emeritus, Dept. of Mathematical and Statistical Sciences, U. of Alberta
- [Peter Solymos](https://peter.solymos.org/­), Senior Data Scientist, Organizer of Edmonton R User Group, R package author

## Sponsors

<img src="https://www.r-consortium.org/wp-content/uploads/sites/13/2016/09/RConsortium_Horizontal_Pantone.png" width="50%" alt="R Consortium logo" />  |  <img src="https://github.com/analythium/assets/raw/master/docs/marks/word-mark-dark-wide.png" width="50%" alt="Analythium logo" />
:-------------------------:|:-------------------------:
Meetup fees, publicity             |  Conferencing, catering

## Synopsis

Mixed models, also known as hierarchical models and multilevel models, is a useful class of models for applied sciences. The goal of this workshop is to give an introduction to the logic, theory, and implementation of these models to solve practical problems. The workshop will include a seminar style overview and hands on exercises including common model classes and examples that participants can extend for their own needs.

Participants are expected to know the basics of R programming and regression techniques. A basic idea about hierarchical models and Bayesian/MCMC techniques will help but is not required.

At the end of the workshop, you will have a good idea about why everyone is not a Bayesian, how to use Bayesian techniques for frequentist inference, how to critically diagnose identifiability issues, and how to make likelihood-based inference for (almost) any hierarchical modeling case.

## Before the course

Follow [instructions](setup.md) to set up your laptop, or wait for the link for a cloud server.

Read the [notes](./prior/) about statistical concepts and the workflow we will rely on, and play with the [Shiny app](./app/).

## Resources

| Topic    | Links |
| -------- | ------- |
| Setup  | [Instructions](setup.md)  |
| Preliminaries  | [Notes](./01-intro/), [App](./app/)  |
| References  | [PDF files](./docs/)  |

## License

The course material is shared under the 
[Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)
license. In publications, please use the following citations to refer to the workshop, theory, and software implementation:

- Lele, S. R., and Solymos, P., 2023. Data Cloning Workshop - Hierarchical Models Made Easy. April 13, 2023. URL <https://github.com/datacloning/workshop-2023-edmonton>
- Lele, S.R., B. Dennis and F. Lutscher, 2007. Data cloning: easy maximum likelihood estimation for complex ecological models using Bayesian Markov chain Monte Carlo methods. Ecology Letters 10, 551-563. [DOI 10.1111/j.1461-0248.2007.01047.x­](https://doi.org/10.1111/j.1461-0248.2007.01047.x)
- Lele, S. R., Nadeem, K., and Schmuland, B., 2010. Estimability and likelihood inference for generalized linear mixed models using data cloning. Journal of the American Statistical Association 105, 1617-1625. [DOI 10.1198/jasa.2010.tm09757­](https://doi.org/10.1198/jasa.2010.tm09757)
- Solymos, P., 2010. dclone: Data Cloning in R. The R Journal 2(2), 29-37. URL <https://journal.r-project.org/archive/2010/RJ-2010-011/RJ-2010-011.pdf>

For software license, please refer to the corresponding packages used during the workshop. JAGS, rjags, R2WinBUGS, and dclone are all licensed under [GPL-2](https://cran.r-project.org/web/licenses/GPL-2).
