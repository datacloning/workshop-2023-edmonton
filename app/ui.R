ui <- dashboardPage(
  dashboardHeader(title = "Data Cloning Apps"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Distributions", tabName = "distributions"),
      menuItem("MLE", tabName = "mle"),
      menuItem("Beta prior", tabName = "betaprior"),
      menuItem("Normal prior", tabName = "normalprior"),
      menuItem("Bimodal prior", tabName = "bimodalprior"),
      menuItem("Data cloning", tabName = "datacloning")
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(tabName = "distributions",
        fluidRow(
          box(title="Histogram",
            plotOutput("distPlot")
          ),
          box(title="Inputs",
            selectInput("distr", "Distribution",
                  choices=c(Bernoulli = "Bernoulli",
                Binomial = "Binomial",
                Poisson = "Poisson",
                Normal = "Normal",
                Lognormal = "Lognormal",
                Uniform = "Uniform",
                Beta = "Beta",
                Gamma = "Gamma")),
            hr(),
              sliderInput("n", label = "Sample size",
                          min = 10, max = 1000, value = 100, step = 1),
              sliderInput("seed", label = "Random seed",
                          min = 0, max = 100, value = 0, step = 1),
              ## Bernoulli
              conditionalPanel(
                condition = "input.distr == 'Bernoulli'",
                  sliderInput("p", label = "Probability",
                          min = 0, max = 1, value = 0.3, step = 0.05)),
              ## Binomial
              conditionalPanel(
                condition = "input.distr == 'Binomial'",
                  sliderInput("p", label = "Probability",
                          min = 0, max = 1, value = 0.3, step = 0.05),
                  sliderInput("size", label = "Size",
                          min = 1, max = 1000, value = 10, step = 1)),
              ## Poisson
              conditionalPanel(
                condition = "input.distr == 'Poisson'",
                  sliderInput("lambda", label = "Mean/Rate",
                          min = 0, max = 100, value = 5, step = 1)),
              ## Normal
              conditionalPanel(
                condition = "input.distr == 'Normal'",
                  sliderInput("mu", label = "Mean",
                          min = -10, max = 10, value = 0, step = 0.1),
                  sliderInput("var", label = "Variance",
                          min = 0.001, max = 10, value = 1, step = 0.1)),
              ## Logormal
              conditionalPanel(
                condition = "input.distr == 'Lognormal'",
                  sliderInput("mux", label = "Mean",
                          min = -10, max = 10, value = -1, step = 0.1),
                  sliderInput("varx", label = "Variance",
                          min = 0.001, max = 10, value = 1, step = 0.1)),
              ## Uniform
              conditionalPanel(
                condition = "input.distr == 'Uniform'",
                  sliderInput("a", label = "Minimum",
                          min = -10, max = 10, value = -1, step = 0.1),
                  sliderInput("b", label = "Maximum",
                          min = -10, max = 10, value = 1, step = 0.1)),
              ## Beta
              conditionalPanel(
                condition = "input.distr == 'Beta'",
                  sliderInput("shape1", label = "Shape 2",
                          min = 0, max = 10, value = 1, step = 0.1),
                  sliderInput("shape2", label = "Shape 1",
                          min = 0, max = 10, value = 1, step = 0.1)),
              ## Gamma
              conditionalPanel(
                condition = "input.distr == 'Gamma'",
                  sliderInput("shape", label = "Shape",
                          min = 0.001, max = 10, value = 1, step = 0.1),
                  sliderInput("rate", label = "Rate",
                          min = 0.001, max = 10, value = 1, step = 0.1))
            )
          )
        ),
      tabItem(tabName = "mle",
        fluidRow(
            box(title="Density",
                plotOutput("mlePlot")),
            box(title="Inputs",
                sliderInput("p_mle", label = "Probability (true)",
                    min = 0, max = 1, value = 0.3, step = 0.05),
                sliderInput("n_mle", label = "Sample size",
                    min = 10, max = 1000, value = 10, step = 10),
                sliderInput("seed_mle", label = "Random seed",
                    min = 0, max = 100, value = 0, step = 10)
                )
        )
      ),
      tabItem(tabName = "betaprior",
        fluidRow(
            box(title="Density",
                plotOutput("betaPlot")),
            box(title="Inputs",
              sliderInput("p_beta", label = "Probability (true)",
                          min = 0, max = 1, value = 0.3, step = 0.05),
              sliderInput("n_beta", label = "Sample size",
                          min = 1, max = 1000, value = 10, step = 10),
              sliderInput("a_beta", label = "Beta prior shape parameter a",
                          min = 0, max = 2, value = 1, step = 0.1),
              sliderInput("b_beta", label = "Beta prior shape parameter b",
                          min = 0, max = 2, value = 1, step = 0.1),
              radioButtons("scale_beta", label="Scale",
                         c("Probability (0, 1)" = "prob",
                           "Logit (-Inf, Inf)" = "logit")),
              sliderInput("seed_beta", label = "Random seed",
                          min = 0, max = 100, value = 0, step = 10)
                )
        )
      ),
      tabItem(tabName = "normalprior",
        fluidRow(
            box(title="Density",
                plotOutput("normPlot")),
            box(title="Inputs",
                sliderInput("p_norm", label = "Probability (true)",
                    min = 0, max = 1, value = 0.3, step = 0.05),
                sliderInput("n_norm", label = "Sample size",
                    min = 1, max = 1000, value = 10, step = 10),
                sliderInput("mu_norm", label = "Normal prior mean",
                    min = -10, max = 10, value = 0, step = 1),
                sliderInput("sig2_norm", label = "Normal prior variance",
                    min = 0.001, max = 100, value = 1, step = 10),
                radioButtons("scale_norm", label="Scale",
                    c("Probability (0, 1)" = "prob",
               "Logit (-Inf, Inf)" = "logit")),
                sliderInput("seed_norm", label = "Random seed",
                    min = 0, max = 100, value = 0, step = 10)
                )
        )
      ),
      tabItem(tabName = "bimodalprior",
        fluidRow(
            box(title="Density",
                plotOutput("bimodPlot")),
            box(title="Inputs",
                sliderInput("p_bimod", label = "Probability (true)",
                  min = 0, max = 1, value = 0.3, step = 0.05),
                sliderInput("n_bimod", label = "Sample size",
                  min = 0, max = 1000, value = 10, step = 10),
                sliderInput("mu_1_bimod", label = "Normal prior mean",
                  min = -10, max = 10, value = -2, step = 1),
                sliderInput("sig2_1_bimod", label = "Normal prior variance",
                  min = 0.001, max = 10, value = 1, step = 1),
                sliderInput("mu_2_bimod", label = "Normal prior mean",
                  min = -10, max = 10, value = 2, step = 1),
                sliderInput("sig2_2_bimod", label = "Normal prior variance",
                  min = 0.001, max = 10, value = 2, step = 1),
                radioButtons("scale_bimod", label="Scale",
                 c("Probability (0, 1)" = "prob",
                   "Logit (-Inf, Inf)" = "logit"),
                 selected = "logit"),
                sliderInput("seed_bimod", label = "Random seed",
                  min = 0, max = 100, value = 0, step = 10)
                )
        )
      ),
      tabItem(tabName = "datacloning",
        fluidRow(
            box(title="Density",
                plotOutput("dcPlot")),
            box(title="Inputs",
                sliderInput("p_dc", label = "Probability (true)",
                  min = 0, max = 1, value = 0.3, step = 0.05),
                sliderInput("n_dc", label = "Sample size",
                  min = 1, max = 50, value = 10, step = 5),
                sliderInput("a_dc", label = "Beta prior shape parameter a",
                  min = 0, max = 2, value = 1, step = 0.1),
                sliderInput("b_dc", label = "Beta prior shape parameter b",
                  min = 0, max = 2, value = 1, step = 0.1),
                sliderInput("K_dc", label = "Number of clones",
                  min = 1, max = 100, value = 1, step = 10),
                radioButtons("scale_dc", label="Scale",
                 c("Probability (0, 1)" = "prob",
                   "Logit (-Inf, Inf)" = "logit")),
                sliderInput("seed_dc", label = "Random seed",
                  min = 0, max = 100, value = 0, step = 10)
                )
        )
      )

    )
  )
)
