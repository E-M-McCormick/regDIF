#' Simulated data example with multiple DIF covariates
#'
#' A simulated dataset containing six binary items and three DIF covariates,
#' designed to demonstrate the \code{regDIF} function. The data were generated
#' from a 2PL IRT model where selected items have DIF effects on the
#' intercept and/or slope parameters as a function of age, gender, and study.
#'
#' @format A data frame with 500 rows and 9 variables:
#' \describe{
#'   \item{item1}{Binary item response (0/1)}
#'   \item{item2}{Binary item response (0/1)}
#'   \item{item3}{Binary item response (0/1)}
#'   \item{item4}{Binary item response (0/1)}
#'   \item{item5}{Binary item response (0/1)}
#'   \item{item6}{Binary item response (0/1)}
#'   \item{age}{Continuous covariate (potential source of DIF)}
#'   \item{gender}{Binary covariate (potential source of DIF)}
#'   \item{study}{Binary covariate (potential source of DIF)}
#' }
#'
#' @examples
#' data(ida)
#' head(ida)
#'
#' # Items are in columns 1-6, covariates in columns 7-9
#' item.data <- ida[, 1:6]
#' pred.data <- ida[, 7:9]
"ida"
