#' regDIF: Regularized Differential Item Functioning
#'
#' \pkg{regDIF} performs regularization of differential item functioning (DIF)
#' in item response theory (IRT) and confirmatory factor analysis (CFA) models
#' using a penalized expectation-maximization (EM) algorithm. The package
#' implements LASSO, MCP, group LASSO, and group MCP penalties to automatically
#' select which items exhibit DIF and on which covariates.
#'
#' The main function is \code{\link{regDIF}}, which fits a sequence of
#' penalized IRT/CFA models across a grid of tuning parameter (tau) values.
#' Model selection is performed via AIC or BIC.
#'
#' @section Supported item types:
#' \itemize{
#'   \item \code{"2pl"}: Two-parameter logistic model for binary items
#'   \item \code{"rasch"}: Rasch model (slopes fixed to 1) for binary items
#'   \item \code{"graded"}: Graded response model for ordinal items
#'   \item \code{"cfa"}: Gaussian model for continuous items
#' }
#'
#' @section Key references:
#' Belzak, W. C. M. & Bauer, D. J. (2020). Improving the assessment of
#' measurement invariance: Using regularization to select anchor tests and
#' identify differential item functioning. \emph{Psychological Methods},
#' 25(6), 673-690.
#'
#' @name regDIF-package
#' @docType package
#' @title Regularized differential item functioning for IRT and CFA models.
#' @author William Belzak \email{wbelzak@@gmail.com}
#' @exportMethod coef plot print
#' @keywords package
NULL
