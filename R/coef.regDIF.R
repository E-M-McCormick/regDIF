###############################################################################
# Coefficient Extraction for regDIF Objects                                   #
#                                                                             #
# S3 method for extracting model coefficients from a fitted regDIF object.    #
# Supports three modes:                                                       #
#   1. tau = NULL: return all coefficients across the full regularization path #
#   2. tau = "tau.min": return coefficients at the optimal tau (min AIC/BIC)  #
#   3. tau = numeric: return coefficients at specific tau index(es)           #
###############################################################################

#' Extract coefficients from a fitted regDIF model.
#'
#' Returns impact, base item, and DIF parameters for one or more
#' values of the tuning parameter tau.
#'
#' @param object Fitted regDIF model object.
#' @param tau Optional character or numeric indicating the tau(s) at
#' which the model coefficients are returned. For character value, may be
#' \code{"tau.min"}, which returns model coefficients for the value of tau
#' at which the minimum fit statistic is identified. For numeric, the value(s)
#' provided corresponds to the index(es) along the tau path.
#' @param method Character value indicating the model fit statistic to be used
#' for determining \code{"tau.min"}. Default is \code{"bic"}. May also be
#' \code{"aic"}.
#' @param ... Additional arguments to be passed through to \code{coef}.
#'
#' @rdname coef.regDIF
#'
#' @return A list with components: \code{tau} (tuning parameter value(s)),
#'   \code{impact} (latent mean/variance parameters), \code{base} (item
#'   intercepts and slopes), and \code{dif} (DIF effects).
#' @export

coef.regDIF <-
  function(object, tau = NULL, method = "bic", ...) {

      # Return coefficients based on the tau selection mode.
      if(is.null(tau)){
        table <- list("tau" = object$tau_vec,
                      "impact" = object$impact,
                      "base" = object$base,
                      "dif" = object$dif)

      } else if(tau == "tau.min") {
        if(method == "aic") {
          table <- list("tau" =
                          object$tau_vec[which.min(object$aic)],
                        "impact" =
                          object$impact[,which.min(object$aic)],
                        "base" =
                          object$base[,which.min(object$aic)],
                        "dif" =
                          object$dif[,which.min(object$aic)])

        } else if(method == "bic") {
          table <- list("tau" =
                          object$tau_vec[which.min(object$bic)],
                        "impact" =
                          object$impact[,which.min(object$bic)],
                        "base" =
                          object$base[,which.min(object$bic)],
                        "dif" =
                          object$dif[,which.min(object$bic)])
        }

      } else if(is.numeric(tau)) {
        table <- list("tau" =
                        object$tau_vec[tau],
                      "impact" =
                        object$impact[,tau],
                      "base" =
                        object$base[,tau],
                      "dif" =
                        object$dif[,tau])
      }


    return(table)
  }


