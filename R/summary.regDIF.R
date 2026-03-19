###############################################################################
# Summary Method for regDIF Objects                                           #
#                                                                             #
# Displays the optimal model from the regularization path (selected by        #
# minimum AIC or BIC) and lists all non-zero DIF effects at that model.       #
###############################################################################

#' Summarize a fitted regDIF model.
#'
#' Displays the model call, the optimal tau value (minimizing AIC or BIC),
#' and all non-zero DIF effects identified at the optimal model.
#'
#' @param object Fitted regDIF model object.
#' @param method Character value indicating the fit statistic to minimize
#'   for selecting the optimal model. Default is \code{"bic"}; may also be
#'   \code{"aic"}.
#' @param ... Additional arguments to be passed through \code{summary}.
#'
#' @rdname summary.regDIF
#'
#' @return Invisibly returns \code{NULL}. Called for its side effect of printing.
#' @export

summary.regDIF <-
  function(object, method = "bic", ...) {
    # Display the original function call.
    cat("Call:\n")
    print(object$call)

    # Identify the optimal model and extract non-zero DIF effects.
    if(method == "aic") {
      sum_results <- c(object$tau_vec[which.min(object$aic)],
                       object$aic[which.min(object$aic)])
      dif <- object$dif[object$dif[,which.min(object$aic)] != 0, which.min(object$aic)]
      if(length(grep(".res.", names(dif))) != 0) dif <- dif[-grep(".res.", names(dif))]
      names(sum_results) <- c("tau","aic")

    } else if(method == "bic") {
      sum_results <- c(object$tau_vec[which.min(object$bic)],
                       object$bic[which.min(object$bic)])
      dif <- object$dif[object$dif[,which.min(object$bic)] != 0, which.min(object$bic)]
      if(length(grep(".res.", names(dif))) != 0) dif <- dif[-grep(".res.", names(dif))]
      names(sum_results) <- c("tau","bic")

    }

    # Print the results table.
    cat(paste0("\nOptimal model (out of ", length(object$tau_vec),"):\n"))
    print(sum_results)
    cat("\nNon-zero DIF effects:\n")
    if(length(dif) != 0) print(dif)
  }
