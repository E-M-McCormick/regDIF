###############################################################################
# Print Method for regDIF Objects                                             #
#                                                                             #
# Displays a concise summary of the regularization path: the original call,   #
# and a table of tau values with corresponding BIC values.                    #
###############################################################################

#' Print a fitted regDIF model object.
#'
#' Displays the model call and a summary table of the regularization path
#' showing tau values and their corresponding BIC values.
#'
#' @param x Fitted regDIF model object.
#' @param ... Additional arguments to be passed through \code{print}.
#'
#' @rdname print.regDIF
#'
#' @return Invisibly returns \code{x}. Called for its side effect of printing.
#' @export

print.regDIF <-
  function(x, ...) {
    # Display the original function call.
    cat("Call:\n")
    print(x$call)

    # Create and display the regularization path summary.
    table <- data.frame(
      "tau" = x$tau_vec,
      "bic" = x$bic
    )
    cat("\nregDIF results:\n")
    print(table)
  }
