###############################################################################
# Plot Method for regDIF Objects                                              #
#                                                                             #
# Creates a regularization path plot showing how DIF parameter estimates      #
# change across values of log(tau). Non-zero DIF effects at the optimal       #
# model (min AIC/BIC) are highlighted with colored lines, while zero          #
# effects are shown in gray. A vertical dashed line marks the optimal tau.    #
###############################################################################

#' Plot the regularization path for a fitted regDIF model.
#'
#' Creates a line plot showing DIF parameter estimates as a function of
#' log(tau). Effects that are non-zero at the optimal model are highlighted
#' with distinct colors and line types; zero effects are shown in gray.
#'
#' @param x Fitted regDIF model object.
#' @param y Unused; included for S3 method compatibility.
#' @param method Character value indicating which fit statistic to use
#'   for identifying the optimal model. Default is \code{"bic"}.
#' @param color.seed Integer random seed for sampling line colors and
#'   line types for non-zero DIF effects. Default is 123.
#' @param legend.plot Logical indicating whether to include a legend
#'   identifying non-zero DIF effects. Default is \code{TRUE}.
#' @param ... Additional arguments to be passed through to \code{plot}.
#'
#' @rdname plot.regDIF
#'
#' @importFrom graphics abline legend lines plot
#'
#' @return Invisibly returns \code{NULL}. Called for its side effect of plotting.
#'
#' @export
#'
plot.regDIF <-
  function(x, y = NULL, method = "bic", color.seed = 123, legend.plot = TRUE, ...) {

    # Transform tau to log scale for more interpretable x-axis.
    tau <- log(x$tau_vec)
    if(length(tau) < 2) stop(
      paste0("Must run multiple tau values to plot."),
      call. = TRUE)

    # Extract intercept and slope DIF parameters across the tau path.
    dif.parms <- x$dif[grep(paste0(c("int","slp"),
                                                   collapse = "|"),
                                            rownames(x$dif)), ]

    # Identify the optimal tau and which DIF effects are non-zero there.
    min.tau <- tau[which.min(unlist(x[method]))]
    dif.min.tau <- dif.parms[,which(tau == min.tau)]
    nonzero.dif <- dif.min.tau[!(dif.min.tau == 0)]

    # # Find first tau with non-zero dif parms.
    #   for(j in 1:ncol(dif.parms)){
    #     if(sum(abs(dif.parms[,j])) > 0){
    #       first.tau <- j
    #       break
    #     } else{
    #       next
    #     }
    #   }

    # Draw the base plot with a horizontal zero line.
    # X-axis is reversed (large tau on left, small on right) to show
    # the regularization path from most to least penalized.
    plot(tau,
         rep(0,length(tau)),
         type = 'l',
         xlim = c(max(tau, na.rm = T), min(tau, na.rm = T)),
         ylim = c(min(dif.parms, na.rm = TRUE), max(dif.parms, na.rm = TRUE)),
         main = "Regularization Path",
         xlab = expression(log(tau)),
         ylab = "Estimate")
    # Vertical dashed line at the optimal tau.
    abline(v = min.tau, lty = 2)

      # Draw each DIF parameter's path across tau values.
      # Non-zero effects at optimal tau get colored lines; zero effects are gray.
      dif.lines <- matrix(NA,ncol=2,nrow=nrow(dif.parms))
      for(i in 1:nrow(dif.parms)){
        if(rownames(dif.parms)[i] %in% names(nonzero.dif)){
          set.seed(color.seed+i)
          linecolor <- sample(grDevices::colors()[grep('gr(a|e)y',
                                                       grDevices::colors(),
                                                       invert = T)], 1)
          linetype <- sample(1:6,1)
          lwdnum <- 2
        } else {
          linecolor <- 'gray72'
          linetype <- 1
          lwdnum <- 1
        }

        lines(tau,dif.parms[i,], col = linecolor, lty = linetype, lwd = lwdnum)
        dif.lines[i,1] <- linecolor
        dif.lines[i,2] <- linetype

      }
    lines(c(tau, tau[1]),
          rep(0,length(tau)+1),
          type = 'l',
          xlim = c(tau[1],max(tau, na.rm = TRUE)))
    nonzero.dif.lines <- dif.lines[!(dif.min.tau == 0)]
    if(legend.plot & (length(names(nonzero.dif)) != 0)) {
      legend("topleft",
             legend = names(nonzero.dif),
             col = nonzero.dif.lines[1:length(nonzero.dif)],
             lty = as.numeric(
               nonzero.dif.lines[(length(nonzero.dif)+1):(length(nonzero.dif)*2)]
             ),
             lwd = 2,
             cex = 0.75,
             bty = "n",
             y.intersp = .1,
             inset = c(0,-.1))
    }


  }
