###############################################################################
### M-step: Univariate Newton-Raphson (UNR) Parameter Updates
###############################################################################
#
# This file implements the maximization (M) step of the EM algorithm for the
# regDIF model using univariate Newton-Raphson (UNR) optimization. In UNR,
# each parameter is updated one at a time while holding all other parameters
# fixed, using the Newton-Raphson formula:
#
#   p_new = p_old - gradient / hessian
#
# where gradient is the first derivative and hessian is the second derivative
# of the observed-data log-likelihood with respect to that parameter.
#
# The update order is:
#   1. Impact mean parameters (alpha) -- one per covariate
#   2. Impact variance parameters (phi) -- one per covariate
#   3. Item parameters, looping over all items:
#      a. Base intercept (c0) -- unpenalized
#      b. Base slope (a0) -- unpenalized (skipped for Rasch models)
#      c. Residual variance (s0) -- CFA items only, unpenalized
#      d. Residual DIF (s1) -- CFA items only
#      e. DIF intercept (c1) -- penalized via thresholding operators
#      f. DIF slope (a1) -- penalized via thresholding operators (skipped for
#         Rasch models)
#
# Three item types are supported, each with its own derivative functions:
#   - "cfa": Continuous/Gaussian items (d_mu_gaussian, d_sigma_gaussian)
#   - "2pl"/"rasch": Binary items using a 2PL or Rasch model (d_bernoulli)
#   - Graded/ordinal items with multiple thresholds (d_categorical)
#
# DIF parameters are regularized using penalty-specific thresholding operators:
#   - Lasso: soft_threshold()
#   - MCP: firm_threshold()
#   - Group lasso: grp_soft_threshold() -- jointly shrinks intercept + slope DIF
#   - Group MCP: grp_firm_threshold() -- jointly shrinks intercept + slope DIF
#
# When pen.deriv = TRUE, the penalty tuning parameter tau is scaled by the
# inverse Hessian (tau / -hessian), incorporating curvature information into
# the regularization strength.
#
# The function can also operate in max_tau mode, where instead of updating
# parameters it computes the maximum tau value that would shrink all DIF
# parameters to zero (i.e., the tau at which the fully penalized/no-DIF
# solution is obtained).
#
# An under_identified flag is set to TRUE if too many DIF parameters become
# non-zero on a single covariate, indicating the model lacks sufficient
# anchor items for identification.
#
###############################################################################

#' Maximization step.
#'
#' @param p List of parameters.
#' @param item_data Matrix or data frame of item responses.
#' @param pred_data Matrix or data frame of DIF and/or impact predictors.
#' @param prox_data Vector of observed proxy scores.
#' @param mean_predictors Possibly different matrix of predictors for the mean
#' impact equation.
#' @param var_predictors Possibly different matrix of predictors for the
#' variance impact equation.
#' @param eout E-step output, including matrix for item and impact equations,
#' in addition to theta values (possibly adaptive).
#' @param item_type Optional character value or vector indicating the type of
#' item to be modeled.
#' @param pen_type Character value indicating the penalty function to use.
#' @param tau_current A single numeric value of tau that exists within
#' \code{tau_vec}.
#' @param pen Current penalty index.
#' @param pen.deriv Logical value indicating whether to use the second
#' derivative of the penalized parameter during regularization. The default is
#' TRUE.
#' @param alpha Numeric value indicating the alpha parameter in the elastic net
#' penalty function.
#' @param gamma Numeric value indicating the gamma parameter in the MCP
#' function.
#' @param anchor Optional numeric value or vector indicating which item
#' response(s) are anchors (e.g., \code{anchor = 1}).
#' @param final_control Control parameters.
#' @param samp_size Sample size in data set.
#' @param num_responses Number of responses for each item.
#' @param num_items Number of items in data set.
#' @param num_quad Number of quadrature points used for approximating the
#' latent variable.
#' @param num_predictors Number of predictors.
#' @param num_tau Logical indicating whether the minimum tau value needs to be
#' identified during the regDIF procedure.
#' @param max_tau Logical indicating whether to output the minimum tau value
#' needed to remove all DIF from the model.
#' @param optim_method Character value of the type of estimation method to use
#'
#' @return a \code{"list"} of estimates obtained from the maximization step using univariate
#' Newton-Raphson
#'
#' @keywords internal
#'
Mstep_simple <-
  function(p,
           item_data,
           pred_data,
           prox_data,
           mean_predictors,
           var_predictors,
           eout,
           item_type,
           pen_type,
           tau_current,
           pen,
           pen.deriv,
           alpha,
           gamma,
           anchor,
           final_control,
           samp_size,
           num_responses,
           num_items,
           num_quad,
           num_predictors,
           num_tau,
           max_tau,
           optim_method) {

    # Initialize the under-identification flag. This will be set to TRUE if
    # too many DIF parameters become non-zero for a given covariate, meaning
    # the model lacks enough anchor items (items with no DIF) to be identified.
    under_identified <- FALSE

    # Extract quadrature points (theta) and the E-step posterior table (etable)
    # from the E-step output. When prox_data is provided (proxy/observed score
    # approach), these are not needed since the latent variable is replaced by
    # the observed proxy.
    if(is.null(prox_data)) {
      # Update theta and etable.
      theta <- eout$theta
      etable <- eout$etable
    }


    # In max_tau mode, we accumulate the unpenalized NR updates (z values)
    # across all DIF parameters. The maximum absolute z value gives the
    # smallest tau that would shrink all DIF to zero.
    if(max_tau) id_max_z <- 0

    ###########################################################################
    ## Step 1: Latent mean (alpha) impact parameter updates
    ###########################################################################
    # Update impact mean parameters one covariate at a time. These control how
    # the latent trait mean varies as a function of observed covariates.
    # The parameter list stores impact mean at index (num_items + 1).
    for(cov in 1:ncol(mean_predictors)) {


      if(is.null(prox_data)) {

        # Compute first and second derivatives of the log-likelihood with
        # respect to the mean impact parameter for this covariate, using the
        # latent variable (quadrature-based) approach.
        anl_deriv <- d_alpha(c(p[[num_items+1]], p[[num_items+2]]),
                             etable,
                             theta,
                             mean_predictors,
                             var_predictors,
                             cov=cov,
                             samp_size,
                             num_items,
                             num_quad)

      } else {

        # Proxy-score approach: compute derivatives using observed proxy scores
        # instead of latent quadrature points.
        anl_deriv <- d_alpha_proxy(c(p[[num_items+1]], p[[num_items+2]]),
                                   prox_data,
                                   mean_predictors,
                                   var_predictors,
                                   cov=cov,
                                   samp_size,
                                   num_items)

      }

      # Newton-Raphson update: p_new = p_old - gradient/hessian.
      p[[num_items+1]][[cov]] <-
        p[[num_items+1]][[cov]] - anl_deriv[[1]]/anl_deriv[[2]]

    }

    ###########################################################################
    ## Step 2: Latent variance (phi) impact parameter updates
    ###########################################################################
    # Update impact variance parameters one covariate at a time. These control
    # how the latent trait variance varies as a function of observed covariates.
    # The parameter list stores impact variance at index (num_items + 2).
    for(cov in 1:ncol(var_predictors)) {

      if(is.null(prox_data)) {

        # Compute derivatives for the variance impact parameter using the
        # quadrature-based latent variable approach.
        anl_deriv <- d_phi(c(p[[num_items+1]], p[[num_items+2]]),
                           etable,
                           theta,
                           mean_predictors,
                           var_predictors,
                           cov=cov,
                           samp_size,
                           num_items,
                           num_quad)


      } else {

        # Proxy-score approach for variance impact derivatives.
        anl_deriv <- d_phi_proxy(c(p[[num_items+1]], p[[num_items+2]]),
                                 prox_data,
                                 mean_predictors,
                                 var_predictors,
                                 cov=cov,
                                 samp_size,
                                 num_items)

      }

      # Newton-Raphson update for the variance impact parameter.
      p[[num_items+2]][[cov]] <-
        p[[num_items+2]][[cov]] - anl_deriv[[1]]/anl_deriv[[2]]

    }

    ###########################################################################
    ## Step 3: Item parameter updates
    ###########################################################################
    # Loop over all items. For each item, update base (unpenalized) parameters
    # first, then DIF (penalized) parameters. The item type determines which
    # derivative functions are called: Gaussian (CFA), Bernoulli (2PL/Rasch),
    # or categorical (graded/ordinal).
    for (item in 1:num_items) {

      # For discrete (non-CFA) items with latent variable integration:
      # Build per-response-category E-tables by zeroing out rows where the
      # observed response does not match the category. These are used by the
      # Bernoulli and categorical derivative functions.
      if(item_type[item] != "cfa" & is.null(prox_data)) {
        etable_item <- lapply(1:num_responses[item], function(x) etable)
        for(resp in 1:num_responses[item]) {
          etable_item[[resp]][which(
            !(item_data[,item] == resp)), ] <- 0
        }
      }

      #########################################################################
      ## CFA (Gaussian/continuous) items
      #########################################################################
      if(item_type[item] == "cfa") {

          ## Base intercept (c0) update -- unpenalized.
          if(is.null(prox_data)) {
            anl_deriv <- d_mu_gaussian("c0",
                                       p[[item]],
                                       etable,
                                       theta,
                                       item_data[,item],
                                       pred_data,
                                       cov=NULL,
                                       samp_size,
                                       num_items,
                                       num_quad)
          } else {
            anl_deriv <- d_mu_gaussian_proxy("c0",
                                             p[[item]],
                                             prox_data,
                                             item_data[,item],
                                             pred_data,
                                             cov=NULL,
                                             samp_size)
          }

          # Newton-Raphson update for base intercept.
          p[[item]][[1]] <- p[[item]][[1]] - anl_deriv[[1]]/anl_deriv[[2]]


        ## Base slope (a0) update -- unpenalized, skipped for Rasch models.
        if(item_type[item] != "rasch") {

          # Identify the base slope parameter index within this item's
          # parameter vector.
          a0_parms <- grep(paste0("a0_item",item,"_"),names(p[[item]]),fixed=T)


            if(is.null(prox_data)) {
              anl_deriv <- d_mu_gaussian("a0",
                                         p[[item]],
                                         etable,
                                         theta,
                                         item_data[,item],
                                         pred_data,
                                         cov=NULL,
                                         samp_size,
                                         num_items,
                                         num_quad)
            } else {
              anl_deriv <- d_mu_gaussian_proxy("a0",
                                               p[[item]],
                                               prox_data,
                                               item_data[,item],
                                               pred_data,
                                               cov=NULL,
                                               samp_size)
            }

            # Newton-Raphson update for base slope.
            p[[item]][a0_parms] <- p[[item]][a0_parms] - anl_deriv[[1]]/anl_deriv[[2]]

        } # End Rasch conditional.


        ## Base residual variance (s0) update -- CFA-specific, unpenalized.
        # The residual variance parameter captures item-level error variance
        # in the Gaussian measurement model.
        s0_parms <- grep(paste0("s0_item",item,"_"),names(p[[item]]),fixed=T)

          if(is.null(prox_data)) {
            anl_deriv <- d_sigma_gaussian("s0",
                                          p[[item]],
                                          etable,
                                          theta,
                                          item_data[,item],
                                          pred_data,
                                          cov=NULL,
                                          samp_size,
                                          num_items,
                                          num_quad)
          } else {
            anl_deriv <- d_sigma_gaussian_proxy("s0",
                                                p[[item]],
                                                prox_data,
                                                item_data[,item],
                                                pred_data,
                                                cov=NULL,
                                                samp_size)
          }

          # Newton-Raphson update for base residual variance.
          p[[item]][s0_parms][[1]] <- p[[item]][s0_parms][[1]] - anl_deriv[[1]]/anl_deriv[[2]]

          # Guard against negative residual variance: reset to 1 if the
          # update produces a negative estimate.
          if(p[[item]][s0_parms][[1]] < 0) p[[item]][s0_parms][[1]] <- 1


        ## DIF parameter updates for CFA items.
        # Anchor items are excluded from DIF estimation since they serve as
        # the identification constraint (assumed DIF-free).
        if(!any(item == anchor)) {


          ## Residual DIF (s1) updates -- CFA-specific.
          # These capture covariate-dependent changes in the residual variance.
          for(cov in 1:num_predictors) {

            # Identify the residual DIF parameter index for this covariate.
            s1_parms <-
              grep(paste0("s1_item",item,"_cov",cov),names(p[[item]]),fixed=T)


              if(is.null(prox_data)) {
                anl_deriv <- d_sigma_gaussian("s1",
                                              p[[item]],
                                              etable,
                                              theta,
                                              item_data[,item],
                                              pred_data,
                                              cov=cov,
                                              samp_size,
                                              num_items,
                                              num_quad)
              } else{
                anl_deriv <- d_sigma_gaussian_proxy("s1",
                                                    p[[item]],
                                                    prox_data,
                                                    item_data[,item],
                                                    pred_data,
                                                    cov=cov,
                                                    samp_size)
              }

              # Newton-Raphson update for residual DIF.
              p[[item]][s1_parms][[1]] <- p[[item]][s1_parms][[1]] -
                anl_deriv[[1]]/anl_deriv[[2]]

              # In UNR mode, skip remaining covariates after the first update
              # (parameters are cycled one at a time in outer EM iterations).
              if(optim_method == "UNR") {
                p[[item]][s1_parms][[1]] <- p[[item]][s1_parms][[1]]
                break
              }


          } # End looping across covariates.

          # Flatten the full parameter list to a named vector for checking
          # how many DIF parameters are currently non-zero.
          p2 <- unlist(p)

          ## Intercept DIF (c1) updates -- penalized.
          for(cov in 1:num_predictors){


            # Under-identification check for intercept DIF: if no anchor
            # items were specified and all but one item already have non-zero
            # intercept DIF on this covariate, the model is not identified.
            # This check is only applied when alpha == 1 (pure lasso, not
            # elastic net), the penalty grid has at least 10 values, and
            # we are past the first penalty or no start values were given.
            if(is.null(anchor) &
               sum(p2[grep(paste0("c1(.*?)cov",cov),names(p2))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }

            # Identify the intercept DIF parameter index for this covariate.
            c1_parms <-
              grep(paste0("c1_item",item,"_cov",cov),names(p[[item]]),fixed=T)

              # Compute first and second derivatives for intercept DIF.
              if(is.null(prox_data)) {
                anl_deriv <- d_mu_gaussian("c1",
                                           p[[item]],
                                           etable,
                                           theta,
                                           item_data[,item],
                                           pred_data,
                                           cov,
                                           samp_size,
                                           num_items,
                                           num_quad)
              } else{
                anl_deriv <- d_mu_gaussian_proxy("c1",
                                                 p[[item]],
                                                 prox_data,
                                                 item_data[,item],
                                                 pred_data,
                                                 cov,
                                                 samp_size)
              }

              # Compute the unpenalized NR update (z_int). This is the value
              # that would result from a standard NR step without any penalty.
              z_int <- p[[item]][c1_parms] - anl_deriv[[1]]/anl_deriv[[2]]

              # In max_tau mode, accumulate z values to find the maximum.
              # When pen.deriv is TRUE, scale by -hessian to account for
              # the derivative-based penalty scaling.
              if(max_tau & pen.deriv) {
                id_max_z <- c(id_max_z,z_int*(-anl_deriv[[2]]))
              } else if(max_tau & !pen.deriv) {
                id_max_z <- c(id_max_z,z_int)
              }

              # Apply the penalty thresholding operator to produce the
              # regularized update. Group penalties (grp.lasso, grp.mcp) are
              # handled below after the slope DIF is also computed.
              if(!(pen_type == "grp.lasso" || pen_type == "grp.mcp")) {

                if(pen.deriv) {
                  # When pen.deriv is TRUE, tau is scaled by the inverse
                  # Hessian: tau / (-hessian). This adapts the penalty
                  # strength to the local curvature of the log-likelihood.
                  p[[item]][c1_parms][[1]] <-
                    ifelse(pen_type == "lasso",
                           soft_threshold(z_int,alpha,tau_current/-anl_deriv[[2]]),
                           firm_threshold(z_int,alpha,tau_current/-anl_deriv[[2]],gamma))
                } else {
                  # When pen.deriv is FALSE, tau is used directly without
                  # Hessian scaling.
                  p[[item]][c1_parms][[1]] <-
                    ifelse(pen_type == "lasso",
                           soft_threshold(z_int,alpha,tau_current),
                           firm_threshold(z_int,alpha,tau_current,gamma))
                }

              }



            # Under-identification check for slope DIF (same logic as for
            # intercept DIF above, but checking a1 parameters).
            if(is.null(anchor) &
               sum(p2[grep(paste0("a1(.*?)cov",cov),names(p2))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }



            ## Slope DIF (a1) update -- penalized, skipped for Rasch models.
            if(item_type[item] != "rasch"){

              # Identify the slope DIF parameter index for this covariate.
              a1_parms <- grep(paste0("a1_item",item,"_cov",cov),names(p[[item]]),fixed=T)

                # Compute first and second derivatives for slope DIF.
                if(is.null(prox_data)) {
                  anl_deriv <- d_mu_gaussian("a1",
                                             p[[item]],
                                             etable,
                                             theta,
                                             item_data[,item],
                                             pred_data,
                                             cov,
                                             samp_size,
                                             num_items,
                                             num_quad)
                } else {
                  anl_deriv <- d_mu_gaussian_proxy("a1",
                                                   p[[item]],
                                                   prox_data,
                                                   item_data[,item],
                                                   pred_data,
                                                   cov,
                                                   samp_size)
                }


                # Compute the unpenalized NR update for slope DIF.
                z_slp <- p[[item]][a1_parms] - anl_deriv[[1]]/anl_deriv[[2]]

                # Accumulate z values for max_tau computation.
                if(max_tau & pen.deriv) {
                  id_max_z <- c(id_max_z,z_slp*(-anl_deriv[[2]]))
                } else if(max_tau & !pen.deriv) {
                  id_max_z <- c(id_max_z,z_slp)
                }


                # Apply element-wise penalty for non-group penalties.
                if(!(pen_type == "grp.lasso" || pen_type == "grp.mcp")) {

                  if(pen.deriv) {
                    p[[item]][a1_parms][[1]] <-
                      ifelse(pen_type == "lasso",
                             soft_threshold(z_slp,alpha,tau_current/-anl_deriv[[2]]),
                             firm_threshold(z_slp,alpha,tau_current/-anl_deriv[[2]],gamma))
                  } else {
                    p[[item]][a1_parms][[1]] <-
                      ifelse(pen_type == "lasso",
                             soft_threshold(z_slp,alpha,tau_current),
                             firm_threshold(z_slp,alpha,tau_current,gamma))
                  }

                } else if(pen_type == "grp.lasso" || pen_type == "grp.mcp"){

                  # Group penalty: jointly shrink the intercept DIF (z_int)
                  # and slope DIF (z_slp) for this covariate. The group norm
                  # ensures both are shrunk to zero together or both remain
                  # non-zero.
                  grp.update <-
                    if(pen_type == "grp.lasso") {

                      grp_soft_threshold(c(z_int,z_slp),
                                         tau_current)

                    } else if(pen_type == "grp.mcp") {

                      grp_firm_threshold(c(z_int,z_slp),
                                         tau_current,
                                         gamma)

                    }

                    # Apply the group-penalized updates to both intercept
                    # and slope DIF simultaneously.
                    p[[item]][c1_parms][[1]] <- grp.update[[1]]
                    p[[item]][a1_parms][[1]] <- grp.update[[2]]


                }


            } # End Rasch conditional.

          } # End looping across covariates.

        } # End anchor item conditional.



      #########################################################################
      ## Binary (2PL / Rasch) items
      #########################################################################
      } else if(item_type[item] %in% c("2pl","rasch")) {                                     # Might need to add rasch as an option here


          ## Base intercept (c0) update -- unpenalized.
          if(is.null(prox_data)) {
            anl_deriv <- d_bernoulli("c0",
                                     p[[item]],
                                     etable_item,
                                     theta,
                                     pred_data,
                                     cov=0,
                                     samp_size,
                                     num_items,
                                     num_quad)
          } else {
            anl_deriv <- d_bernoulli_proxy("c0",
                                           p[[item]],
                                           prox_data,
                                           pred_data,
                                           item_data[,item],
                                           cov=0,
                                           samp_size,
                                           num_items)
          }

          # Newton-Raphson update for base intercept.
          p[[item]][[1]] <- p[[item]][[1]] - anl_deriv[[1]]/anl_deriv[[2]]


        ## Base slope (a0) update -- unpenalized, skipped for Rasch models.
        if(item_type[item] != "rasch") {

            if(is.null(prox_data)) {
              anl_deriv <- d_bernoulli("a0",
                                       p[[item]],
                                       etable_item,
                                       theta,
                                       pred_data,
                                       cov=0,
                                       samp_size,
                                       num_items,
                                       num_quad)
            } else {
              anl_deriv <- d_bernoulli_proxy("a0",
                                             p[[item]],
                                             prox_data,
                                             pred_data,
                                             item_data[,item],
                                             cov=0,
                                             samp_size,
                                             num_items)
            }

            # Newton-Raphson update for base slope.
            p[[item]][[2]] <- p[[item]][[2]] - anl_deriv[[1]]/anl_deriv[[2]]

        } # End Rasch conditional.


        ## DIF parameter updates for binary items.
        # Anchor items are excluded from DIF estimation.
        if(!any(item == anchor)) {

          # Flatten parameters to check non-zero DIF counts.
          p2 <- unlist(p)

          ## Intercept DIF (c1) updates -- penalized.
          for(cov in 1:num_predictors) {

            # if(pen_type == "grp.lasso" || pen_type == "grp.mcp") {
            #
            #   # End routine if only one anchor item is left on each covariate
            #   # for each item parameter.
            #   if(is.null(anchor) &&
            #      sum(p2[c(grep(paste0("c1(.*?)cov",cov),names(p2)),
            #               grep(paste0("a1(.*?)cov",cov),names(p2)))] != 0) >
            #      (num_items*2 - 1) &&
            #      (length(final_control$start.values) == 0 || pen > 1) &&
            #      num_tau >= 10){
            #     under_identified <- TRUE
            #     break
            #   }
            #
            # }

            # Under-identification check for intercept DIF: stop if all but
            # one item already have non-zero intercept DIF on this covariate.
            if(is.null(anchor) &
               sum(p2[grep(paste("c1(.*?)cov",cov, sep = ""),names(p2))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }


              # Compute derivatives for intercept DIF.
              if(is.null(prox_data)) {
                anl_deriv <- d_bernoulli("c1",
                                         p[[item]],
                                         etable_item,
                                         theta,
                                         pred_data,
                                         cov,
                                         samp_size,
                                         num_items,
                                         num_quad)
              } else {
                anl_deriv <- d_bernoulli_proxy("c1",
                                               p[[item]],
                                               prox_data,
                                               pred_data,
                                               item_data[,item],
                                               cov,
                                               samp_size,
                                               num_items)
              }

              # Compute the unpenalized NR update for intercept DIF.
              # For binary items, the DIF intercept is stored at position
              # (2 + cov) in the item parameter vector.
              z_int <- p[[item]][[2+cov]] - anl_deriv[[1]]/anl_deriv[[2]]

              # Accumulate z values for max_tau computation.
              if(max_tau & pen.deriv) {
                id_max_z <- c(id_max_z,z_int*(-anl_deriv[[2]]))
              } else if(max_tau & !pen.deriv) {
                id_max_z <- c(id_max_z,z_int)
              }

              # Apply element-wise penalty thresholding for non-group penalties.
              if(!(pen_type == "grp.lasso" || pen_type == "grp.mcp")) {

                if(pen.deriv) {
                  p[[item]][[2+cov]] <-
                    ifelse(pen_type == "lasso",
                           soft_threshold(z_int,alpha,tau_current/-anl_deriv[[2]]),
                           firm_threshold(z_int,alpha,tau_current/-anl_deriv[[2]],gamma))
                } else {
                  p[[item]][[2+cov]] <-
                    ifelse(pen_type == "lasso",
                           soft_threshold(z_int,alpha,tau_current),
                           firm_threshold(z_int,alpha,tau_current,gamma))
                }

              }



            # Under-identification check for slope DIF.
            if(is.null(anchor) &
               sum(p2[grep(paste0("a1(.*?)cov",cov),names(p2))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }

            ## Slope DIF (a1) update -- penalized, skipped for Rasch models.
            if(item_type[item] != "rasch") {


                # Compute derivatives for slope DIF.
                if(is.null(prox_data)) {
                  anl_deriv <- d_bernoulli("a1",
                                           p[[item]],
                                           etable_item,
                                           theta,
                                           pred_data,
                                           cov,
                                           samp_size,
                                           num_items,
                                           num_quad)
                } else {
                  anl_deriv <- d_bernoulli_proxy("a1",
                                                 p[[item]],
                                                 prox_data,
                                                 pred_data,
                                                 item_data[,item],
                                                 cov,
                                                 samp_size,
                                                 num_items)
                }

                # Compute the unpenalized NR update for slope DIF.
                # For binary items, slope DIF is stored at position
                # (2 + num_predictors + cov) in the item parameter vector.
                z_slp <- p[[item]][[2+num_predictors+cov]] - anl_deriv[[1]]/anl_deriv[[2]]

                # Accumulate z values for max_tau computation.
                if(max_tau & pen.deriv) {
                  id_max_z <- c(id_max_z,z_slp*(-anl_deriv[[2]]))
                } else if(max_tau & !pen.deriv) {
                  id_max_z <- c(id_max_z,z_slp)
                }

                # Apply element-wise penalty for non-group penalties.
                if(!(pen_type == "grp.lasso" || pen_type == "grp.mcp")) {

                  if(pen.deriv) {
                    p[[item]][[2+num_predictors+cov]] <-
                      ifelse(pen_type == "lasso",
                             soft_threshold(z_slp,alpha,tau_current/-anl_deriv[[2]]),
                             firm_threshold(z_slp,alpha,tau_current/-anl_deriv[[2]],gamma))
                  } else {
                    p[[item]][[2+num_predictors+cov]] <-
                      ifelse(pen_type == "lasso",
                             soft_threshold(z_slp,alpha,tau_current),
                             firm_threshold(z_slp,alpha,tau_current,gamma))
                  }

                } else if(pen_type == "grp.lasso" || pen_type == "grp.mcp") {

                  # Group penalty: jointly shrink intercept DIF (z_int) and
                  # slope DIF (z_slp) for this covariate together.
                  grp.update <-
                    if(pen_type == "grp.lasso") {

                      grp_soft_threshold(c(z_int,z_slp),
                                         tau_current)

                    } else if(pen_type == "grp.mcp") {

                      grp_firm_threshold(c(z_int,z_slp),
                                         tau_current,
                                         gamma)

                    }

                  # Apply the group-penalized updates to both intercept
                  # and slope DIF simultaneously.
                  p[[item]][[2+cov]] <- grp.update[[1]]
                  p[[item]][[2+num_predictors+cov]] <- grp.update[[2]]

                }


            } # End Rasch conditional.

          } # End looping across covariates.

        } # End anchor item conditional.

      #########################################################################
      ## Graded/ordinal (categorical) items
      #########################################################################
      } else {


          ## Base intercept (c0) update for first threshold -- unpenalized.
          # For graded items, the first element is the intercept for the first
          # threshold boundary. Additional thresholds are updated separately.
          if(is.null(prox_data)) {
            anl_deriv <- d_categorical("c0",
                                       p[[item]],
                                       etable_item,
                                       theta,
                                       pred_data,
                                       thr=-1,
                                       cov=-1,
                                       samp_size,
                                       num_responses[[item]],
                                       num_items,
                                       num_quad)
          } else {
            anl_deriv <- d_categorical_proxy("c0",
                                             p[[item]],
                                             prox_data,
                                             pred_data,
                                             item_data[,item],
                                             thr=-1,
                                             cov=-1,
                                             samp_size,
                                             num_responses[[item]],
                                             num_items)
          }
          # Newton-Raphson update for the first threshold intercept.
          p[[item]][[1]] <- p[[item]][[1]] - anl_deriv[[1]]/anl_deriv[[2]]

          # if(method == "UNR") {
          #   p[[item]][[1]] <- p[[item]][[1]]
          #   break
          # }


        ## Additional threshold (c0) updates for thresholds 2 through
        ## (num_responses - 1). Each threshold defines a boundary between
        ## adjacent response categories in the graded response model.

        for(thr in 2:(num_responses[item]-1)) {


            if(is.null(prox_data)) {
              anl_deriv <- d_categorical("c0",
                                         p[[item]],
                                         etable_item,
                                         theta,
                                         pred_data,
                                         thr=thr,
                                         cov=-1,
                                         samp_size,
                                         num_responses[[item]],
                                         num_items,
                                         num_quad)
            } else {
              anl_deriv <- d_categorical_proxy("c0",
                                               p[[item]],
                                               prox_data,
                                               pred_data,
                                               item_data[,item],
                                               thr=thr,
                                               cov=-1,
                                               samp_size,
                                               num_responses[[item]],
                                               num_items)
            }
            # Newton-Raphson update for this threshold.
            p[[item]][[thr]] <- p[[item]][[thr]] - anl_deriv[[1]]/anl_deriv[[2]]

            # If the threshold estimate is NA (e.g., due to numerical issues),
            # replace it with the average of that threshold across all items
            # as a stabilizing fallback.
            if(is.na(p[[item]][[thr]])) {
              p[[item]][[thr]] <-
                mean(sapply(1:num_items, function(x) p[[x]][[thr]]), na.rm = T)

            }



        } # End looping across thresholds.

        ## Base slope (a0) update -- unpenalized, skipped for Rasch models.
        # For graded items, the slope is stored at position num_responses
        # (after all threshold parameters).
        if(item_type[item] != "rasch") {


            if(is.null(prox_data)) {
              anl_deriv <- d_categorical("a0",
                                         p[[item]],
                                         etable_item,
                                         theta,
                                         pred_data,
                                         thr=-1,
                                         cov=-1,
                                         samp_size,
                                         num_responses[[item]],
                                         num_items,
                                         num_quad)
            } else {
              anl_deriv <- d_categorical_proxy("a0",
                                               p[[item]],
                                               prox_data,
                                               pred_data,
                                               item_data[,item],
                                               thr=-1,
                                               cov=-1,
                                               samp_size,
                                               num_responses[[item]],
                                               num_items)
            }
            # Newton-Raphson update for base slope.
            p[[item]][[num_responses[[item]]]] <-
              p[[item]][[num_responses[[item]]]] - anl_deriv[[1]]/anl_deriv[[2]]


        } # End Rasch conditional.

        ## DIF parameter updates for graded items.
        # Anchor items are excluded from DIF estimation.
        if(!any(item == anchor)){

          # Flatten parameters to check non-zero DIF counts.
          p2 <- unlist(p)

          ## Intercept DIF (c1) updates -- penalized.
          # For graded items, intercept DIF is stored at position
          # (num_responses + cov) in the item parameter vector.
          for(cov in 1:num_predictors) {

            # Under-identification check for intercept DIF.
            if(is.null(anchor) &
               sum(p2[grep(paste0("c1(.*?)cov",cov),names(p2))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }

              # Compute derivatives for intercept DIF.
              if(is.null(prox_data)) {
                anl_deriv <- d_categorical("c1",
                                           p[[item]],
                                           etable_item,
                                           theta,
                                           pred_data,
                                           thr=-1,
                                           cov,
                                           samp_size,
                                           num_responses[[item]],
                                           num_items,
                                           num_quad)
              } else {
                anl_deriv <- d_categorical_proxy("c1",
                                                 p[[item]],
                                                 prox_data,
                                                 pred_data,
                                                 item_data[,item],
                                                 thr=-1,
                                                 cov,
                                                 samp_size,
                                                 num_responses[[item]],
                                                 num_items)
              }

              # Compute the unpenalized NR update for intercept DIF.
              z <- p[[item]][[num_responses[[item]]+cov]] - anl_deriv[[1]]/anl_deriv[[2]]

              # Accumulate z values for max_tau computation.
              if(max_tau & pen.deriv) {
                id_max_z <- c(id_max_z,z*(-anl_deriv[[2]]))
              } else if(max_tau & !pen.deriv) {
                id_max_z <- c(id_max_z,z)
              }

              # Apply penalty thresholding to intercept DIF.
              if(pen.deriv) {
                p[[item]][[num_responses[[item]]+cov]] <-
                  ifelse(pen_type == "lasso",
                         soft_threshold(z,alpha,tau_current/-anl_deriv[[2]]),
                         firm_threshold(z,alpha,tau_current/-anl_deriv[[2]],gamma))
              } else {
                p[[item]][[num_responses[[item]]+cov]] <-
                  ifelse(pen_type == "lasso",
                         soft_threshold(z,alpha,tau_current),
                         firm_threshold(z,alpha,tau_current,gamma))
              }


          } # End looping across covariates.

          ## Slope DIF (a1) updates -- penalized, skipped for Rasch models.
          # For graded items, slope DIF is stored at the end of the item
          # parameter vector: position (length - num_predictors + cov).
          for(cov in 1:num_predictors) {

            # Under-identification check for slope DIF.
            if(is.null(anchor) &
               sum(p2[grep(paste0("a1(.*?)cov",cov),names(p2))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }

            if(item_type[item] != "rasch") {


                # Compute derivatives for slope DIF.
                if(is.null(prox_data)) {
                  anl_deriv <- d_categorical("a1",
                                             p[[item]],
                                             etable_item,
                                             theta,
                                             pred_data,
                                             thr=-1,
                                             cov,
                                             samp_size,
                                             num_responses[[item]],
                                             num_items,
                                             num_quad)
                } else {
                  anl_deriv <- d_categorical_proxy("a1",
                                                   p[[item]],
                                                   prox_data,
                                                   pred_data,
                                                   item_data[,item],
                                                   thr=-1,
                                                   cov,
                                                   samp_size,
                                                   num_responses[[item]],
                                                   num_items)
                }

                # Compute the unpenalized NR update for slope DIF.
                z <- p[[item]][[length(p[[item]])-ncol(pred_data)+cov]] -
                  anl_deriv[[1]]/anl_deriv[[2]]

                # Accumulate z values for max_tau computation.
                if(max_tau & pen.deriv) {
                  id_max_z <- c(id_max_z,z*(-anl_deriv[[2]]))
                } else if(max_tau & !pen.deriv) {
                  id_max_z <- c(id_max_z,z)
                }


                # Apply penalty thresholding to slope DIF.
                if(pen.deriv) {
                  p[[item]][[length(p[[item]])-ncol(pred_data)+cov]] <-
                    ifelse(pen_type == "lasso",
                           soft_threshold(z,alpha,tau_current/-anl_deriv[[2]]),
                           firm_threshold(z,alpha,tau_current/-anl_deriv[[2]],gamma))
                } else {
                  p[[item]][[length(p[[item]])-ncol(pred_data)+cov]] <-
                    ifelse(pen_type == "lasso",
                           soft_threshold(z,alpha,tau_current),
                           firm_threshold(z,alpha,tau_current,gamma))
                }



            } # End Rasch conditional.

          } # End looping across covariates.

        } # End anchor item condtional.


      } # End item type conditional.

    } # End looping through items.



    ###########################################################################
    ## Return results
    ###########################################################################

    if(max_tau) {

      # In max_tau mode, return the maximum absolute unpenalized z value
      # across all DIF parameters. This is the smallest tau value that
      # would shrink all DIF parameters to zero.
      id_max_z <- max(abs(unlist(id_max_z)))

      return(id_max_z)

    } else {

        # In normal mode, return the updated parameter list and the
        # under-identification flag.
        return(list(p=p,
                    under_identified=under_identified))
    }

  }
