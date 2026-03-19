###############################################################################
###
### M-Step: Single-Pass Coordinate Descent
###
### This function implements the maximization step of the EM algorithm using
### a single pass of coordinate descent. Each parameter receives exactly one
### Newton-Raphson update (i.e., parameter -= gradient/hessian_diag) before
### moving on to the next parameter. This contrasts with m_step_cd2.R, which
### iterates coordinate descent to full convergence within the M-step.
###
### Update order:
###   1. Latent mean impact parameters (alpha)
###   2. Latent variance impact parameters (phi)
###   3. Item parameters, looping over all items:
###      a. Baseline intercept/threshold(s) and slope (unpenalized)
###      b. DIF intercept parameters (penalized via soft/firm thresholding)
###      c. DIF slope parameters (penalized via soft/firm thresholding)
###
### Supported item types:
###   - "cfa"   : Continuous/Gaussian (confirmatory factor analysis indicators)
###   - "2pl"   : Binary items (two-parameter logistic / Bernoulli)
###   - "graded": Ordinal items (graded response model / categorical)
###
### Penalty functions applied to DIF parameters:
###   - Lasso  : soft thresholding
###   - MCP    : firm thresholding (requires gamma tuning parameter)
###   - Elastic net blending controlled by alpha (alpha=1 is pure lasso)
###
### Under-identification protection:
###   When no anchor items are user-specified, the routine checks whether all
###   but one item already have non-zero DIF on a given covariate. If so, the
###   model would be under-identified and the routine flags this condition.
###
### Parallel execution:
###   When final_control$parallel is enabled, item-level updates are
###   distributed across workers using foreach/%dopar%. The parallel branch
###   currently handles binary (2pl) and graded item types only.
###
###############################################################################

#' Maximization step using coordinate descent optimization.
#'
#' @param p List of parameters.
#' @param item_data Matrix or data frame of item responses.
#' @param pred_data Matrix or data frame of DIF and/or impact predictors.
#' @param mean_predictors Possibly different matrix of predictors for the mean
#' impact equation.
#' @param var_predictors Possibly different matrix of predictors for the
#' variance impact equation.
#' @param eout E step output, including matrix for item and impact equations,
#' in addition to theta values (possibly adaptive).
#' @param item_type Optional character value or vector indicating the type of
#' item to be modeled.
#' @param pen_type Character value indicating the penalty function to use.
#' @param tau_current A single numeric value of tau that exists within
#' \code{tau_vec}.
#' @param pen Current penalty index.
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
#' @param max_tau Logical indicating whether to output the maximum tau value
#' needed to remove all DIF from the model.
#'
#' @return a \code{"list"} of estimates obtained from the maximization step using univariate
#' Newton-Raphson (i.e., one step of coordinate descent)
#'
#' @importFrom foreach %dopar%
#' @keywords internal
#'
Mstep_cd <-
  function(p,
           item_data,
           pred_data,
           mean_predictors,
           var_predictors,
           eout,
           item_type,
           pen_type,
           tau_current,
           pen,
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
           max_tau) {

  ###########################################################################
  ## Initialization
  ###########################################################################

  # Set under-identified model to FALSE until proven TRUE. Will be flipped

  # to TRUE if all items (minus one) show non-zero DIF on any covariate,
  # which would make the model unidentifiable without an anchor.
  under_identified <- FALSE

  # Extract quadrature points (theta) and the E-table of posterior
  # probabilities from the E-step output.
  theta <- eout$theta
  etable <- eout$etable

  # When max_tau mode is active, collect all unpenalized Newton-Raphson
  # proposals (z values) for DIF parameters. The maximum absolute z across
  # all DIF parameters gives the smallest tau that would shrink everything
  # to zero (i.e., the maximum tau needed to produce a no-DIF model).
  if(max_tau) id_max_z <- 0

  ###########################################################################
  ## Impact parameter updates (latent mean and variance)
  ##
  ## Impact parameters model how the latent trait distribution shifts
  ## (mean) and scales (variance) as a function of observed covariates.
  ## These are always updated first, before item parameters, because
  ## item parameter updates condition on the current impact estimates.
  ## Each covariate coefficient gets a single Newton-Raphson step:
  ##   param_new = param_old - first_deriv / second_deriv
  ###########################################################################

  # Latent mean impact updates: one NR step per covariate coefficient.
  for(cov in 1:ncol(mean_predictors)) {
    # d_alpha returns list(gradient, hessian_diagonal) for the mean impact
    # coefficient on the given covariate.
    anl_deriv <- d_alpha(c(p[[num_items+1]],p[[num_items+2]]),
                             etable,
                             theta,
                             mean_predictors,
                             var_predictors,
                             cov=cov,
                             samp_size,
                             num_items,
                             num_quad)
    # Single Newton-Raphson step: param -= gradient / hessian_diag.
    p[[num_items+1]][[cov]] <-
      p[[num_items+1]][[cov]] - anl_deriv[[1]]/anl_deriv[[2]]
  }

  # Latent variance impact updates: one NR step per covariate coefficient.
  for(cov in 1:ncol(var_predictors)) {
    # d_phi returns list(gradient, hessian_diagonal) for the variance impact
    # coefficient on the given covariate.
    anl_deriv <- d_phi(c(p[[num_items+1]],p[[num_items+2]]),
                           etable,
                           theta,
                           mean_predictors,
                           var_predictors,
                           cov=cov,
                           samp_size,
                           num_items,
                           num_quad)
    # Single Newton-Raphson step for variance impact coefficient.
    p[[num_items+2]][[cov]] <-
      p[[num_items+2]][[cov]] - anl_deriv[[1]]/anl_deriv[[2]]
  }


  ###########################################################################
  ## Item parameter updates
  ##
  ## Two execution paths: parallel (foreach/%dopar%) or sequential.
  ## Within each item, the update order is:
  ##   1. Baseline (non-DIF) intercept/threshold and slope parameters
  ##      - These receive unpenalized Newton-Raphson steps
  ##   2. DIF intercept parameters (one per covariate)
  ##      - NR proposal z is computed, then penalized via thresholding
  ##   3. DIF slope parameters (one per covariate, skipped for Rasch)
  ##      - Same NR-then-threshold approach
  ##
  ## For DIF parameters, the update is a proximal gradient step:
  ##   z = param_old - gradient/hessian  (unpenalized NR proposal)
  ##   param_new = threshold(z, tau)     (apply penalty)
  ###########################################################################

  # Item response updates.
  # Branch 1: Parallel execution across items using foreach/%dopar%.
  if(final_control$parallel[[1]]) {
    # Export all required objects to the parallel cluster workers so they
    # are available in the foreach loop environment.
    parallel::clusterExport(final_control$parallel[[2]],
                            c("num_responses", "pred_data", "item_data",
                               "samp_size", "num_predictors", "num_quad", "num_items",
                               "prox_data", "item_type", "anchor", "pen_type",
                               "num_tau", "pen", "tau_vec", "alpha", "final_control",
                               "max_tau", "d_bernoulli_itemblock", "d_bernoulli_itemblock_proxy",
                               "grp_soft_threshold", "grp_firm_threshold", "soft_threshold",
                               "firm_threshold", "p", "eout", "inv_hess_diag", "tau_current",
                               "under_identified", "id_max_z"),
                            envir=environment())

    # Distribute item-level updates across parallel workers. Each worker
    # returns a list: [[1]] updated item parameters, [[2]] max_tau z values.
    p_items <- foreach::foreach(item=1:num_items) %dopar% {

      # Obtain E-tables for each response category.
      # For non-CFA items, create a copy of the full E-table for each
      # response category, then zero out rows where the person did not
      # endorse that category. This produces category-specific posterior
      # weights used in the derivative computations.
      if(item_type[item] != "cfa") {
        etable_item <- lapply(1:num_responses[item], function(x) etable)
        for(resp in 1:num_responses[item]) {
          etable_item[[resp]][which(
            !(item_data[,item] == resp)), ] <- 0
        }
      }

      #################################################################
      ## Binary (2PL) items -- parallel branch
      #################################################################
      if(item_type[item] == "2pl") {

        # Baseline intercept (c0): unpenalized NR update.
        anl_deriv <- d_bernoulli("c0",
                                 p[[item]],
                                 etable_item,
                                 theta,
                                 pred_data,
                                 cov=0,
                                 samp_size,
                                 num_items,
                                 num_quad)
        p[[item]][[1]] <- p[[item]][[1]] - anl_deriv[[1]]/anl_deriv[[2]]

        # Baseline slope (a0): unpenalized NR update (skipped for Rasch).
        if(item_type[item] != "rasch") {
          anl_deriv <- d_bernoulli("a0",
                                   p[[item]],
                                   etable_item,
                                   theta,
                                   pred_data,
                                   cov=0,
                                   samp_size,
                                   num_items,
                                   num_quad)
          p[[item]][[2]] <- p[[item]][[2]] - anl_deriv[[1]]/anl_deriv[[2]]
        }

        # Skip DIF updates for anchor items (their DIF is fixed at zero).
        if(!any(item == anchor)) {

          # Flatten all parameters to a named vector for the
          # under-identification check below.
          p2 <- unlist(p)

          # Intercept DIF (c1) updates: one per covariate, penalized.
          for(cov in 1:num_predictors) {

            # Under-identification guard: if all items except one already
            # have non-zero intercept DIF on this covariate, adding another
            # would leave no anchor, making the model unidentified.
            # Conditions: no user-specified anchors, near-saturated DIF,
            # pure lasso (alpha=1), not the first penalty, and enough tau
            # values in the grid (>=10).
            if(is.null(anchor) &
               sum(p2[grep(paste0("c1(.*?)cov",cov),names(p2))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }

            # Compute gradient and diagonal Hessian for the DIF intercept.
            anl_deriv <- d_bernoulli("c1",
                                     p[[item]],
                                     etable_item,
                                     theta,
                                     pred_data,
                                     cov,
                                     samp_size,
                                     num_items,
                                     num_quad)
            # Unpenalized NR proposal (z), then apply penalty threshold.
            z <- p[[item]][[2+cov]] - anl_deriv[[1]]/anl_deriv[[2]]
            # Collect z for max_tau identification.
            if(max_tau) id_max_z <- c(id_max_z,z)
            # Apply soft (lasso) or firm (MCP) thresholding to obtain the
            # penalized parameter estimate.
            p[[item]][[2+cov]] <- ifelse(pen_type == "lasso",
                                         soft_threshold(z,alpha,tau_current),
                                         firm_threshold(z,alpha,tau_current,gamma))
          }

          # Slope DIF (a1) updates: one per covariate, penalized.
          for(cov in 1:num_predictors) {

            # Under-identification guard for slope DIF (same logic as
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

            # Skip slope DIF for Rasch-constrained items.
            if(item_type[item] != "rasch") {
              anl_deriv <- d_bernoulli("a1",
                                       p[[item]],
                                       etable_item,
                                       theta,
                                       pred_data,
                                       cov,
                                       samp_size,
                                       num_items,
                                       num_quad)
              # Slope DIF parameters are stored after intercept + slope +
              # intercept DIF parameters: index = 2 + num_predictors + cov.
              z <- p[[item]][[2+num_predictors+cov]] -
                anl_deriv[[1]]/anl_deriv[[2]]
              if(max_tau) id_max_z <- c(id_max_z,z)
              p[[item]][[2+num_predictors+cov]] <-
                ifelse(pen_type == "lasso",
                       soft_threshold(z,alpha,tau_current),
                       firm_threshold(z,alpha,tau_current,gamma))
            }

          }

        }

      #################################################################
      ## Graded/categorical items -- parallel branch
      #################################################################
      } else {

        # Baseline intercept (first threshold, c0): unpenalized NR update.
        # For graded items, thr=-1 signals the first threshold.
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
        p[[item]][[1]] <- p[[item]][[1]] - anl_deriv[[1]]/anl_deriv[[2]]

        # Higher threshold updates (thresholds 2 through K-1 for K
        # response categories). Each gets an unpenalized NR step.
        for(thr in 2:(num_responses[item]-1)) {
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
          p[[item]][[thr]] <- p[[item]][[thr]] - anl_deriv[[1]]/anl_deriv[[2]]
        }

        # Baseline slope (a0): unpenalized NR update (skipped for Rasch).
        # For graded items, the slope is stored at position num_responses.
        if(item_type[item] != "rasch") {
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
          p[[item]][[num_responses[[item]]]] <-
            p[[item]][[num_responses[[item]]]] - anl_deriv[[1]]/anl_deriv[[2]]
        }

        # Skip DIF updates for anchor items.
        if(!any(item == anchor)){

          p2 <- unlist(p)

          # Intercept DIF (c1) updates: one per covariate, penalized.
          for(cov in 1:num_predictors) {

            # Under-identification guard (same logic as binary items).
            if(is.null(anchor) &
               sum(p2[grep(paste0("c1(.*?)cov",cov),names(p2))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }

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
            # For graded items, intercept DIF is stored after the
            # num_responses positions (thresholds + slope).
            z <- p[[item]][[num_responses[[item]]+cov]] -
              anl_deriv[[1]]/anl_deriv[[2]]
            if(max_tau) id_max_z <- c(id_max_z,z)
            p[[item]][[num_responses[[item]]+cov]] <-
              ifelse(pen_type == "lasso",
                     soft_threshold(z,alpha,tau_current),
                     firm_threshold(z,alpha,tau_current,gamma))
          }

          # Slope DIF (a1) updates: one per covariate, penalized.
          for(cov in 1:num_predictors) {

            # Under-identification guard for slope DIF.
            if(is.null(anchor) &
               sum(p2[grep(paste0("a1(.*?)cov",cov),names(p2))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }

            # Skip slope DIF for Rasch-constrained items.
            if(item_type[item] != "rasch") {
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
              # Slope DIF stored at the tail of the parameter vector.
              z <- p[[item]][[length(p[[item]])-ncol(pred_data)+cov]] -
                anl_deriv[[1]]/anl_deriv[[2]]
              if(max_tau) id_max_z <- c(id_max_z,z)
              p[[item]][[length(p[[item]])-ncol(pred_data)+cov]] <-
                ifelse(pen_type == "lasso",
                       soft_threshold(z,alpha,tau_current),
                       firm_threshold(z,alpha,tau_current,gamma))
            }
          }
        }


      }
      # Return updated item parameters and collected z values for max_tau.
      return(list(unlist(p[item]),id_max_z))
    }


    # Collect parallel results: copy updated item parameters back into
    # the main parameter list p.
    for(item in 1:num_items) {
      p[[item]] <- p_items[[item]][[1]]
    }

  } else {

    ###################################################################
    ## Sequential (non-parallel) item updates
    ##
    ## Same logic as the parallel branch above but executed serially.
    ## Supports all three item types: CFA, binary (2pl), and graded.
    ## The CFA (Gaussian) path is only available here (not in the
    ## parallel branch) because it requires additional derivative
    ## functions for the mean and residual variance.
    ###################################################################

    for (item in 1:num_items) {

      # Obtain E-tables for each response category.
      # Creates category-specific posterior weight matrices by zeroing
      # out rows for persons who did not endorse each category.
      if(item_type[item] != "cfa") {
        etable_item <- lapply(1:num_responses[item], function(x) etable)
        for(resp in 1:num_responses[item]) {
          etable_item[[resp]][which(
            !(item_data[,item] == resp)), ] <- 0
        }
      }

      #################################################################
      ## CFA / Gaussian items -- sequential branch only
      ##
      ## For continuous indicators, the model has three parameter
      ## families: mean (intercept c0, slope a0), residual variance
      ## (s0), and their DIF counterparts (c1, a1, s1).
      ## Residual variance (s0) is constrained to be positive; if the
      ## NR step produces a negative value, it is reset to 1.
      #################################################################
      if(item_type[item] == "cfa") {

        # Baseline intercept (c0): unpenalized NR update for the
        # Gaussian mean model.
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
        p[[item]][[1]] <- p[[item]][[1]] - anl_deriv[[1]]/anl_deriv[[2]]

        # Baseline slope (a0): unpenalized NR update (skipped for Rasch).
        # Uses named parameter lookup to find the a0 index.
        if(item_type[item] != "rasch") {
          a0_parms <- grep(paste0("a0_item",item,"_"),names(p[[item]]),fixed=T)
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
          p[[item]][[2]] <- p[[item]][a0_parms] - anl_deriv[[1]]/anl_deriv[[2]]
        }


        # Baseline residual variance (s0): unpenalized NR update.
        # Uses the Gaussian sigma derivative function.
        s0_parms <- grep(paste0("s0_item",item,"_"),names(p[[item]]),fixed=T)
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
        p[[item]][s0_parms][[1]] <- p[[item]][s0_parms][[1]] -
          anl_deriv[[1]]/anl_deriv[[2]]
        # Positivity constraint: if NR step produces a negative residual
        # variance, reset to 1 (a safe default).
        if(p[[item]][s0_parms][[1]] < 0) p[[item]][s0_parms][[1]] <- 1

        # Skip DIF updates for anchor items.
        if(!any(item == anchor)) {

          # Residual variance DIF (s1) updates: one per covariate.
          # Note: residual DIF is updated BEFORE intercept/slope DIF
          # for CFA items, and these updates are NOT penalized.
          for(cov in 1:num_predictors) {
            s1_parms <-
              grep(paste0("s1_item",item,"_cov",cov),names(p[[item]]),fixed=T)
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
            p[[item]][s1_parms][[1]] <- p[[item]][s1_parms][[1]] -
              anl_deriv[[1]]/anl_deriv[[2]]
          }

          # Flatten all parameters for the under-identification check.
          p2 <- unlist(p)

          # Intercept DIF (c1) updates: one per covariate, penalized.
          for(cov in 1:num_predictors){

            # Under-identification guard (see binary section for details).
            if(is.null(anchor) &
               sum(p2[grep(paste0("c1(.*?)cov",cov),names(p2))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }

            # Locate the named c1 parameter for this item-covariate pair.
            c1_parms <-
              grep(paste0("c1_item",item,"_cov",cov),names(p[[item]]),fixed=T)
            # Compute gradient/Hessian using the Gaussian mean derivative.
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
            # NR proposal, then penalty thresholding.
            z <- p[[item]][c1_parms] - anl_deriv[[1]]/anl_deriv[[2]]
            if(max_tau) id_max_z <- c(id_max_z,z)
            p[[item]][c1_parms][[1]] <-
              ifelse(pen_type == "lasso",
                     soft_threshold(z,alpha,tau_current),
                     firm_threshold(z,alpha,tau_current,gamma))
          }

          # Slope DIF (a1) updates: one per covariate, penalized.
          for(cov in 1:num_predictors){

            # Under-identification guard for slope DIF.
            if(is.null(anchor) &
               sum(p2[grep(paste0("a1(.*?)cov",cov),names(p2))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }

            # Skip slope DIF for Rasch-constrained items.
            if(item_type[item] != "rasch"){
              a1_parms <-
                grep(paste0("a1_item",item,"_cov",cov),names(p[[item]]),fixed=T)
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
              z <- p[[item]][a1_parms] - anl_deriv[[1]]/anl_deriv[[2]]
              if(max_tau) id_max_z <- c(id_max_z,z)
              p[[item]][a1_parms][[1]] <- ifelse(pen_type == "lasso",
                                                 soft_threshold(z,alpha,tau_current),
                                                 firm_threshold(z,alpha,tau_current,gamma))
            }
          }
        }


        #################################################################
        ## Binary (2PL) items -- sequential branch
        #################################################################
      } else if(item_type[item] == "2pl") {

        # Baseline intercept (c0): unpenalized NR update.
        anl_deriv <- d_bernoulli("c0",
                                 p[[item]],
                                 etable_item,
                                 theta,
                                 pred_data,
                                 cov=0,
                                 samp_size,
                                 num_items,
                                 num_quad)
        p[[item]][[1]] <- p[[item]][[1]] - anl_deriv[[1]]/anl_deriv[[2]]

        # Baseline slope (a0): unpenalized NR update (skipped for Rasch).
        if(item_type[item] != "rasch") {
          anl_deriv <- d_bernoulli("a0",
                                   p[[item]],
                                   etable_item,
                                   theta,
                                   pred_data,
                                   cov=0,
                                   samp_size,
                                   num_items,
                                   num_quad)
          p[[item]][[2]] <- p[[item]][[2]] - anl_deriv[[1]]/anl_deriv[[2]]
        }

        # Skip DIF updates for anchor items.
        if(!any(item == anchor)) {

          # Flatten all parameters for the under-identification check.
          p2 <- unlist(p)

          # Intercept DIF (c1) updates: one per covariate, penalized.
          for(cov in 1:num_predictors) {

            # Under-identification guard (see parallel binary section for
            # detailed explanation of the conditions).
            if(is.null(anchor) &
               sum(p2[grep(paste0("c1(.*?)cov",cov),names(p2))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }

            # NR proposal then penalty thresholding for intercept DIF.
            anl_deriv <- d_bernoulli("c1",
                                     p[[item]],
                                     etable_item,
                                     theta,
                                     pred_data,
                                     cov,
                                     samp_size,
                                     num_items,
                                     num_quad)
            z <- p[[item]][[2+cov]] - anl_deriv[[1]]/anl_deriv[[2]]
            if(max_tau) id_max_z <- c(id_max_z,z)
            p[[item]][[2+cov]] <- ifelse(pen_type == "lasso",
                                         soft_threshold(z,alpha,tau_current),
                                         firm_threshold(z,alpha,tau_current,gamma))
          }

          # Slope DIF (a1) updates: one per covariate, penalized.
          for(cov in 1:num_predictors) {

            # Under-identification guard for slope DIF.
            if(is.null(anchor) &
               sum(p2[grep(paste0("a1(.*?)cov",cov),names(p2))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }

            # Skip slope DIF for Rasch-constrained items.
            if(item_type[item] != "rasch") {
              anl_deriv <- d_bernoulli("a1",
                                       p[[item]],
                                       etable_item,
                                       theta,
                                       pred_data,
                                       cov,
                                       samp_size,
                                       num_items,
                                       num_quad)
              # Slope DIF index: 2 (c0+a0) + num_predictors (c1) + cov.
              z <- p[[item]][[2+num_predictors+cov]] -
                anl_deriv[[1]]/anl_deriv[[2]]
              if(max_tau) id_max_z <- c(id_max_z,z)
              p[[item]][[2+num_predictors+cov]] <-
                ifelse(pen_type == "lasso",
                       soft_threshold(z,alpha,tau_current),
                       firm_threshold(z,alpha,tau_current,gamma))
            }

          }

        }

        #################################################################
        ## Graded/categorical items -- sequential branch
        #################################################################
      } else {

        # Baseline intercept (first threshold, c0): unpenalized NR update.
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
        p[[item]][[1]] <- p[[item]][[1]] - anl_deriv[[1]]/anl_deriv[[2]]

        # Higher threshold updates (thresholds 2 through K-1).
        for(thr in 2:(num_responses[item]-1)) {
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
          p[[item]][[thr]] <- p[[item]][[thr]] - anl_deriv[[1]]/anl_deriv[[2]]
        }

        # Baseline slope (a0): unpenalized NR update (skipped for Rasch).
        if(item_type[item] != "rasch") {
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
          p[[item]][[num_responses[[item]]]] <-
            p[[item]][[num_responses[[item]]]] - anl_deriv[[1]]/anl_deriv[[2]]
        }

        # Skip DIF updates for anchor items.
        if(!any(item == anchor)){

          p2 <- unlist(p)

          # Intercept DIF (c1) updates: one per covariate, penalized.
          for(cov in 1:num_predictors) {

            # Under-identification guard.
            if(is.null(anchor) &
               sum(p2[grep(paste0("c1(.*?)cov",cov),names(p2))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }

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
            # For graded items, intercept DIF stored after threshold+slope
            # positions: index = num_responses + cov.
            z <- p[[item]][[num_responses[[item]]+cov]] -
              anl_deriv[[1]]/anl_deriv[[2]]
            if(max_tau) id_max_z <- c(id_max_z,z)
            p[[item]][[num_responses[[item]]+cov]] <-
              ifelse(pen_type == "lasso",
                     soft_threshold(z,alpha,tau_current),
                     firm_threshold(z,alpha,tau_current,gamma))
          }

          # Slope DIF (a1) updates: one per covariate, penalized.
          for(cov in 1:num_predictors) {

            # Under-identification guard for slope DIF.
            if(is.null(anchor) &
               sum(p2[grep(paste0("a1(.*?)cov",cov),names(p2))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }

            # Skip slope DIF for Rasch-constrained items.
            if(item_type[item] != "rasch") {
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
              # Slope DIF stored at the tail of the parameter vector.
              z <- p[[item]][[length(p[[item]])-ncol(pred_data)+cov]] -
                anl_deriv[[1]]/anl_deriv[[2]]
              if(max_tau) id_max_z <- c(id_max_z,z)
              p[[item]][[length(p[[item]])-ncol(pred_data)+cov]] <-
                ifelse(pen_type == "lasso",
                       soft_threshold(z,alpha,tau_current),
                       firm_threshold(z,alpha,tau_current,gamma))
            }
          }
        }



      }

    }

  }


  ###########################################################################
  ## Return values
  ##
  ## Two return modes:
  ##   1. max_tau mode: returns the maximum absolute unpenalized NR proposal
  ##      (z) across all DIF parameters. This is the smallest penalty value
  ##      that would shrink all DIF to zero (useful for setting the tau grid).
  ##   2. Normal mode: returns the updated parameter list and a flag
  ##      indicating whether the model is under-identified.
  ###########################################################################
  if(max_tau) {
    # In parallel mode, z values were collected per-item; aggregate here.
    if(final_control$parallel[[1]]) {
      id_max_z <- max(abs(sapply(1:num_items, function(items) p_items[[items]][[2]])))
    } else{
      id_max_z <- max(abs(id_max_z))
    }
    return(id_max_z)
  } else {
    return(list(p=p,
                under_identified=under_identified))
  }

}
