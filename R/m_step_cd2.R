###############################################################################
### M-Step: Full Coordinate Descent (Iterates to Convergence)
###
### This function implements the M-step of the EM algorithm using full
### coordinate descent optimization. Unlike Mstep_cd (single-pass coordinate
### descent in m_step_cd.R), this variant wraps all parameter updates inside
### an outer while-loop that repeats the full cycle of updates until overall
### convergence is reached (eps_cd_all <= tol).
###
### Convergence structure (nested loops):
###   OUTER LOOP -- repeats entire parameter sweep until all parameters
###                 converge jointly (eps_cd_all).
###     IMPACT PARAMETERS -- each mean/variance impact covariate has its own
###                          inner Newton-Raphson while-loop converging
###                          individually (eps_cd).
###     ITEM PARAMETERS   -- for 2PL items, each baseline and DIF parameter
###                          has its own inner Newton-Raphson while-loop.
###                          For CFA and categorical items, single-step
###                          Newton-Raphson updates are used (no inner loop).
###
### Parameter storage:
###   p_cd  -- working copy of parameter list, updated in-place during
###            coordinate descent iterations.
###   p     -- used directly (not via p_cd) for CFA and categorical item
###            updates, which do not use inner convergence loops.
###
### DIF penalty:
###   After each Newton-Raphson update of a DIF parameter, the intermediate
###   value z is passed through a soft_threshold (lasso) or firm_threshold
###   (MCP) function to apply the regularization penalty.
###
### Identification:
###   The routine monitors whether too many DIF parameters are nonzero for
###   a given covariate. If all but one item have nonzero DIF on a covariate,
###   the model is flagged as under-identified and the routine exits early.
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
#' @return a \code{"list"} of estimates obtained from the maximization step using coordinate
#' descent
#'
#' @keywords internal
#'
Mstep_cd2 <-
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

    # Set under-identified model to FALSE until proven TRUE. Will be flipped
    # to TRUE if too many DIF parameters are nonzero on any covariate.
    under_identified <- FALSE

    # Extract posterior quadrature points and expected counts from E-step output.
    theta <- eout$theta
    etable <- eout$etable

    # Initialize the working parameter list for coordinate descent. All updates
    # within the outer loop modify p_cd in-place; the original p is preserved
    # for CFA/categorical branches that do not use inner convergence loops.
    p_cd <- p

    # When max_tau is TRUE, this M-step is only used to identify the maximum
    # tau value needed to shrink all DIF to zero. Collect pre-threshold z
    # values to determine that maximum.
    if(max_tau) id_max_z <- 0


    ###########################################################################
    ### Outer convergence loop: iterate over ALL parameters until joint
    ### convergence. lastp_cd_all stores the parameter snapshot from the
    ### previous full sweep; eps_cd_all measures the Euclidean distance
    ### between successive full sweeps.
    ###########################################################################
    lastp_cd_all <- p_cd
    eps_cd_all <- Inf
    iter_cd_all <- 1

    # Outer loop: repeat the full cycle of impact + item updates until the
    # overall change in all parameters (eps_cd_all) drops below tolerance.
    while(eps_cd_all > final_control$tol){



    #########################################################################
    ### Impact parameter updates: latent mean (alpha) and variance (phi).
    ### Each covariate's impact parameter is updated in its own inner
    ### Newton-Raphson loop until that single parameter converges (eps_cd).
    ### Impact parameters are stored at positions num_items+1 (mean) and
    ### num_items+2 (variance) in the parameter list p_cd.
    #########################################################################

    # --- Latent mean impact updates (alpha parameters) ---
    # Loop over each covariate in the mean impact equation.
    for(cov in 1:ncol(mean_predictors)) {

      # Inner-loop initialization for this mean impact covariate.
      lastp_cd <- p_cd
      eps_cd <- Inf
      iter_cd <- 1

      # Inner Newton-Raphson loop: update this single alpha coefficient
      # until its own convergence criterion is satisfied.
      while(eps_cd > final_control$tol){

      # Compute first derivative (gradient) and second derivative (Hessian)
      # of the log-likelihood with respect to this alpha coefficient.
      # d_alpha returns a list: [[1]] = gradient, [[2]] = Hessian.
      anl_deriv <- d_alpha(c(p_cd[[num_items+1]],p_cd[[num_items+2]]),
                           etable,
                           theta,
                           mean_predictors,
                           var_predictors,
                           cov=cov,
                           samp_size,
                           num_items,
                           num_quad)

      # Newton-Raphson update: parameter = parameter - gradient / Hessian.
      p_cd[[num_items+1]][[cov]] <-
        p_cd[[num_items+1]][[cov]] - anl_deriv[[1]]/anl_deriv[[2]]

      # Inner convergence check: Euclidean distance between current and
      # previous full parameter vectors.
      eps_cd = sqrt(sum((unlist(p_cd)-unlist(lastp_cd))^2))

      # Snapshot current parameters for next iteration comparison.
      lastp_cd <- p_cd

      # Increment inner iteration counter.
      iter_cd = iter_cd + 1


      }


    }

    # --- Latent variance impact updates (phi parameters) ---
    # Loop over each covariate in the variance impact equation.
    for(cov in 1:ncol(var_predictors)) {

      # Inner-loop initialization for this variance impact covariate.
      lastp_cd <- p_cd
      eps_cd <- Inf
      iter_cd <- 1

      # Inner Newton-Raphson loop: update this single phi coefficient
      # until its own convergence criterion is satisfied.
      while(eps_cd > final_control$tol){

      # Compute gradient and Hessian for this phi coefficient.
      # d_phi returns a list: [[1]] = gradient, [[2]] = Hessian.
      anl_deriv <- d_phi(c(p_cd[[num_items+1]],p_cd[[num_items+2]]),
                         etable,
                         theta,
                         mean_predictors,
                         var_predictors,
                         cov=cov,
                         samp_size,
                         num_items,
                         num_quad)

      # Newton-Raphson update: parameter = parameter - gradient / Hessian.
      p_cd[[num_items+2]][[cov]] <-
        p_cd[[num_items+2]][[cov]] - anl_deriv[[1]]/anl_deriv[[2]]

      # Inner convergence check.
      eps_cd = sqrt(sum((unlist(p_cd)-unlist(lastp_cd))^2))

      # Snapshot current parameters for next iteration comparison.
      lastp_cd <- p_cd

      # Increment inner iteration counter.
      iter_cd = iter_cd + 1
      }

    }


    #########################################################################
    ### Item parameter updates.
    ### Loop over all items. The update strategy depends on item type:
    ###   - "cfa"  (Gaussian): single-step Newton-Raphson (no inner loop),
    ###            updates use p directly (not p_cd).
    ###   - "2pl"  (Bernoulli): inner Newton-Raphson loops per parameter,
    ###            updates use p_cd.
    ###   - else   (Categorical/graded): single-step Newton-Raphson (no
    ###            inner loop), updates use p directly (not p_cd).
    ###
    ### For non-anchor items, DIF parameters (c1, a1, s1) are updated
    ### after the baseline parameters. DIF updates apply a penalty
    ### threshold (lasso or MCP) to the Newton-Raphson intermediate value.
    #########################################################################
    for (item in 1:num_items) {

      # Construct per-response-category E-tables for non-CFA items.
      # Each etable_item[[resp]] zeros out rows where the person did not
      # endorse response category resp, so the likelihood contribution
      # is restricted to the correct category.
      if(item_type[item] != "cfa") {
        etable_item <- lapply(1:num_responses[item], function(x) etable)
        for(resp in 1:num_responses[item]) {
          etable_item[[resp]][which(
            !(item_data[,item] == resp)), ] <- 0
        }
      }

      if(item_type[item] == "cfa") {
        #####################################################################
        ### CFA (Gaussian/continuous) item updates.
        ### These use single-step Newton-Raphson without inner convergence
        ### loops. Updates modify p directly (not the p_cd working copy).
        ### Parameters: c0 (intercept), a0 (slope), s0 (residual variance),
        ###             c1/a1/s1 (DIF on intercept/slope/residual).
        #####################################################################

        # Intercept (c0) update: single Newton-Raphson step.
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

        # Slope (a0) update: skip for Rasch models (slope fixed at 1).
        if(item_type[item] != "Rasch") {
          a0_parms <- grep(paste0("a0_itm",item,"_"),names(p[[item]]),fixed=T)
          anl_deriv <- d_mu_gaussian("a0",
                                     p[[item]],
                                     etable_item,
                                     theta,
                                     item_data[,item],
                                     pred_data,
                                     cov=NULL,
                                     samp_size,
                                     num_items,
                                     num_quad)
          p[[item]] <- p[[item]][a0_parms] - anl_deriv[[1]]/anl_deriv[[2]]
        }

        # Residual variance (s0) update: single Newton-Raphson step.
        # If the updated residual variance goes negative, reset it to 1
        # to maintain a valid (positive) variance estimate.
        s0_parms <- grep(paste0("s0_itm",item,"_"),names(p[[item]]),fixed=T)
        anl_deriv <- d_sigma_gaussian("s0",
                                      p[[item]],
                                      etable_item,
                                      theta,
                                      item_data[,item],
                                      pred_data,
                                      cov=NULL,
                                      samp_size,
                                      num_items,
                                      num_quad)
        p[[item]][s0_parms][1] <- p[[item]][s0_parms][1] -
          anl_deriv[[1]]/anl_deriv[[2]]
        if(p[[item]][s0_parms][[1]] < 0) p[[item]][s0_parms][[1]] <- 1


        # Skip DIF updates for anchor items (their DIF is fixed at zero).
        if(!any(item == anchor)) {

          # Residual DIF (s1) updates: one Newton-Raphson step per covariate.
          # These are unpenalized (no thresholding applied).
          for(cov in 1:num_predictors) {
            s1_parms <-
              grep(paste0("s1_itm",item,"_cov",cov),names(p[[item]]),fixed=T)
            anl_deriv <- d_sigma_gaussian("s1",
                                          p[[item]],
                                          etable_item,
                                          theta,
                                          item_data[,item],
                                          pred_data,
                                          cov=cov,
                                          samp_size,
                                          num_items,
                                          num_quad)
            p[[item]][s1_parms][1] <- p[[item]][s1_parms][1] -
              anl_deriv[[1]]/anl_deriv[[2]]
          }

          # Flatten all parameters to a named vector for the under-
          # identification check below.
          p2 <- unlist(p)

          # Intercept DIF (c1) updates: penalized Newton-Raphson.
          # For each covariate, compute the Newton-Raphson intermediate z,
          # then apply lasso (soft threshold) or MCP (firm threshold).
          for(cov in 1:num_predictors){

            # Under-identification guard: if all items except one already
            # have nonzero intercept DIF on this covariate, adding another
            # would leave only one anchor, making the model unidentified.
            # This check only triggers when no explicit anchors are set,
            # alpha=1 (pure lasso), and the tau grid is sufficiently fine.
            if(is.null(anchor) &
               sum(p2[grep(paste0("c1(.*?)cov",cov),names(p2))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }

            c1_parms <-
              grep(paste0("c1_itm",item,"_cov",cov),names(p[[item]]),fixed=T)
            # Compute gradient and Hessian for the intercept DIF parameter.
            anl_deriv <- d_mu_gaussian("c1",
                                       p[[item]],
                                       etable_item,
                                       theta,
                                       item_data[,item],
                                       pred_data,
                                       cov,
                                       samp_size,
                                       num_items,
                                       num_quad)
            # Newton-Raphson intermediate value (pre-threshold).
            z <- p[[item]][c1_parms] - anl_deriv[[1]]/anl_deriv[[2]]
            # Collect z for max_tau identification.
            if(max_tau) id_max_z <- c(id_max_z,z)
            # Apply penalty thresholding to produce the final DIF estimate.
            p[[item]][c1_parms] <-
              ifelse(pen_type == "lasso",
                     soft_threshold(z,alpha,tau_current),
                     firm_threshold(z,alpha,tau_current,gamma))
          }

          # Slope DIF (a1) updates: penalized Newton-Raphson.
          # Same structure as intercept DIF above.
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

            # Skip slope DIF for Rasch items (slope is not estimated).
            if(item_type[item] != "Rasch"){
              a1_parms <-
                grep(paste0("a1_itm",item,"_cov",cov),names(p[[item]]),fixed=T)
              anl_deriv <- d_mu_gaussian("a1",
                                         p[[item]],
                                         etable_item,
                                         theta,
                                         item_data[,item],
                                         pred_data,
                                         cov,
                                         samp_size,
                                         num_items,
                                         num_quad)
              # Newton-Raphson intermediate, then threshold.
              z <- p[[item]][a1_parms] - anl_deriv[[1]]/anl_deriv[[2]]
              if(max_tau) id_max_z <- c(id_max_z,z)
              p[[item]][a1_parms] <- ifelse(pen_type == "lasso",
                                            soft_threshold(z,alpha,tau_current),
                                            firm_threshold(z,alpha,tau_current,gamma))
            }
          }
        }

        #####################################################################
        ### Bernoulli (2PL) item updates.
        ### Unlike CFA/categorical, 2PL items use inner Newton-Raphson
        ### convergence loops for each parameter. Updates modify p_cd.
        ### Parameters: c0 (intercept), a0 (slope),
        ###             c1 (intercept DIF), a1 (slope DIF).
        #####################################################################
      } else if(item_type[item] == "2pl") {

        # --- Intercept (c0) inner convergence loop ---
        lastp_cd <- p_cd
        eps_cd <- Inf
        iter_cd <- 1

        # Inner loop: Newton-Raphson updates for the intercept until
        # this single parameter converges.
        while(eps_cd > final_control$tol){

        # Compute gradient and Hessian for the 2PL intercept.
        anl_deriv <- d_bernoulli("c0",
                                 p_cd[[item]],
                                 etable_item,
                                 theta,
                                 pred_data,
                                 cov=0,
                                 samp_size,
                                 num_items,
                                 num_quad)
        # Newton-Raphson update for intercept.
        p_cd[[item]][[1]] <- p_cd[[item]][[1]] - anl_deriv[[1]]/anl_deriv[[2]]

        # Inner convergence check.
        eps_cd = sqrt(sum((unlist(p_cd)-unlist(lastp_cd))^2))

        # Snapshot current parameters for next iteration comparison.
        lastp_cd <- p_cd

        # Increment inner iteration counter.
        iter_cd = iter_cd + 1

        }

        # --- Slope (a0) inner convergence loop ---
        # Skip for Rasch items (slope is not estimated).
        if(item_type[item] != "Rasch") {

          lastp_cd <- p_cd
          eps_cd <- Inf
          iter_cd <- 1

          # Inner loop: Newton-Raphson updates for the slope until
          # this single parameter converges.
          while(eps_cd > final_control$tol){
          # Compute gradient and Hessian for the 2PL slope.
          anl_deriv <- d_bernoulli("a0",
                                   p_cd[[item]],
                                   etable_item,
                                   theta,
                                   pred_data,
                                   cov=0,
                                   samp_size,
                                   num_items,
                                   num_quad)
          # Newton-Raphson update for slope (stored at position 2).
          p_cd[[item]][[2]] <- p_cd[[item]][[2]] - anl_deriv[[1]]/anl_deriv[[2]]

          # Inner convergence check.
          eps_cd = sqrt(sum((unlist(p_cd)-unlist(lastp_cd))^2))

          # Snapshot current parameters for next iteration comparison.
          lastp_cd <- p_cd

          # Increment inner iteration counter.
          iter_cd = iter_cd + 1
          }

        }


        # Skip DIF updates for anchor items (their DIF is fixed at zero).
        if(!any(item == anchor)) {

          # Flatten all p_cd parameters for under-identification checks.
          p2_cd <- unlist(p_cd)

          # --- Intercept DIF (c1) updates: penalized, with inner loop ---
          # Each covariate's intercept DIF parameter gets its own inner
          # Newton-Raphson convergence loop. After each Newton-Raphson step,
          # the penalty threshold is applied to the intermediate z value.
          for(cov in 1:num_predictors) {

            # Under-identification guard for intercept DIF.
            # If all items except one already have nonzero c1 DIF on this
            # covariate, adding another would leave the model unidentified.
            if(is.null(anchor) &
               sum(p2_cd[grep(paste0("c1(.*?)cov",cov),names(p2_cd))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }

            # Inner-loop initialization for this intercept DIF covariate.
            lastp_cd <- p_cd
            eps_cd <- Inf
            iter_cd <- 1

            # Inner loop: Newton-Raphson + penalty threshold for c1.
            while(eps_cd > final_control$tol){

            # Compute gradient and Hessian for the intercept DIF parameter.
            anl_deriv <- d_bernoulli("c1",
                                     p_cd[[item]],
                                     etable_item,
                                     theta,
                                     pred_data,
                                     cov,
                                     samp_size,
                                     num_items,
                                     num_quad)
            # Newton-Raphson intermediate value (pre-threshold).
            # Position in parameter list: 2 (c0, a0) + covariate index.
            z <- p_cd[[item]][[2+cov]] - anl_deriv[[1]]/anl_deriv[[2]]
            # Collect z for max_tau identification.
            if(max_tau) id_max_z <- c(id_max_z,z)
            # Apply penalty thresholding to produce the final DIF estimate.
            p_cd[[item]][[2+cov]] <- ifelse(pen_type == "lasso",
                                         soft_threshold(z,alpha,tau_current),
                                         firm_threshold(z,alpha,tau_current,gamma))

            # Inner convergence check.
            eps_cd = sqrt(sum((unlist(p_cd)-unlist(lastp_cd))^2))

            # Snapshot current parameters for next iteration comparison.
            lastp_cd <- p_cd

            # Increment inner iteration counter.
            iter_cd = iter_cd + 1
            }

          }

          # --- Slope DIF (a1) updates: penalized, with inner loop ---
          # Same structure as intercept DIF above, but for slope DIF.
          for(cov in 1:num_predictors) {

            # Under-identification guard for slope DIF.
            if(is.null(anchor) &
               sum(p2_cd[grep(paste0("a1(.*?)cov",cov),names(p2_cd))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }

            # Skip slope DIF for Rasch items (slope is not estimated).
            if(item_type[item] != "Rasch") {

              # Inner-loop initialization for this slope DIF covariate.
              lastp_cd <- p_cd
              eps_cd <- Inf
              iter_cd <- 1

              # Inner loop: Newton-Raphson + penalty threshold for a1.
              while(eps_cd > final_control$tol){

              # Compute gradient and Hessian for the slope DIF parameter.
              anl_deriv <- d_bernoulli("a1",
                                       p_cd[[item]],
                                       etable_item,
                                       theta,
                                       pred_data,
                                       cov,
                                       samp_size,
                                       num_items,
                                       num_quad)
              # Newton-Raphson intermediate value (pre-threshold).
              # Position: 2 (c0, a0) + num_predictors (c1 slots) + cov index.
              z <- p_cd[[item]][[2+num_predictors+cov]] -
                anl_deriv[[1]]/anl_deriv[[2]]
              # Collect z for max_tau identification.
              if(max_tau) id_max_z <- c(id_max_z,z)
              # Apply penalty thresholding.
              p_cd[[item]][[2+num_predictors+cov]] <-
                ifelse(pen_type == "lasso",
                       soft_threshold(z,alpha,tau_current),
                       firm_threshold(z,alpha,tau_current,gamma))

              # Inner convergence check.
              eps_cd = sqrt(sum((unlist(p_cd)-unlist(lastp_cd))^2))

              # Snapshot current parameters for next iteration comparison.
              lastp_cd <- p_cd

              # Increment inner iteration counter.
              iter_cd = iter_cd + 1
              }

            }

          }

        }

        #####################################################################
        ### Categorical (graded response) item updates.
        ### These use single-step Newton-Raphson without inner convergence
        ### loops. Updates modify p directly (not the p_cd working copy).
        ### Parameters: c0 (first threshold/intercept), additional
        ###             thresholds (thr=2..K-1), a0 (slope),
        ###             c1 (intercept DIF), a1 (slope DIF).
        #####################################################################
      } else {

        # First threshold/intercept (c0, thr=-1) update.
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

        # Remaining threshold updates (thr=2 through K-1, where K is the
        # number of response categories).
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

        # Slope (a0) update: skip for Rasch models (slope fixed at 1).
        # The slope is stored at position num_responses[[item]] in the
        # parameter vector (after all threshold parameters).
        if(item_type[item] != "Rasch") {
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

        # Skip DIF updates for anchor items (their DIF is fixed at zero).
        if(!any(item == anchor)){

          # Flatten all parameters for under-identification checks.
          p2 <- unlist(p)

          # Intercept DIF (c1) updates: penalized Newton-Raphson.
          # Position in parameter vector: num_responses + covariate index.
          for(cov in 1:num_predictors) {

            # Under-identification guard for intercept DIF.
            if(is.null(anchor) &
               sum(p2[grep(paste0("c1(.*?)cov",cov),names(p2))] != 0) >
               (num_items - 1) &
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }

            # Compute gradient and Hessian for intercept DIF (categorical).
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
            # Newton-Raphson intermediate (pre-threshold).
            z <- p[[item]][[num_responses[[item]]+cov]] -
              anl_deriv[[1]]/anl_deriv[[2]]
            # Collect z for max_tau identification.
            if(max_tau) id_max_z <- c(id_max_z,z)
            # Apply penalty thresholding.
            p[[item]][[num_responses[[item]]+cov]] <-
              ifelse(pen_type == "lasso",
                     soft_threshold(z,alpha,tau_current),
                     firm_threshold(z,alpha,tau_current,gamma))
          }

          # Slope DIF (a1) updates: penalized Newton-Raphson.
          # Position in parameter vector: computed from the end of the
          # vector minus the number of predictors, plus the covariate index.
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

            # Skip slope DIF for Rasch items.
            if(item_type[item] != "Rasch") {
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
              # Newton-Raphson intermediate, then threshold.
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


        # End of item-type branching (CFA / 2PL / categorical).
      }

    } # End item loop.

      #####################################################################
      ### Outer-loop convergence check.
      ### After completing one full sweep over all impact and item
      ### parameters, check whether the overall parameter vector has
      ### changed by more than the tolerance since the last full sweep.
      #####################################################################
      p_cd_all <- p_cd

      # Outer convergence: Euclidean distance between the full parameter
      # vector at the end of this sweep vs. the end of the previous sweep.
      eps_cd_all = sqrt(sum((unlist(p_cd_all)-unlist(lastp_cd_all))^2))

      # Snapshot the full parameter vector for the next sweep comparison.
      lastp_cd_all <- p_cd_all

      # Print diagnostic: current outer iteration number.
      print(paste0("CD iter: ",iter_cd_all))
      # Increment outer iteration counter.
      iter_cd_all = iter_cd_all + 1

    } # End outer convergence loop.

    ###########################################################################
    ### Return results.
    ### If max_tau is TRUE, return only the maximum absolute pre-threshold z
    ### value across all DIF parameters (used to determine the maximum
    ### penalty needed to shrink all DIF to zero).
    ### Otherwise, return the converged parameter estimates and the
    ### under-identification flag.
    ###########################################################################
    if(max_tau) {
      id_max_z <- max(abs(id_max_z))
      return(id_max_z)
    } else {
      return(list(p=p_cd_all,
                  under_identified=under_identified))
    }

  }
