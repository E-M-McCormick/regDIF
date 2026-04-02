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

          ## Intercept and slope DIF (c1, a1) updates -- penalized.
          # Build a derivative closure for CFA items that captures
          # item-specific data, keeping update_dif_groups item-type-agnostic.
          compute_deriv_cfa <- function(family_name, p_item, cov) {
            if (is.null(prox_data)) {
              d_mu_gaussian(family_name, p_item, etable, theta,
                            item_data[,item], pred_data, cov,
                            samp_size, num_items, num_quad)
            } else {
              d_mu_gaussian_proxy(family_name, p_item, prox_data,
                                  item_data[,item], pred_data, cov,
                                  samp_size)
            }
          }

          dif_result <- update_dif_groups(
            p                  = p,
            item               = item,
            item_type_item     = item_type[item],
            pen_type           = pen_type,
            tau_current        = tau_current,
            alpha              = alpha,
            gamma              = gamma,
            pen                = pen,
            pen.deriv          = pen.deriv,
            anchor             = anchor,
            num_items          = num_items,
            num_predictors     = num_predictors,
            num_tau            = num_tau,
            max_tau            = max_tau,
            final_control      = final_control,
            compute_deriv_fn   = compute_deriv_cfa,
            num_responses_item = num_responses[item]
          )

          p[[item]] <- dif_result$p_item
          if (dif_result$under_identified) {
            under_identified <- TRUE
            break
          }
          if (max_tau) id_max_z <- c(id_max_z, dif_result$id_max_z)

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

          # Build derivative closure for binary items.
          compute_deriv_bin <- function(family_name, p_item, cov) {
            if (is.null(prox_data)) {
              d_bernoulli(family_name, p_item, etable_item, theta,
                          pred_data, cov, samp_size, num_items, num_quad)
            } else {
              d_bernoulli_proxy(family_name, p_item, prox_data,
                                pred_data, item_data[,item], cov,
                                samp_size, num_items)
            }
          }

          dif_result <- update_dif_groups(
            p                  = p,
            item               = item,
            item_type_item     = item_type[item],
            pen_type           = pen_type,
            tau_current        = tau_current,
            alpha              = alpha,
            gamma              = gamma,
            pen                = pen,
            pen.deriv          = pen.deriv,
            anchor             = anchor,
            num_items          = num_items,
            num_predictors     = num_predictors,
            num_tau            = num_tau,
            max_tau            = max_tau,
            final_control      = final_control,
            compute_deriv_fn   = compute_deriv_bin,
            num_responses_item = num_responses[item]
          )

          p[[item]] <- dif_result$p_item
          if (dif_result$under_identified) {
            under_identified <- TRUE
            break
          }
          if (max_tau) id_max_z <- c(id_max_z, dif_result$id_max_z)

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

          # Build derivative closure for graded items.
          compute_deriv_graded <- function(family_name, p_item, cov) {
            if (is.null(prox_data)) {
              d_categorical(family_name, p_item, etable_item, theta,
                            pred_data, thr=-1, cov, samp_size,
                            num_responses[[item]], num_items, num_quad)
            } else {
              d_categorical_proxy(family_name, p_item, prox_data,
                                  pred_data, item_data[,item], thr=-1,
                                  cov, samp_size, num_responses[[item]],
                                  num_items)
            }
          }

          dif_result <- update_dif_groups(
            p                  = p,
            item               = item,
            item_type_item     = item_type[item],
            pen_type           = pen_type,
            tau_current        = tau_current,
            alpha              = alpha,
            gamma              = gamma,
            pen                = pen,
            pen.deriv          = pen.deriv,
            anchor             = anchor,
            num_items          = num_items,
            num_predictors     = num_predictors,
            num_tau            = num_tau,
            max_tau            = max_tau,
            final_control      = final_control,
            compute_deriv_fn   = compute_deriv_graded,
            num_responses_item = num_responses[item]
          )

          p[[item]] <- dif_result$p_item
          if (dif_result$under_identified) {
            under_identified <- TRUE
            break
          }
          if (max_tau) id_max_z <- c(id_max_z, dif_result$id_max_z)

        } # End anchor item conditional.


      } # End item type conditional.

    } # End looping through items.



    ###########################################################################
    ## Return results
    ###########################################################################

    if(max_tau) {

      # In max_tau mode, compute the smallest tau that zeros all DIF.
      #
      # For scalar penalties: max(|z|) — the largest absolute z value.
      # For group penalties:  max_g(||z_g||_2 / w_g) — the largest
      #   weighted group norm. This ensures the tau path is correctly
      #   scaled when groups contain multiple covariates (e.g., spline
      #   bases), where the L2 norm of a group can be much larger than
      #   any individual element.
      #
      # For default singleton groups with w=1, both formulas give the
      # same result (backward compatible).
      if (pen_type %in% c("grp.lasso", "grp.mcp") &&
          !is.null(final_control$group_spec$groups_idx)) {
        id_max_z <- compute_max_tau_groups(
          id_max_z   = unlist(id_max_z),
          group_spec = final_control$group_spec,
          num_items  = num_items,
          item_type  = item_type,
          anchor     = anchor
        )
      } else {
        id_max_z <- max(abs(unlist(id_max_z)))
      }

      return(id_max_z)

    } else {

        # In normal mode, return the updated parameter list and the
        # under-identification flag.
        return(list(p=p,
                    under_identified=under_identified))
    }

  }
