###############################################################################
### M-Step: Block Multivariate Newton-Raphson (MNR) Parameter Updates
###############################################################################
#
# This file implements the maximization (M) step of the EM algorithm for the
# regDIF model using a block-wise multivariate Newton-Raphson (MNR) approach.
#
# Overview:
#   The M-step updates two classes of parameters:
#     1. Latent impact parameters -- mean and variance of the latent trait as
#        functions of covariates. These are updated jointly (as a single block)
#        using the combined gradient vector and Hessian matrix.
#     2. Item parameters -- intercept (c0), slope (a0), and DIF coefficients
#        (c1, a1 per covariate) for each item. These are updated jointly per
#        item using item-level gradients and Hessians.
#
# Newton-Raphson update formula:
#   theta_new = theta_old - H^{-1} * g
#   where H is the Hessian matrix and g is the gradient vector.
#
# After the MNR update, DIF parameters (c1 and a1 coefficients) are subjected
# to penalized thresholding (e.g., lasso soft-thresholding or MCP firm-
# thresholding) to perform variable selection. Group penalty variants
# (grp.lasso, grp.mcp) threshold the c1 and a1 DIF parameters for a given
# covariate jointly as a group.
#
# Model identification:
#   The function monitors whether the model becomes under-identified. This
#   occurs when too many DIF parameters are non-zero on a single covariate
#   (i.e., fewer than one anchor item remains per covariate). When detected,
#   estimation halts early via the `under_identified` flag.
#
# Parallel computation:
#   Item-level updates can be run in parallel using the foreach/%dopar%
#   framework. When enabled, required objects are exported to the cluster
#   and item updates are distributed across workers.
#
# Special modes:
#   - max_tau: When TRUE, the function does not return parameter updates.
#     Instead, it identifies the maximum absolute MNR proposal value (z)
#     across all DIF parameters, which represents the minimum penalty (tau)
#     needed to shrink all DIF parameters to zero.
#
# Parameter list structure (p):
#   p[[1]] through p[[num_items]]   -- item parameter vectors
#   p[[num_items+1]]                -- mean impact coefficients
#   p[[num_items+2]]                -- variance impact coefficients
#
# Item parameter vector layout (for a 2PL item with K covariates):
#   [1]   c0            -- baseline intercept (non-DIF)
#   [2]   a0            -- baseline slope (non-DIF)
#   [3:(2+K)]           -- c1 DIF intercept coefficients (one per covariate)
#   [(3+K):(2+2K)]      -- a1 DIF slope coefficients (one per covariate)
#
###############################################################################

#' Maximization step using latent variable and item response blocks.
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
#'
#' @return a \code{"list"} of estimates obtained from the maximization step using multivariate
#' Newton-Raphson
#'
#' @importFrom foreach %dopar%
#'
#' @keywords internal
#'
Mstep_block <-
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

    # Set under-identified model to FALSE until proven TRUE.
    # This flag is flipped to TRUE if too many DIF parameters are non-zero,
    # leaving fewer than one anchor item per covariate (model not identified).
    under_identified <- FALSE

    # When max_tau mode is active, initialize collector for MNR proposal values.
    # These z-values represent the unpenalized Newton-Raphson updates for DIF
    # parameters; the maximum absolute value gives the tau needed to zero them.
    if(max_tau) id_max_z <- 0

    ###########################################################################
    ## Block 1: Latent Impact Parameter Updates (Mean + Variance)
    ###########################################################################
    # Impact parameters model the latent trait distribution as a function of
    # covariates: E(theta|X) = X * beta_mean, Var(theta|X) = exp(X * beta_var).
    # Mean and variance coefficients are updated jointly in one MNR block,
    # using the combined gradient and Hessian over all impact parameters.

    if(is.null(prox_data)) {
      # Standard latent variable approach: compute joint gradient (element [[1]])
      # and Hessian (element [[2]]) for mean and variance impact parameters.
      anl_deriv_impact <- d_impact_block(p[[num_items+1]],
                                         p[[num_items+2]],
                                         eout$etable,
                                         eout$theta,
                                         mean_predictors,
                                         var_predictors,
                                         samp_size,
                                         num_items,
                                         num_quad,
                                         num_predictors)
    } else {
      # Proxy score approach: uses observed proxy scores instead of the full
      # latent variable quadrature for impact parameter estimation.
      anl_deriv_impact <- d_impact_block_proxy(p[[num_items+1]],
                                               p[[num_items+2]],
                                               prox_data,
                                               mean_predictors,
                                               var_predictors,
                                               samp_size,
                                               num_items,
                                               num_quad,
                                               num_predictors)
    }


    # MNR update for impact parameters:
    # Invert the Hessian, then apply: theta_new = theta_old - H^{-1} * gradient.
    inv_hess_impact <- solve(anl_deriv_impact[[2]])

    # Save the negative diagonal of the inverse Hessian for potential use in
    # standard error computation (asymptotic variance estimates).
    inv_hess_impact_diag <- -diag(inv_hess_impact)

    # Compute the Newton-Raphson proposal for all impact parameters jointly.
    m <- c(p[[num_items+1]],p[[num_items+2]]) -
      inv_hess_impact %*% anl_deriv_impact[[1]]
    names(m) <- names(c(p[[num_items+1]],p[[num_items+2]]))


    # Store the updated mean impact coefficients back into the parameter list.
    p[[num_items+1]] <- m[1:ncol(mean_predictors)]
    # Store the updated variance impact coefficients back into the parameter list.
    p[[num_items+2]] <- m[(ncol(mean_predictors)+1):length(m)]


    # Initialize storage for inverse Hessian diagonals across all parameter
    # blocks (num_items item blocks + 2 impact blocks for mean and variance).
    inv_hess_diag <- vector('list',num_items+2)

    ###########################################################################
    ## Block 2: Item Parameter Updates
    ###########################################################################
    # Each item's parameters (c0, a0, c1_1..c1_K, a1_1..a1_K) are updated as
    # a single block using the item-level gradient and Hessian.
    #
    # The update proceeds in two stages per item:
    #   Stage 1: MNR update -- compute unpenalized Newton-Raphson proposals (z)
    #            for all item parameters jointly.
    #   Stage 2: Penalization -- apply thresholding operators to the DIF
    #            parameters (c1, a1) using the current tau penalty value.
    #            Baseline parameters (c0, a0) are never penalized.

    # Item response updates.
    if(final_control$parallel[[1]]) {

      #########################################################################
      ## Parallel item updates (foreach/%dopar%)
      #########################################################################
      # Export all required objects to the parallel cluster workers so they
      # are available in the foreach loop environment.
      parallel::clusterExport(final_control$parallel[[2]], c("num_responses", "pred_data", "item_data",
                          "samp_size", "num_predictors", "num_quad", "num_items",
                          "prox_data", "item_type", "anchor", "pen_type",
                          "num_tau", "pen", "tau_vec", "alpha", "final_control",
                          "max_tau", "d_bernoulli_itemblock", "d_bernoulli_itemblock_proxy",
                          "grp_soft_threshold", "grp_firm_threshold", "soft_threshold",
                          "firm_threshold", "p", "eout", "inv_hess_diag", "tau_current",
                          "under_identified", "id_max_z"),
                    envir=environment())

      # Distribute item-level updates across parallel workers.
      # Each worker returns a list: [[1]] updated item parameters,
      # [[2]] max z-values for max_tau identification.
      p_items <- foreach::foreach(item=1:num_items) %dopar% {

        if(item_type[item] == "2pl") {

          #####################################################################
          ## Stage 1: MNR update for this item
          #####################################################################

          if(is.null(prox_data)) {
            # Compute joint gradient ([[1]]) and Hessian ([[2]]) for all
            # parameters of this binary item using the E-step posterior.
            anl_deriv_item <- d_bernoulli_itemblock(p[[item]],
                                                    eout$etable,
                                                    eout$theta,
                                                    pred_data,
                                                    item_data[,item],
                                                    samp_size,
                                                    num_items,
                                                    num_predictors,
                                                    num_quad)
          } else {
            # Proxy score variant: uses observed proxy scores in place of
            # quadrature-based posterior expectations.
            anl_deriv_item <- d_bernoulli_itemblock_proxy(p[[item]],
                                                          pred_data,
                                                          item_data[,item],
                                                          prox_data,
                                                          samp_size,
                                                          num_items,
                                                          num_predictors,
                                                          num_quad)
          }


          # Invert the item-level Hessian for the Newton-Raphson step.
          inv_hess_item <- solve(anl_deriv_item[[2]])

          # Save the negative diagonal of the inverse Hessian for this item,
          # used later for standard error computation.
          inv_hess_diag[[item]] <- -diag(inv_hess_item)

          # Compute unpenalized MNR proposals (z) for all item parameters.
          # z = theta_old - H^{-1} * gradient
          z <- p[[item]] - inv_hess_item %*% anl_deriv_item[[1]]

          # c0 update: baseline intercept is never penalized.
          p[[item]][[1]] <- z[1]

          # a0 update: baseline slope is never penalized.
          # Skip for Rasch models where slope is fixed.
          if(item_type[item] != "rasch") p[[item]][[2]] <- z[2]

          # Don't update DIF estimates if anchor item.
          # Anchor items have their DIF parameters fixed at zero to provide
          # identification constraints for the model.
          if(any(item == anchor)) {
            return(list(unlist(p[item]),id_max_z))
          }

          ###################################################################
          ## Stage 2: Penalized updates for DIF parameters
          ###################################################################
          # Flatten current parameters for model identification checks.
          p2 <- unlist(p)

          # Loop over each covariate to apply penalty to its DIF parameters.
          for(cov in 1:num_predictors) {

            #################################################################
            ## Group penalty path (grp.lasso or grp.mcp)
            #################################################################
            # Group penalties threshold the c1 and a1 DIF parameters for a
            # given covariate jointly, enforcing that both are shrunk to zero
            # (or both remain non-zero) together.
            if(pen_type == "grp.lasso" ||
               pen_type == "grp.mcp") {

              # Model identification check for group penalties:
              # If all but one item has non-zero DIF on this covariate (for
              # both c1 and a1 combined), the model is under-identified.
              # The check requires: no user-specified anchors, the count of
              # non-zero grouped DIF params exceeds (num_items*2 - 1), and
              # we are not on the first penalty with user start values.
              if(is.null(anchor) &&
                 sum(p2[c(grep(paste0("c1(.*?)cov",cov),names(p2)),
                          grep(paste0("a1(.*?)cov",cov),names(p2)))] != 0) >
                 (num_items*2 - 1) &&
                 (length(final_control$start.values) == 0 || pen > 1) &&
                 num_tau >= 10) {
                under_identified <- TRUE
                break
              }

              # In max_tau mode, collect the unpenalized MNR proposal values
              # for this covariate's DIF parameters (c1 and a1).
              if(max_tau) {
                id_max_z <- c(id_max_z,
                              z[2+cov],
                              z[2+num_predictors+cov])
              }


              # Apply the group thresholding operator to the paired (c1, a1)
              # DIF parameters for this covariate.
              grp.update <-
                if(pen_type == "grp.lasso") {
                  # Group lasso: soft-threshold on the L2 norm of the pair.
                  grp_soft_threshold(z[c(2+cov,
                                         2+num_predictors+cov)],
                                     tau_current)
                } else if(pen_type == "grp.mcp") {
                  # Group MCP: firm-threshold on the L2 norm of the pair.
                  grp_firm_threshold(z[c(2+cov,
                                         2+num_predictors+cov)],
                                     tau_current,
                                     gamma)
                }

              # c1 updates: penalized DIF intercept for this covariate.
              p[[item]][[2+cov]] <- grp.update[[1]]
              # a1 updates: penalized DIF slope for this covariate.
              p[[item]][[2+num_predictors+cov]] <- grp.update[[2]]

            }

            #################################################################
            ## Element-wise penalty path (lasso or mcp)
            #################################################################

            # Model identification check for c1 (intercept DIF):
            # If all but one item has non-zero c1 DIF on this covariate,
            # the model is under-identified. Only checked when alpha == 1
            # (pure lasso, not elastic net mix).
            if(is.null(anchor) &&
               sum(p2[grep(paste0("c1(.*?)cov",cov),names(p2))] != 0) >
               (num_items - 1) &&
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10) {
              under_identified <- TRUE
              break
            }

            # In max_tau mode, collect unpenalized MNR proposals for c1 and a1.
            if(max_tau) {
              id_max_z <- c(id_max_z,
                            z[2+cov],
                            z[2+num_predictors+cov])
            }

            # c1 updates: apply element-wise thresholding to the intercept
            # DIF parameter for this covariate.
            p[[item]][[2+cov]] <-
              if(pen_type == "lasso") {
                # Lasso: soft-threshold with elastic net mixing (alpha).
                soft_threshold(z[2+cov],
                               alpha,
                               tau_current)
              } else if(pen_type == "mcp") {
                # MCP: firm-threshold with gamma controlling concavity.
                firm_threshold(z[2+cov],
                               alpha,
                               tau_current,
                               gamma)
              }


            # a1 updates: skip slope DIF for Rasch items (slope is fixed).
            if(item_type[item] == "rasch") {
              return(list(unlist(p[item]),id_max_z))
            }

            # Model identification check for a1 (slope DIF):
            # Same logic as the c1 check above, but for slope DIF parameters.
            if(is.null(anchor) &&
               sum(p2[grep(paste0("a1(.*?)cov",cov),names(p2))] != 0) >
               (num_items - 1) &&
               alpha == 1 &&
               (length(final_control$start.values) == 0 || pen > 1) &&
               num_tau >= 10){
              under_identified <- TRUE
              break
            }

            # a1 updates: apply element-wise thresholding to the slope DIF
            # parameter for this covariate.
            p[[item]][[2+num_predictors+cov]] <-
              ifelse(pen_type == "lasso",
                     soft_threshold(z[2+num_predictors+cov],
                                    alpha,
                                    tau_current),
                     firm_threshold(z[2+num_predictors+cov],
                                    alpha,
                                    tau_current,
                                    gamma))

          }
          if(under_identified) return(list(unlist(p[item]),id_max_z))


        }
        return(list(unlist(p[item]),id_max_z))

    }

      # Collect parallel results: copy updated item parameters back into the
      # main parameter list from the foreach output.
      for(item in 1:num_items) {
        p[[item]] <- p_items[[item]][[1]]
      }

      } else{

        #######################################################################
        ## Sequential item updates (no parallelism)
        #######################################################################
        # Same logic as the parallel branch above, but items are updated
        # sequentially in a for loop. This means each item's update can use
        # the latest parameter values from previously updated items within
        # the same M-step iteration (Gauss-Seidel style).

        for (item in 1:num_items) {

          # Bernoulli responses (binary items with 2 response categories).
          if(num_responses[item] == 2) {

            ###################################################################
            ## Stage 1: MNR update for this item
            ###################################################################

            if(is.null(prox_data)) {
              # Compute joint gradient ([[1]]) and Hessian ([[2]]) for all
              # parameters of this binary item using the E-step posterior.
              anl_deriv_item <- d_bernoulli_itemblock(p[[item]],
                                                      eout$etable,
                                                      eout$theta,
                                                      pred_data,
                                                      item_data[,item],
                                                      samp_size,
                                                      num_items,
                                                      num_predictors,
                                                      num_quad)
            } else {
              # Proxy score variant: uses observed proxy scores in place of
              # quadrature-based posterior expectations.
              anl_deriv_item <- d_bernoulli_itemblock_proxy(p[[item]],
                                                            pred_data,
                                                            item_data[,item],
                                                            prox_data,
                                                            samp_size,
                                                            num_items,
                                                            num_predictors,
                                                            num_quad)
            }


            # Invert the item-level Hessian for the Newton-Raphson step.
            inv_hess_item <- solve(anl_deriv_item[[2]])

            # Save the negative diagonal of the inverse Hessian for this item,
            # used later for standard error computation.
            inv_hess_diag[[item]] <- -diag(inv_hess_item)

            # Compute unpenalized MNR proposals (z) for all item parameters.
            # z = theta_old - H^{-1} * gradient
            z <- p[[item]] - inv_hess_item %*% anl_deriv_item[[1]]

            # c0 update: baseline intercept is never penalized.
            p[[item]][[1]] <- z[1]

            # a0 update: baseline slope is never penalized.
            # Skip for Rasch models where slope is fixed.
            if(item_type[item] != "rasch") p[[item]][[2]] <- z[2]

            # Don't update DIF estimates if anchor item.
            # Anchor items have their DIF parameters fixed at zero to provide
            # identification constraints for the model.
            if(any(item == anchor)) next

            ###################################################################
            ## Stage 2: Penalized updates for DIF parameters
            ###################################################################
            # Flatten current parameters for model identification checks.
            p2 <- unlist(p)

            # Loop over each covariate to apply penalty to its DIF parameters.
            for(cov in 1:num_predictors) {

              #################################################################
              ## Group penalty path (grp.lasso or grp.mcp)
              #################################################################
              # Group penalties threshold the c1 and a1 DIF parameters for a
              # given covariate jointly, enforcing that both are shrunk to zero
              # (or both remain non-zero) together.
              if(pen_type == "grp.lasso" ||
                 pen_type == "grp.mcp") {

                # Model identification check for group penalties:
                # If all but one item has non-zero DIF on this covariate (for
                # both c1 and a1 combined), the model is under-identified.
                if(is.null(anchor) &&
                   sum(p2[c(grep(paste0("c1(.*?)cov",cov),names(p2)),
                            grep(paste0("a1(.*?)cov",cov),names(p2)))] != 0) >
                   (num_items*2 - 1) &&
                   (length(final_control$start.values) == 0 || pen > 1) &&
                   num_tau >= 10){
                  under_identified <- TRUE
                  break
                }

                # In max_tau mode, collect the unpenalized MNR proposal values
                # for this covariate's DIF parameters (c1 and a1).
                if(max_tau) {
                  id_max_z <- c(id_max_z,
                                z[2+cov],
                                z[2+num_predictors+cov])
                }


                # Apply the group thresholding operator to the paired (c1, a1)
                # DIF parameters for this covariate.
                grp.update <-
                  if(pen_type == "grp.lasso") {
                    # Group lasso: soft-threshold on the L2 norm of the pair.
                    grp_soft_threshold(z[c(2+cov,
                                           2+num_predictors+cov)],
                                       tau_current)
                  } else if(pen_type == "grp.mcp") {
                    # Group MCP: firm-threshold on the L2 norm of the pair.
                    grp_firm_threshold(z[c(2+cov,
                                           2+num_predictors+cov)],
                                       tau_current,
                                       gamma)
                  }

                # c1 updates: penalized DIF intercept for this covariate.
                p[[item]][[2+cov]] <- grp.update[[1]]
                # a1 updates: penalized DIF slope for this covariate.
                p[[item]][[2+num_predictors+cov]] <- grp.update[[2]]

                # Skip the element-wise penalty section below; group penalties
                # handle both c1 and a1 together.
                next

              }

              #################################################################
              ## Element-wise penalty path (lasso or mcp)
              #################################################################

              # Model identification check for c1 (intercept DIF):
              # If all but one item has non-zero c1 DIF on this covariate,
              # the model is under-identified. Only checked when alpha == 1
              # (pure lasso, not elastic net mix).
              if(is.null(anchor) &&
                 sum(p2[grep(paste0("c1(.*?)cov",cov),names(p2))] != 0) >
                 (num_items - 1) &&
                 alpha == 1 &&
                 (length(final_control$start.values) == 0 || pen > 1) &&
                 num_tau >= 10){
                under_identified <- TRUE
                break
              }

              # In max_tau mode, collect unpenalized MNR proposals for c1 and a1.
              if(max_tau) {
                id_max_z <- c(id_max_z,
                              z[2+cov],
                              z[2+num_predictors+cov])
              }

              # c1 updates: apply element-wise thresholding to the intercept
              # DIF parameter for this covariate.
              p[[item]][[2+cov]] <-
                if(pen_type == "lasso") {
                  # Lasso: soft-threshold with elastic net mixing (alpha).
                  soft_threshold(z[2+cov],
                                 alpha,
                                 tau_current)
                } else if(pen_type == "mcp") {
                  # MCP: firm-threshold with gamma controlling concavity.
                  firm_threshold(z[2+cov],
                                 alpha,
                                 tau_current,
                                 gamma)
                }


              # a1 updates: skip slope DIF for Rasch items (slope is fixed).
              if(item_type[item] == "rasch") next

              # Model identification check for a1 (slope DIF):
              # Same logic as the c1 check above, but for slope DIF parameters.
              if(is.null(anchor) &&
                 sum(p2[grep(paste0("a1(.*?)cov",cov),names(p2))] != 0) >
                 (num_items - 1) &&
                 alpha == 1 &&
                 (length(final_control$start.values) == 0 || pen > 1) &&
                 num_tau >= 10){
                under_identified <- TRUE
                break
              }

              # a1 updates: apply element-wise thresholding to the slope DIF
              # parameter for this covariate.
              p[[item]][[2+num_predictors+cov]] <-
                ifelse(pen_type == "lasso",
                       soft_threshold(z[2+num_predictors+cov],
                                      alpha,
                                      tau_current),
                       firm_threshold(z[2+num_predictors+cov],
                                      alpha,
                                      tau_current,
                                      gamma))

            }
            # If the model became under-identified during covariate loop,
            # stop updating further items.
            if(under_identified) break

          }
          }

        }


    ###########################################################################
    ## Collect Results and Return
    ###########################################################################

    # Store the inverse Hessian diagonals for impact parameters, splitting
    # mean and variance components into their respective list positions.
    inv_hess_diag[[num_items+1]] <-
      inv_hess_impact_diag[1:ncol(mean_predictors)]
    inv_hess_diag[[num_items+2]] <-
      inv_hess_impact_diag[-(1:ncol(mean_predictors))]

    if(max_tau) {
      # In max_tau mode, return the maximum absolute MNR proposal value across
      # all DIF parameters. This value represents the smallest penalty (tau)
      # that would shrink all DIF parameters to exactly zero.
      if(final_control$parallel[[1]]) {
        # Parallel: gather z-values from each worker's output (element [[2]]).
        id_max_z <- max(abs(unlist(sapply(1:num_items, function(items) p_items[[items]][[2]]))))
      } else{
        id_max_z <- max(abs(id_max_z))
      }
      return(id_max_z)
    } else {
      # Normal mode: return updated parameters, inverse Hessian diagonals
      # (for standard error computation), and the identification flag.
      return(list(p=p,
                  inv_hess_diag=inv_hess_diag,
                  under_identified=under_identified))
    }

  }
