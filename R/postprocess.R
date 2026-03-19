###############################################################################
###
### postprocess.R
###
### Post-processing of EM algorithm estimation results for regDIF.
###
### After the EM algorithm converges for a given tau (regularization penalty)
### value, this function organizes the raw parameter estimates into a
### user-friendly output format. It performs the following steps:
###   1. Extracts converged parameters, information criteria, and diagnostics
###      from the EM output.
###   2. Builds human-readable names for impact parameters (latent mean and
###      variance regression coefficients).
###   3. Builds human-readable names for item baseline parameters (intercepts/
###      thresholds and slopes) and DIF effect parameters (covariate effects
###      on intercepts and slopes).
###   4. Transforms graded-response threshold parameters from an offset
###      parameterization (relative to the first intercept) back to absolute
###      threshold values.
###   5. Stores all results in the `final` list, which accumulates output
###      across all tau values in the regularization path.
###   6. Reorders parameters so that intercepts/thresholds come first,
###      followed by slopes, then residual variances (for CFA items).
###   7. Checks for model under-identification and warns when a large jump
###      in the number of DIF parameters occurs between consecutive tau
###      values, which may indicate instability in the regularization path.
###
###############################################################################

#' Post-process EM estimation results into user-friendly output.
#'
#' Organizes raw EM algorithm estimates into named parameter vectors with
#' readable labels, transforms threshold parameters for graded items, and
#' stores results in the \code{final} list for a given tau value along the
#' regularization path.
#'
#' @param estimates List of converged parameters from the EM algorithm,
#'   including parameter values (\code{p}), information criteria
#'   (\code{infocrit}), EAP scores (\code{eap}), and convergence diagnostics.
#' @param item.data User-given matrix or data.frame of DIF and/or impact
#' predictors.
#' @param pred.data User-given matrix or data.frame of item responses.
#' @param prox.data User-given matrix or data.frame of observed proxy scores.
#' @param item_data Processed matrix or data.frame of item responses.
#' @param pred_data Processed matrix or data.frame of DIF and/or impact
#' predictors.
#' @param prox_data Processed matrix or data.frame of observed proxy scores.
#' @param mean_predictors Possibly different matrix of predictors for the mean
#' impact equation.
#' @param var_predictors Possibly different matrix of predictors for the
#' variance impact equation.
#' @param item_type Optional character value or vector indicating the type of
#' item to be modeled.
#' @param tau_vec Optional numeric vector of tau values.
#' @param num_tau Logical indicating whether the minimum tau value needs to be
#' identified during the regDIF procedure.
#' @param alpha Numeric value indicating the alpha parameter in the elastic net
#' penalty function.
#' @param pen Tuning parameter index (position along the tau regularization
#'   path, i.e., which tau value is currently being processed).
#' @param anchor Anchor item(s).
#' @param control Optional list of user-defined control parameters
#' @param final_control List of final control parameters.
#' @param final List of model results that accumulates output across all tau
#'   values. Each element (e.g., \code{final$impact}, \code{final$base},
#'   \code{final$dif}) is a matrix where columns correspond to tau values.
#' @param samp_size Sample size in dataset.
#' @param num_responses Number of responses for each item.
#' @param num_predictors Number of predictors.
#' @param num_items Number of items in dataset.
#' @param num_quad Number of quadrature points used for approximating the
#' latent variable.
#' @param NA_cases Logical vector indicating NA cases.
#'
#' @return a \code{"list"} object of processed \code{"regDIF"} results
#'
#' @keywords internal
#'
postprocess <-
  function(estimates,
           item.data,
           pred.data,
           prox.data,
           item_data,
           pred_data,
           prox_data,
           mean_predictors,
           var_predictors,
           item_type,
           tau_vec,
           num_tau,
           alpha,
           pen,
           anchor,
           control,
           final_control,
           final,
           samp_size,
           num_responses,
           num_predictors,
           num_items,
           num_quad,
           NA_cases) {

  ###########################################################################
  ## Extract converged estimates and diagnostics from the EM output.
  ##
  ## `p` is a list of parameter vectors: elements 1..num_items hold item
  ## parameters, element num_items+1 holds mean impact coefficients, and
  ## element num_items+2 holds variance impact coefficients.
  ## `infocrit` contains AIC, BIC, and the complete-data log-likelihood.
  ## `under_identified` is a logical flag set during estimation when the
  ## model becomes under-identified (e.g., too many free parameters for
  ## the given tau value).
  ###########################################################################
  p <- estimates$p
  infocrit <- estimates$infocrit
  estimator_history <- estimates$estimator_history
  # complete_info <- estimates$complete_info
  under_identified <- estimates$under_identified
  eap_scores <- estimates$eap$eap_scores
  eap_sd <- estimates$eap$eap_sd
  exit_code <- estimates$exit_code

  ###########################################################################
  ## Build human-readable names for impact (latent variable) parameters.
  ##
  ## Impact parameters model how the latent mean and variance depend on
  ## observed covariates. Names take the form "mean.<covariate>" and
  ## "var.<covariate>". If the user supplied separate predictor matrices
  ## for the mean and variance equations (via control$impact.mean.data or
  ## control$impact.var.data), column names are drawn from those matrices;
  ## otherwise, column names from the main predictor matrix (pred_data)
  ## are used, falling back to generic labels like "mean.cov1" if no
  ## column names exist.
  ###########################################################################
  # --- Mean impact parameter names ---
  # Use column names from user-supplied mean predictor data if available;
  # otherwise fall back to the main predictor data column names or generic
  # labels (e.g., "mean.cov1", "mean.cov2").
  if(is.null(control$impact.mean.data)) {
    if(is.null(colnames(pred_data)) |
       length(colnames(pred_data)) == 0) {
      mean_names <- paste0('mean.cov',1:ncol(mean_predictors))
    } else {
      mean_names <- paste0('mean.',colnames(pred_data))
    }

  } else {
    if(is.null(colnames(control$impact.mean.data)) |
       length(colnames(control$impact.mean.data)) == 0) {
      mean_names <- paste0('mean.cov',1:ncol(mean_predictors))
    } else {
      mean_names <- paste0('mean.',colnames(control$impact.mean.data))
    }

  }

  # --- Variance impact parameter names ---
  # Same logic as mean names but for the variance equation.
  if(is.null(control$impact.var.data)) {
    if(is.null(colnames(pred_data)) |
       length(colnames(pred_data)) == 0) {
      var_names <- paste0('var.cov',1:ncol(var_predictors))
    } else {
      var_names <- paste0('var.',colnames(pred_data))
    }

  } else {
    if(is.null(colnames(pred_data)) |
       length(colnames(pred_data)) == 0){
      var_names <- paste0('var.cov',1:ncol(var_predictors))
    } else {
      var_names <- paste0('var.',colnames(control$impact.var.data))
    }

  }

  # Combine mean and variance impact parameter values into a single vector.
  # In the `p` list, element num_items+1 = mean coefficients, num_items+2 =
  # variance coefficients.
  lv_parms <- c(p[[num_items+1]],p[[num_items+2]])

  # Build the full vector of impact parameter names. When the latent
  # variance is freely estimated (free.theta.var == TRUE), include an
  # explicit "var.intercept" label for the variance intercept term.
  if(final_control$free.theta.var){
    lv_names <- c(mean_names,"var.intercept",var_names)
  } else {
    lv_names <- c(mean_names,var_names)
  }

  ###########################################################################
  ## Build human-readable names for item baseline (non-DIF) parameters.
  ##
  ## Baseline parameters are the item intercepts (or thresholds for graded
  ## items), slopes (discrimination), and residual variances (CFA items
  ## only). Internal names use prefixes like "c0_item1_" (intercept),
  ## "a0_item1_" (slope), and "s0_item1_" (residual). These are mapped
  ## to user-friendly names like "item1.int.", "item1.slp.", "item1.res.".
  ##
  ## For graded (polytomous) items with K response categories, there are
  ## K-1 threshold parameters, so names are indexed: "item1.int.1",
  ## "item1.int.2", etc.
  ###########################################################################
  # Flatten the list of parameter vectors into a single named vector so
  # that individual parameters can be extracted by pattern matching on
  # their internal names (e.g., "c0_item1_cov1", "a0_item2_cov1").
  p2 <- unlist(p)
  all_items_parms_base <- NULL
  all_items_names_base <- NULL

  # Determine item names: use column names from item_data if available,
  # otherwise generate generic names ("item1", "item2", ...).
  if(is.null(colnames(item_data)) |
     length(colnames(item_data)) == 0){
    item_names <- paste0("item",1:num_items)

  } else{
    item_names <- colnames(item_data)

  }

  # Loop over items and collect baseline parameters and their names.
  # Internal naming convention:
  #   "c0_itemX_" = baseline intercept/threshold parameters for item X
  #   "a0_itemX_" = baseline slope (discrimination) for item X
  #   "s0_itemX_" = baseline residual variance for item X (CFA only)
  for(item in 1:num_items) {
    if(item_type[item] == "cfa") {
      # CFA items have intercept, slope, AND residual variance.
      item_parms_base <- c(p2[grep(paste0("c0_item",item,"_"),names(p2))],
                           p2[grep(paste0("a0_item",item,"_"),names(p2))],
                           p2[grep(paste0("s0_item",item,"_"),names(p2))])
      item_names_base <- c(paste0(item_names[item],".int."),
                           paste0(item_names[item],".slp."),
                           paste0(item_names[item],".res."))

    } else if(num_responses[item] == 2) {
      # Binary items have a single intercept and a single slope.
      item_parms_base <- c(p2[grep(paste0("c0_item",item,"_"),names(p2))],
                           p2[grep(paste0("a0_item",item,"_"),names(p2))])
      item_names_base <- c(paste0(item_names[item],".int."),
                           paste0(item_names[item],".slp."))

    } else {
      # Polytomous items (e.g., graded) have multiple intercept/threshold
      # parameters (one per category boundary, i.e., num_responses - 1)
      # but still a single slope.
      item_parms_base <- c(p2[grep(paste0("c0_item",item,"_"),names(p2))],
                           p2[grep(paste0("a0_item",item,"_"),names(p2))])
      item_names_base <- c(paste0(item_names[item],".int.",
                                  1:(num_responses[item]-1)),
                           paste0(item_names[item],".slp."))

    }
    # Accumulate parameters and names across all items.
    all_items_parms_base <- c(all_items_parms_base,item_parms_base)
    all_items_names_base <- c(all_items_names_base,item_names_base)

  }

  ###########################################################################
  ## Transform threshold parameters for graded (ordered categorical) items.
  ##
  ## During estimation, graded item thresholds are stored in an offset
  ## parameterization where the first parameter is the intercept and
  ## subsequent parameters are offsets (distances) from that intercept.
  ## Here we convert back to absolute thresholds:
  ##   absolute_threshold_k = intercept - offset_k
  ##
  ## This makes the output more interpretable: each threshold value
  ## directly represents the boundary between adjacent response categories
  ## on the latent scale.
  ###########################################################################
  if(any(item_type == "graded")) {
    for(item in 1:num_items) {
        if(item_type[item] == "graded") {
          # Extract offset parameters (all thresholds except the first
          # intercept) and the first intercept for this graded item.
          threshold_parms <- p2[grep(paste0("c0_item",item,"_"),
                                     names(p2))][2:(num_responses[item]-1)]
          intercept_parm <- p2[grep(paste0("c0_item",item,"_"),names(p2))][1]

          # Replace offset values with absolute thresholds in the output.
          all_items_parms_base[grep(paste0("c0_item",item,"_"),
                                    names(all_items_parms_base))][2:(
                                      num_responses[item]-1)] <-
            intercept_parm - threshold_parms
        }
    }
  }



  ###########################################################################
  ## Build human-readable names for item DIF (differential item functioning)
  ## parameters.
  ##
  ## DIF parameters capture how item intercepts, slopes, and residual
  ## variances (CFA only) vary as a function of observed covariates.
  ## Internal names use "c1_" (intercept DIF), "a1_" (slope DIF), and
  ## "s1_" (residual DIF) prefixes. These are mapped to user-friendly
  ## names like "item1.int.age", "item1.slp.gender", etc.
  ##
  ## Note: fixed=TRUE is used in grep() to match the prefix literally
  ## (e.g., "c1_item1_" should not match "c1_item10_" or "c1_item11_").
  ###########################################################################
  all_items_parms_dif <- NULL
  all_items_names_dif <- NULL

  # Determine covariate names for DIF parameter labeling.
  if(is.null(colnames(pred_data)) |
     length(colnames(pred_data)) == 0) {
    cov_names <- paste0("cov",1:num_predictors)

  } else {
    cov_names <- colnames(pred_data)

  }

  # Loop over items and collect DIF parameters and their names.
  # Each item has one DIF parameter per covariate for intercepts ("c1_")
  # and slopes ("a1_"), plus residual DIF ("s1_") for CFA items.
  for(item in 1:num_items) {
    if(item_type[item] == "cfa") {
      # CFA items: DIF on intercept, slope, AND residual variance.
      item_parms_dif <- c(p2[grep(paste0("c1_item",item,"_"),names(p2),fixed=T)],
                          p2[grep(paste0("a1_item",item,"_"),names(p2),fixed=T)],
                          p2[grep(paste0("s1_item",item,"_"),names(p2),fixed=T)])
      item_names_dif <- c(paste0(rep(item_names[item],
                                     each = num_predictors),'.int.',cov_names),
                          paste0(rep(item_names[item],
                                     each = num_predictors),'.slp.',cov_names),
                          paste0(rep(item_names[item],
                                     each = num_predictors),'.res.',cov_names))

    } else {
      # Non-CFA items (binary/graded): DIF on intercept and slope only.
      item_parms_dif <- c(p2[grep(paste0("c1_item",item,"_"),names(p2),fixed=T)],
                          p2[grep(paste0("a1_item",item,"_"),
                                  names(p2),
                                  fixed=T)])
      item_names_dif <- c(paste0(rep(item_names[item],
                                     each = num_predictors),'.int.',cov_names),
                          paste0(rep(item_names[item],
                                     each = num_predictors),'.slp.',cov_names))

    }
    # Accumulate DIF parameters and names across all items.
    all_items_parms_dif <- c(all_items_parms_dif,item_parms_dif)
    all_items_names_dif <- c(all_items_names_dif,item_names_dif)

  }

  ###########################################################################
  ## Handle under-identified models.
  ##
  ## If the EM algorithm detected that the model is under-identified for
  ## this tau value (e.g., too many free DIF parameters relative to the
  ## data), the results for this tau are left as NA in the `final` list
  ## and the function returns early. The progress message is still printed
  ## so the user can see where along the regularization path the issue
  ## occurred.
  ###########################################################################
  if(under_identified) {
    # Print progress message showing this model completed (with zeroed
    # iteration/change values to indicate no valid convergence).
    cat('\r',
        sprintf(paste0("Models Completed: %d of %d  Iteration: %d  Change: %d",
                       "              "),
                pen, length(tau_vec), 0, 0))

    if(pen != length(tau_vec)) utils::flush.console()
    return(final)
    }



  ###########################################################################
  ## Store results for the current tau value in the `final` list.
  ##
  ## The `final` list is the main output structure that accumulates results
  ## across all tau values in the regularization path. Each matrix in
  ## `final` has rows = parameters and columns = tau values, so column
  ## `pen` corresponds to the current tau. All parameter estimates are
  ## rounded to 4 decimal places for display.
  ###########################################################################
  final$tau_vec[pen] <- tau_vec[pen]
  final$aic[pen] <- round(infocrit$aic,4)
  final$bic[pen] <- round(infocrit$bic,4)
  final$impact[,pen] <- round(lv_parms,4)
  final$base[,pen] <- round(all_items_parms_base,4)
  final$dif[,pen] <- round(all_items_parms_dif,4)

  # EAP (expected a posteriori) scores and their standard deviations are
  # only stored when proxy scores are not provided (i.e., the latent
  # variable is estimated rather than observed).
  if(is.null(prox.data)) {
    final$eap$scores[,pen] <- eap_scores
    final$eap$sd[,pen] <- eap_sd
    final$estimator_history[[pen]] <- estimator_history[[pen]]
  }
  # final$complete_ll_info <- complete_info
  final$log_like[pen] <- infocrit$complete_ll

  # Store the original (unprocessed) user data for later use in
  # summary/print methods and for refitting.
  final$data <- list(item.data=item.data, pred.data=pred.data, prox.data=prox.data)

  # Assign human-readable row names to the parameter matrices.
  rownames(final$impact) <- lv_names
  rownames(final$base) <- all_items_names_base
  rownames(final$dif) <- all_items_names_dif
  # if(!(any(num_responses > 2)) && !is.null(complete_info)) {
  #   for(item in 1:num_items) {
  #     names(final$complete_ll_info[[item]]) <- names(p[[item]])
  #   }
  #   names(final$complete_ll_info[[num_items+1]]) <- names(p[[num_items+1]])
  #   names(final$complete_ll_info[[num_items+2]]) <- names(p[[num_items+2]])
  # }
  final$exit_code <- exit_code
  final$missing_obs <- which(NA_cases)

  ###########################################################################
  ## Reorder baseline parameters by parameter type.
  ##
  ## Up to this point, parameters are ordered by item (all parameters for
  ## item 1, then all for item 2, etc.). This block reorders them so that
  ## all intercepts/thresholds come first, then all slopes, then all
  ## residual variances. This grouping makes the output easier to scan
  ## when comparing across items.
  ###########################################################################
  final_int_thr_base <-
    final$base[grep(paste0(c("\\.int\\.","\\.thr\\."),
                           collapse = "|"),
                    rownames(final$base)),
               pen]
  final_slp_base <-
    final$base[grep("\\.slp\\.",
                    rownames(final$base)),
               pen]
  final_res_base <-
    final$base[grep("\\.res\\.",
                    rownames(final$base)),
               pen]
  final_names_base <-
    names(c(final_int_thr_base,final_slp_base,final_res_base))
  final$base[,pen] <-
    matrix(c(final_int_thr_base,final_slp_base,final_res_base), ncol = 1)
  rownames(final$base) <- final_names_base

  # Reorder DIF parameters by parameter type (same logic as baseline).
  final_int_dif <- final$dif[grep("\\.int\\.",
                                  rownames(final$dif)),
                             pen]
  final_slp_dif <- final$dif[grep("\\.slp\\.",
                                  rownames(final$dif)),
                             pen]
  final_res_dif <- final$dif[grep("\\.res\\.",
                                  rownames(final$dif)),
                             pen]
  final_names_dif <- names(c(final_int_dif,final_slp_dif,final_res_dif))
  final$dif[,pen] <-
    matrix(c(final_int_dif,final_slp_dif,final_res_dif), ncol = 1)
  rownames(final$dif) <- final_names_dif

  ###########################################################################
  ## Warn if there is a large jump in the number of non-zero DIF parameters
  ## between consecutive tau values.
  ##
  ## As tau decreases along the regularization path, more DIF parameters
  ## become non-zero. A large sudden jump (more parameters freed than
  ## num_predictors * num_items in a single step) suggests the tau grid
  ## is too coarse or the model lacks an anchor item for identification.
  ## The first DIF parameter is excluded from the count (hence -1) because
  ## it may serve as a reference.
  ###########################################################################
  if(pen > 1){
    # Count how many DIF parameters are zero at the previous and current
    # tau values (excluding the first DIF parameter).
    second_last <- sum(final$dif[-1,pen-1] == 0)
    last <- sum(final$dif[-1,pen] == 0)

    # If the decrease in zero-count (i.e., the increase in non-zero DIF
    # parameters) exceeds the threshold, issue a warning.
    if((second_last - last) > (num_predictors*num_items)) {
      warning(paste0("Large increase in the number of DIF parameters ",
                  "from iteration ",
                  pen-1,
                  " to ",
                  pen,
                  ".\n  Two Options:\n  1. Provide smaller differences ",
                  "between tau values.\n  2. Provide anchor item(s)."),
           call. = FALSE)
    }

  }

  # Re-extract the flattened parameter vector to check whether the largest
  # tau value was sufficient to penalize all DIF parameters to zero.
  # DIF parameters contain "cov" in their internal names.
  p2 <- unlist(estimates$p)
  dif_parms <- p2[grep(paste0("cov"),names(p2))]

  # Warn if tau does not remove all DIF (currently disabled).
  # if(is.null(anchor) &&
  #    pen == 1 &&
  #    sum(abs(dif_parms)) > 0 &&
  #    alpha == 1 &&
  #    num_tau >= 10
  #    ) {
  #   warning(paste0("\nAutomatically-generated or user-defined ",
  #                  "tau value is too small to penalize all parameters to ",
  #                  "zero without anchor item. Larger values of tau are ",
  #                  "recommended."), call. = FALSE)
  # }

  # Print a progress message to the console. When proxy scores are
  # provided, progress printing is suppressed (handled elsewhere).
  # flush.console() forces the carriage-return overwrite to display
  # immediately, except on the final tau value where we let the line
  # persist.
  if(is.null(prox.data)) {
    cat('\r',
        sprintf(paste0("Models Completed: %d of %d  Iteration: %d  Change: %d",
                       "              "),
                pen, length(tau_vec), 0, 0))
  }

  if(pen != length(tau_vec)) utils::flush.console()

  # Return the updated `final` list with results for this tau value added.
  return(final)

}
