###############################################################################
## em_estimation.R
##
## Core EM (Expectation-Maximization) algorithm loop for penalized DIF
## (Differential Item Functioning) estimation in the regDIF package.
##
## This file implements the iterative EM procedure that:
##   1. Alternates between an E-step (computing posterior expectations of the
##      latent trait given observed data and current parameters) and an M-step
##      (maximizing the penalized complete-data log-likelihood to update item
##      and structural parameters).
##   2. Monitors convergence via the Euclidean distance between successive
##      parameter vectors (stored in `eps`).
##   3. Records the full parameter trajectory across EM iterations in
##      `estimator_history`, which is later used by the supplemental EM
##      algorithm to compute standard errors.
##   4. After convergence, computes information criteria (AIC/BIC), EAP
##      (Expected A Posteriori) trait scores, and optionally identifies the
##      maximum penalty (tau) that shrinks all DIF effects to zero.
##
## The function is called once per tau value along the regularization path.
## It is an internal workhorse and is not exported to the user.
###############################################################################

#' Penalized expectation-maximization algorithm.
#'
#' Runs the EM algorithm for a single penalty (tau) value along the
#' regularization path. The E-step computes posterior quadrature weights
#' (or uses proxy scores), and the M-step updates item and structural
#' parameters subject to a penalty on the DIF coefficients. Iteration
#' continues until the Euclidean distance between successive parameter
#' vectors falls below the convergence tolerance or the iteration limit
#' is reached.
#'
#' @param p List of parameters with starting values obtained from preprocess.
#'   This list is organized by parameter type (e.g., item intercepts,
#'   slopes, DIF effects, and impact/mean-variance parameters).
#' @param item_data Matrix or data frame of item responses.
#' @param pred_data Matrix or data frame of DIF and/or impact predictors.
#' @param prox_data Vector of observed proxy scores. When non-NULL, a
#'   proxy E-step is used instead of numerical quadrature.
#' @param mean_predictors Possibly different matrix of predictors for the mean
#'   impact equation.
#' @param var_predictors Possibly different matrix of predictors for the
#'   variance impact equation.
#' @param item_type Character value or vector indicating the type of
#'   item to be modeled (e.g., \code{"rasch"}, \code{"2pl"}, \code{"graded"}).
#' @param theta Vector of fixed quadrature points for numerical integration
#'   over the latent trait.
#' @param pen_type Character value indicating the penalty function to use
#'   (e.g., \code{"lasso"}, \code{"ridge"}, \code{"mcp"}, \code{"enet"}).
#' @param tau_vec Vector of tau values that either are automatically generated
#'   or provided by the user. The first \code{tau_vec} will be equal to \code{Inf}
#'   to identify a minimal value of tau in which all DIF is removed from the
#'   model.
#' @param id_tau Logical indicating whether to identify the minimum value of tau
#'   in which all DIF parameters are removed from the model. When TRUE, an
#'   additional M-step call with \code{max_tau = TRUE} is run after convergence
#'   to find the largest tau that zeroes out all DIF.
#' @param num_tau Numeric value indicating the number of tau values to run
#'   regDIF on.
#' @param alpha Numeric value indicating the alpha parameter in the elastic net
#'   penalty function. Controls the mix between L1 (lasso) and L2 (ridge)
#'   penalties: alpha = 1 is pure lasso, alpha = 0 is pure ridge.
#' @param gamma Numeric value indicating the gamma parameter in the MCP
#'   (minimax concave penalty) function. Controls the concavity of the
#'   penalty; larger values make MCP behave more like lasso.
#' @param pen Index for the tau vector, indicating which penalty value in
#'   \code{tau_vec} is currently being fit.
#' @param pen.deriv Logical value indicating whether to use the second
#'   derivative of the penalized parameter during regularization. The default is
#'   TRUE.
#' @param anchor Optional numeric value or vector indicating which item
#'   response(s) are anchors (e.g., \code{anchor = 1}). Anchor items are
#'   assumed DIF-free and their DIF parameters are not penalized.
#' @param final_control Control parameters (list), including convergence
#'   tolerance (\code{tol}) and maximum iterations (\code{maxit}).
#' @param samp_size Numeric value indicating the sample size.
#' @param num_items Numeric value indicating the number of items.
#' @param num_responses Vector with number of responses for each item.
#' @param num_predictors Numeric value indicating the number of predictors.
#' @param num_quad Numeric value indicating the number of quadrature points.
#' @param adapt_quad Logical value indicating whether to use adaptive quad.
#'   needs to be identified.
#' @param optim_method Character value indicating the type of optimization
#'   method to use.
#' @param estimator_history List to save EM iterations for the supplemental EM
#'   algorithm. Each element corresponds to a tau value and stores a matrix
#'   where columns are EM iterations and rows are parameter values (plus the
#'   observed log-likelihood in the last row).
#' @param estimator_limit Logical value indicating whether the EM algorithm reached
#'   the maxit limit in the previous estimation round. This is tracked because
#'   non-convergence can indicate problems (especially with non-convex penalties
#'   like MCP).
#' @param NA_cases Logical vector indicating if observation is missing.
#' @param exit_code Integer indicating if the model has converged properly.
#'   Values: 0 = success, 1 = iteration limit reached, 4 = M-step failure.
#'
#' @return a \code{"list"} of matrices with unprocessed model estimates,
#'   containing:
#'   \describe{
#'     \item{p}{Final parameter estimates after EM convergence.}
#'     \item{infocrit}{Information criteria (AIC, BIC) for model selection.}
#'     \item{max_tau}{Maximum tau that removes all DIF (NULL if not requested).}
#'     \item{estimator_history}{Parameter trajectory across EM iterations.}
#'     \item{under_identified}{Logical flag if model became under-identified.}
#'     \item{estimator_limit}{Logical flag if EM hit the iteration limit.}
#'     \item{eap}{EAP (Expected A Posteriori) trait score estimates.}
#'     \item{exit_code}{Final exit code indicating convergence status.}
#'   }
#'
#' @keywords internal
#'
em_estimation <- function(p,
                          item_data,
                          pred_data,
                          prox_data,
                          mean_predictors,
                          var_predictors,
                          item_type,
                          theta,
                          pen_type,
                          tau_vec,
                          id_tau,
                          num_tau,
                          alpha,
                          gamma,
                          pen,
                          pen.deriv,
                          anchor,
                          final_control,
                          samp_size,
                          num_items,
                          num_responses,
                          num_predictors,
                          num_quad,
                          adapt_quad,
                          optim_method,
                          estimator_history,
                          estimator_limit,
                          NA_cases,
                          exit_code) {


  ###########################################################################
  ## Initialization
  ###########################################################################

  # Store the initial parameter values so we can measure change after the
  # first iteration. `lastp` always holds the parameter vector from the
  # previous EM iteration.
  lastp <- p

  # `eps` tracks convergence: it is the Euclidean distance (L2 norm) between
  # the current and previous parameter vectors. Initialized to Inf so the
  # while-loop is guaranteed to execute at least once.
  eps <- Inf

  # EM iteration counter (1-indexed; incremented at the end of each cycle).
  iter <- 1

  # Total number of models to fit along the regularization path. When
  # `id_tau` is TRUE, the first tau value (Inf) is used only to identify the
  # maximum tau, so the count comes from `num_tau`; otherwise it is the full
  # length of the user-supplied or auto-generated tau vector.
  models_to_fit <- ifelse(id_tau,num_tau,length(tau_vec))

  ###########################################################################
  ## Main EM Loop
  ##
  ## Each iteration performs:
  ##   (a) E-step  -- compute posterior quadrature weights (responsibilities)
  ##                  for each person x quadrature-point combination, given
  ##                  current parameters. This yields the expected complete-
  ##                  data sufficient statistics needed by the M-step.
  ##   (b) M-step  -- maximize the penalized Q-function (expected complete-
  ##                  data log-likelihood minus the penalty on DIF parameters)
  ##                  to obtain updated parameter estimates.
  ##   (c) Convergence check -- compute eps = ||p_new - p_old||_2. If eps
  ##                  drops below `final_control$tol`, the loop terminates.
  ###########################################################################
  while(eps > final_control$tol && iter < final_control$maxit){


    # ----- E-step ---------------------------------------------------------
    # Compute posterior weights over quadrature points for each person.
    # When `prox_data` is NULL (the typical case), the standard E-step uses
    # Gauss-Hermite quadrature (optionally adaptive). The E-step also
    # returns the marginal (observed-data) log-likelihood, which is stored
    # in `estimator_history` for use in the supplemental EM standard errors.
    # `get_eap = FALSE` because EAP scores are only needed after convergence.
    eout <- if(is.null(prox_data)) Estep(p,
                                         item_data,
                                         pred_data,
                                         item_type,
                                         mean_predictors,
                                         var_predictors,
                                         theta,
                                         samp_size,
                                         num_items,
                                         num_responses,
                                         adapt_quad,
                                         num_quad,
                                         get_eap = FALSE,
                                         NA_cases = NA_cases)



    # ----- M-step ---------------------------------------------------------
    # Maximize the penalized Q-function to update all model parameters.
    # Wrapped in tryCatch because numerical optimization can fail (e.g.,
    # singular Hessian, non-convergence of the inner optimizer). If the
    # M-step fails, `mout` is set to NULL and the EM loop breaks with
    # exit_code = 4 (see below).
    # `tau_vec[pen]` selects the current penalty strength; `max_tau = FALSE`
    # means this is a regular estimation M-step (not the tau-identification
    # M-step that runs after convergence).
    mout <- tryCatch(
      {
        Mstep_simple(p,
                     item_data,
                     pred_data,
                     prox_data,
                     mean_predictors,
                     var_predictors,
                     eout,
                     item_type,
                     pen_type,
                     tau_vec[pen],
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
                     max_tau = FALSE,
                     optim_method)
      },
      error = function(e) {e; return(NULL)},
      warning = function(w) {} )


    # If the M-step failed (returned NULL), set exit_code to 4 and abort
    # the EM loop. The caller will detect exit_code == 4 and handle the
    # failure gracefully (typically by skipping this tau value).
    if(is.null(mout)) {
      exit_code <- 4
      break
    }

    # ----- Parameter update -----------------------------------------------
    # Extract the updated parameter list from the M-step output.
    p <- mout$p

    # ----- Convergence check ----------------------------------------------
    # Compute the Euclidean distance (L2 norm) between the current and
    # previous parameter vectors. Both `p` and `lastp` are nested lists,
    # so they are unlisted into flat numeric vectors before computing the
    # norm. Convergence is declared when eps < final_control$tol.
    eps = sqrt(sum((unlist(p)-unlist(lastp))^2))

    # ----- Record parameter trajectory for supplemental EM ----------------
    # The supplemental EM algorithm (used for computing standard errors)
    # requires the full history of parameter estimates and the observed-data
    # log-likelihood at each EM iteration. Each column of
    # `estimator_history[[pen]]` corresponds to one EM iteration; rows
    # contain the unlisted parameter vector followed by the observed LL.
    eout_obs_ll <- ifelse(is.null(eout), NA, eout$observed_ll)

    if(!is.null(eout)) {
      estimator_history[[pen]][,iter] <- c(unlist(p), eout_obs_ll)
    } else {
      estimator_history[[pen]][,iter] <- NA
    }


    # If the algorithm has not yet converged, expand the history matrix by
    # one column to make room for the next iteration's estimates.
    if(eps > final_control$tol) {
      estimator_history[[pen]] <- cbind(estimator_history[[pen]],
                                        matrix(0,ncol=1,nrow=length(unlist(p))+1))
    }

    # Save the current parameters as `lastp` for the next convergence check.
    lastp <- p


    # ----- Iteration bookkeeping ------------------------------------------
    # Increment the iteration counter.
    iter = iter + 1

    # If the maximum number of EM iterations has been reached without
    # convergence, issue a warning. The `estimator_limit` flag is set to
    # TRUE so that downstream code knows this model did not converge.
    # This is especially relevant for non-convex penalties (e.g., MCP),
    # where non-convergence may signal multimodality or a poor starting
    # point.
    if(iter == final_control$maxit) {
      warning("Iteration limit reached without convergence", call. = FALSE, immediate. = TRUE)
      estimator_limit <- T
      exit_code <- exit_code + 1
    }

    # ----- Console progress reporting --------------------------------------
    # Print a carriage-return-based progress line showing: how many models
    # (tau values) have been completed so far, the current EM iteration
    # number, and the latest convergence criterion (eps). The `\r` overwrites
    # the previous line in the console for a clean progress display.
    # if(final_control$optim.method == "CD") {
    #   cat('\r', '                                         ',
    #       sprintf("Models Completed: %d of %d  EM Iteration: %d  EM Change: %f",
    #               pen - 1,
    #               models_to_fit,
    #               iter,
    #               round(eps, nchar(final_control$tol))))
    # } else {
      cat('\r', sprintf("Models Completed: %d of %d  Iteration: %d  Change: %f",
                        pen - 1,
                        models_to_fit,
                        iter,
                        round(eps, nchar(final_control$tol))))
    # }



    utils::flush.console()

    # ----- Under-identification guard -------------------------------------
    # If the M-step detected that the model would become under-identified
    # (e.g., too many DIF parameters relative to sample size for this tau),
    # stop the EM loop early for this tau value.
    if(mout$under_identified) break
    # if(!is.null(prox_data)) break



  }

  ###########################################################################
  ## Post-convergence computations
  ###########################################################################

  # If the M-step failed during the EM loop (exit_code == 4), return NULL
  # immediately. The caller will skip this tau value on the regularization
  # path.
  if(exit_code == 4) return(NULL)

  # eout <- if(!is.null(prox_data)) Estep_proxy(p,
  #                                            item_data,
  #                                            pred_data,
  #                                            item_type,
  #                                            mean_predictors,
  #                                            var_predictors,
  #                                            prox_data,
  #                                            samp_size,
  #                                            num_items,
  #                                            num_responses,
  #                                            get_eap = FALSE,
  #                                            NA_cases = NA_cases)

  # ----- Information criteria (AIC / BIC) ---------------------------------
  # Compute AIC and BIC based on the observed-data log-likelihood from the
  # final E-step and the effective number of parameters (accounting for
  # parameters that have been penalized to zero). These criteria are used
  # for selecting the optimal tau along the regularization path.
  infocrit <- information_criteria(eout,
                                   p,
                                   item_data,
                                   pred_data,
                                   prox_data,
                                   mean_predictors,
                                   var_predictors,
                                   item_type,
                                   gamma,
                                   samp_size,
                                   num_responses,
                                   num_items,
                                   num_quad)

  # ----- EAP (Expected A Posteriori) trait scores -------------------------
  # Run a final E-step with `get_eap = TRUE` to compute person-level
  # trait estimates. The EAP score for each person is the posterior mean
  # of the latent trait distribution given their item responses and the
  # converged parameter estimates. EAP computation is skipped when proxy
  # scores are used, since the proxy itself serves as the trait estimate.
  eout_eap <- if(is.null(prox_data)) {
    Estep(p,
          item_data,
          pred_data,
          item_type,
          mean_predictors,
          var_predictors,
          theta,
          samp_size,
          num_items,
          num_responses,
          adapt_quad,
          num_quad,
          get_eap = TRUE,
          NA_cases = NA_cases)
  } else {
    NULL
  }

  # ----- Identify maximum tau (regularization path construction) ----------
  # When `id_tau = TRUE`, we need to find the smallest penalty value (tau)
  # that is large enough to shrink ALL DIF parameters to exactly zero.
  # This "max tau" serves as the starting point of the regularization path:
  # any tau >= max_tau yields a model with no DIF, and the path then
  # decreases tau to progressively allow DIF effects to enter the model.
  #
  # This is accomplished by calling Mstep_simple with `max_tau = TRUE`,
  # which triggers special logic inside the M-step to compute the largest
  # penalty gradient (subgradient) across all DIF parameters. That value
  # defines the threshold above which all DIF is zeroed out.
  if(id_tau) {

    max_tau <- Mstep_simple(p,
                     item_data,
                     pred_data,
                     prox_data,
                     mean_predictors,
                     var_predictors,
                     eout,
                     item_type,
                     pen_type,
                     tau_vec[1],
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
                     max_tau = TRUE,
                     optim_method)


  } else {
    # If tau identification is not requested (e.g., the user supplied their
    # own tau vector), skip this step.
    max_tau <- NULL
  }

  ###########################################################################
  ## Return results
  ##
  ## The returned list is consumed by the outer regularization-path loop
  ## (in regDIF_main) which aggregates results across all tau values.
  ###########################################################################
  return(list(p=p,                                # Final parameter estimates
              # complete_info=mout$inv_hess_diag,
              infocrit=infocrit,                  # AIC / BIC
              max_tau=max_tau,                    # Max tau for path (or NULL)
              # max_tau=mout$id_max_z,
              estimator_history=estimator_history, # Full EM trajectory (for SEM SEs)
              under_identified=mout$under_identified, # Under-identification flag
              estimator_limit=estimator_limit,    # TRUE if EM hit maxit
              eap=eout_eap,                       # EAP trait scores (or NULL)
              exit_code=exit_code))               # Convergence status code

}
