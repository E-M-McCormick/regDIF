###############################################################################
# Item Response Traceline Functions                                           #
#                                                                             #
# Compute item characteristic curves (ICCs) / tracelines for different item   #
# response types (binary, ordinal, continuous). These tracelines represent    #
# the probability (or density) of a response given the latent trait (theta)   #
# and person-level covariates (which produce DIF).                            #
#                                                                             #
# Each function has two variants:                                             #
#   - Standard: evaluates across a grid of quadrature points (theta)          #
#   - Proxy: evaluates at observed proxy scores instead of latent theta       #
#                                                                             #
# Parameter vector layout for binary/ordinal items:                           #
#   p_item = [c0, a0, c1_cov1..c1_covP, a1_cov1..a1_covP]                   #
#   where c0 = base intercept, a0 = base slope,                              #
#         c1 = intercept DIF effects, a1 = slope DIF effects                  #
###############################################################################

#' Binary item tracelines.
#'
#' Computes the 2PL item response function across a grid of theta values for
#' each person. The model includes person-specific DIF effects on both the
#' intercept and slope via covariates:
#'   P(X=1|theta,covs) = 1 / (1 + exp(-((c0 + c1*covs) + (a0 + a1*covs)*theta)))
#'
#' @param p_item Vector of item parameters: [c0, a0, c1_1..c1_P, a1_1..a1_P].
#' @param theta Vector of quadrature points for latent variable approximation.
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors
#'   (N x P, where P = number of covariates).
#' @param samp_size Sample size in dataset.
#'
#' @return a \code{"matrix"} of probability values for Bernoulli item likelihood
#'   (N x Q, where Q = number of quadrature points). Each cell gives
#'   P(X=1|theta_q, covs_i) for person i at quadrature point q.
#'
#' @keywords internal
#'
bernoulli_traceline_pts <-
  function(p_item,
           theta,
           pred_data,
           samp_size) {

    # Compute ICC for each quadrature point (columns) across all persons (rows).
    # The linear predictor is: (c0 + c1*X) + (a0 + a1*X)*theta
    # where X is the predictor matrix, c1/a1 are DIF effect vectors.
    traceline <-
      vapply(theta,
             function(x) {
               1 / (1 + exp(
                 -((p_item[1] + pred_data %*% p_item[3:(2+ncol(pred_data))]) +
                   (p_item[2] + pred_data %*% p_item[(3+ncol(pred_data)):length(p_item)])*x)
                 ))
               }, numeric(samp_size))

    return(traceline)

  }

#' Binary item tracelines (averaged across persons).
#'
#' Same as \code{bernoulli_traceline_pts} but returns the column means
#' (averaged across persons) instead of the full N x Q matrix. Useful for
#' obtaining marginal tracelines that average over the covariate distribution.
#'
#' @param p_item Vector of item parameters: [c0, a0, c1_1..c1_P, a1_1..a1_P].
#' @param theta Vector of quadrature points for latent variable approximation.
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#' @param samp_size Sample size in dataset.
#'
#' @return a \code{"numeric"} vector of length Q: average probability values
#'   across persons at each quadrature point.
#'
#' @keywords internal
#'
bernoulli_traceline_pts2 <-
  function(p_item,
           theta,
           pred_data,
           samp_size) {

    traceline <-
      vapply(theta,
             function(x) {
               1 / (1 + exp(-(
                 (p_item[1] + pred_data %*% p_item[3:(2+ncol(pred_data))]) +
                 (p_item[2] + pred_data %*% p_item[(3+ncol(pred_data)):length(p_item)])*x
                 )))
             }, numeric(samp_size))

    return(colMeans(traceline))

  }

#' Binary item tracelines for proxy scores.
#'
#' Evaluates the 2PL ICC at observed proxy scores (e.g., sum scores) rather
#' than at a grid of quadrature points. This avoids numerical integration
#' when an observed proxy for the latent variable is available.
#'
#' @param p_item Vector of item parameters: [c0, a0, c1_1..c1_P, a1_1..a1_P].
#' @param prox_data Vector of observed proxy scores (length N).
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#'
#' @return a \code{"matrix"} of probability values for Bernoulli item likelihood using observed
#' proxy scores (N x 1 matrix).
#'
#' @keywords internal
#'
bernoulli_traceline_pts_proxy <-
  function(p_item,
           prox_data,
           pred_data) {

    # Evaluate ICC at each person's observed proxy score (single value per person).
    traceline <-
      1 / (1 + exp(
        -((p_item[1] + pred_data %*% p_item[3:(2+ncol(pred_data))]) +
            (p_item[2] + pred_data %*% p_item[(3+ncol(pred_data)):length(p_item)])*prox_data)))

    return(traceline)

  }

#' Ordinal (graded response) tracelines.
#'
#' Computes cumulative probability tracelines for the graded response model
#' (Samejima, 1969). For a J-category item, computes J-1 boundary curves:
#'   P(Y >= j | theta, covs) = 1 / (1 + exp(-((c0_1 - d_j + c1*X) + (a0 + a1*X)*theta)))
#' where d_j are threshold parameters (d_1 = 0 by definition).
#'
#' Category probabilities are derived from differences of adjacent cumulative
#' probabilities in the E-step (not computed here).
#'
#' @param p_item Vector of item parameters:
#'   [c0_int1..c0_intJ-1, a0, c1_cov1..c1_covP, a1_cov1..a1_covP].
#'   c0 parameters include the base intercept and threshold offsets.
#' @param theta Vector of quadrature points for latent variable approximation.
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#' @param samp_size Sample size in dataset.
#' @param num_responses_item Number of response categories for this item.
#' @param num_quad Number of quadrature points used for approximating the
#' latent variable.
#'
#' @return a \code{"list"} of (J-1) matrices, each of dimension N x Q.
#'   Element k gives P(Y >= k+1 | theta_q, covs_i) for the k-th boundary.
#'
#' @keywords internal
#'
cumulative_traceline_pts <-
  function(p_item,
           theta,
           pred_data,
           samp_size,
           num_responses_item,
           num_quad) {

  # Allocate space for J-1 cumulative boundary curves.
  # Each boundary curve is an N x Q matrix.
  traceline <-
    lapply(1:(num_responses_item-1), function(x) {
              matrix(0,nrow=samp_size,ncol=num_quad)})

  # Identify parameter indices by name prefix.
  c0_parms <- grepl("c0",names(p_item),fixed=T)
  c1_parms <- grepl("c1",names(p_item),fixed=T)
  a0_parms <- grepl("a0",names(p_item),fixed=T)
  a1_parms <- grepl("a1",names(p_item),fixed=T)

  # First boundary: P(Y >= 2 | theta).
  # Uses the base intercept (c0[1]) without threshold offset.
  traceline[[1]] <-
    vapply(theta,
          function(x) {
            1 / (1 + exp(-((p_item[c0_parms][1] +
                              pred_data %*% p_item[c1_parms]) +
                             (p_item[a0_parms] +
                                pred_data %*% p_item[a1_parms])*x)
                         )
                 )
            }, numeric(samp_size))

  # Subsequent boundaries: P(Y >= j+1 | theta) for j = 2, ..., J-1.
  # Each successive boundary subtracts the threshold offset (c0[thr])
  # from the base intercept, shifting the curve rightward.
    for(thr in 2:(num_responses_item-1)) {
      traceline[[thr]] <-
        vapply(theta,
              function(x) {
                1 / (1 + exp(-((p_item[c0_parms][1] -
                                  p_item[c0_parms][thr] +
                                  pred_data %*% p_item[c1_parms]) +
                                 (p_item[a0_parms] +
                                    pred_data %*% p_item[a1_parms])*x)
                             )
                     )
                }, numeric(samp_size))
    }

  return(traceline)

}

#' Ordinal (graded response) tracelines using proxy data.
#'
#' Proxy-score version of \code{cumulative_traceline_pts}. Evaluates
#' cumulative boundary curves at observed proxy scores instead of
#' quadrature points.
#'
#' @param p_item Vector of item parameters (same layout as cumulative version).
#' @param prox_data Vector of observed proxy scores (length N).
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#' @param samp_size Sample size in dataset.
#' @param num_responses_item Number of response categories for this item.
#'
#' @return a \code{"list"} of (J-1) matrices (each N x 1): cumulative boundary
#'   probabilities evaluated at observed proxy scores.
#'
#' @keywords internal
#'
cumulative_traceline_pts_proxy <-
  function(p_item,
           prox_data,
           pred_data,
           samp_size,
           num_responses_item) {

    # Allocate space for J-1 cumulative boundary curves (N x 1 each).
    traceline <- lapply(1:(num_responses_item-1), function(x) {
                  matrix(0,nrow=samp_size,ncol=1)})

    # Identify parameter indices by name prefix.
    c0_parms <- grepl("c0",names(p_item),fixed=T)
    c1_parms <- grepl("c1",names(p_item),fixed=T)
    a0_parms <- grepl("a0",names(p_item),fixed=T)
    a1_parms <- grepl("a1",names(p_item),fixed=T)

    # First boundary: P(Y >= 2 | proxy_score).
    traceline[[1]] <- 1 / (1 + exp(-((p_item[c0_parms][1] +
                                 pred_data %*% p_item[c1_parms]) +
                                (p_item[a0_parms] +
                                   pred_data %*% p_item[a1_parms])*prox_data))
                           )

    # Subsequent boundaries with threshold offsets.
    for(thr in 2:(num_responses_item-1)) {
      traceline[[thr]] <- 1 / (1 + exp(-((p_item[c0_parms][1] -
                                   p_item[c0_parms][thr] +
                                   pred_data %*% p_item[c1_parms]) +
                                  (p_item[a0_parms] +
                                     pred_data %*% p_item[a1_parms])*prox_data))
                               )
    }

    return(traceline)

  }


#' Continuous (Gaussian / CFA) item tracelines.
#'
#' Computes the Gaussian density for continuous item responses conditional on
#' the latent trait. The model allows DIF on the mean (intercept and slope)
#' and on the residual variance:
#'   mu_i(theta) = (c0 + c1*X_i) + (a0 + a1*X_i)*theta
#'   sigma_i = sqrt(s0 * exp(s1*X_i))
#'   f(y_i | theta) = dnorm(y_i, mu_i(theta), sigma_i)
#'
#' The residual variance uses a log-linear model (exp link) to ensure
#' positivity, with s0 as the base variance and s1 as DIF on variance.
#'
#' @param p_item Vector of item parameters:
#'   [c0, a0, c1_1..c1_P, a1_1..a1_P, s0, s1_1..s1_P].
#' @param theta Vector of quadrature points for latent variable approximation.
#' @param responses_item Vector of observed continuous item responses (length N).
#' @param pred_data Matrix or data frame of DIF and/or impact predictors.
#' @param samp_size Sample size in data set.
#'
#' @return a \code{"matrix"} of density values for Gaussian item likelihood
#'   (N x Q). Each cell gives f(y_i | theta_q, covs_i).
#'
#' @keywords internal
#'
gaussian_traceline_pts <-
  function(p_item,
           theta,
           responses_item,
           pred_data,
           samp_size) {

  # Identify parameter indices by name prefix.
  c0_parms <- grepl("c0",names(p_item),fixed=T)
  c1_parms <- grepl("c1",names(p_item),fixed=T)
  a0_parms <- grepl("a0",names(p_item),fixed=T)
  a1_parms <- grepl("a1",names(p_item),fixed=T)
  s0_parms <- grepl("s0",names(p_item),fixed=T)
  s1_parms <- grepl("s1",names(p_item),fixed=T)

  # Compute conditional mean: mu_i(theta) = (c0 + c1*X) + (a0 + a1*X)*theta.
  # Result is an N x Q matrix.
  mu <-
    vapply(theta,
          function(x) {
            (p_item[c0_parms] +
               pred_data %*% p_item[c1_parms]) +
              (p_item[a0_parms] +
                 pred_data %*% p_item[a1_parms])*x
            }, numeric(samp_size))

  # Compute person-specific residual SD using log-linear model.
  # sigma_i = sqrt(s0 * exp(s1 * X_i)), ensuring positivity.
  sigma <-
    sqrt(p_item[s0_parms][1]*exp(pred_data %*% p_item[s1_parms]))

  # Evaluate Gaussian density for each person's response at each quadrature point.
  traceline <- t(sapply(1:samp_size,
                        function(x) {
                          dnorm(responses_item[x],mu[x,],sigma[x])
                          }
                        ))

  return(traceline)

}

#' Continuous (Gaussian / CFA) item tracelines using proxy data.
#'
#' Proxy-score version of \code{gaussian_traceline_pts}. Evaluates
#' the Gaussian density at observed proxy scores instead of quadrature points.
#'
#' @param p_item Vector of item parameters (same layout as Gaussian version).
#' @param prox_data Vector of observed proxy scores (length N).
#' @param responses_item Vector of observed continuous item responses (length N).
#' @param pred_data Matrix or data frame of DIF and/or impact predictors.
#' @param samp_size Sample size in data set.
#'
#' @return a \code{"matrix"} of density values for Gaussian item likelihood
#'   (N x 1), evaluated at each person's proxy score.
#'
#' @keywords internal
#'
gaussian_traceline_pts_proxy <-
  function(p_item,
           prox_data,
           responses_item,
           pred_data,
           samp_size) {

    c0_parms <- grepl("c0",names(p_item),fixed=T)
    c1_parms <- grepl("c1",names(p_item),fixed=T)
    a0_parms <- grepl("a0",names(p_item),fixed=T)
    a1_parms <- grepl("a1",names(p_item),fixed=T)
    s0_parms <- grepl("s0",names(p_item),fixed=T)
    s1_parms <- grepl("s1",names(p_item),fixed=T)

    mu <- (p_item[c0_parms] +
                  pred_data %*% p_item[c1_parms]) +
                 (p_item[a0_parms] +
                    pred_data %*% p_item[a1_parms])*prox_data
    sigma <-
      sqrt(p_item[s0_parms][1]*exp(pred_data %*% p_item[s1_parms]))

    traceline <- t(sapply(1:samp_size,
                          function(x) {
                            dnorm(responses_item[x],mu[x,],sigma[x])
                          }
    ))

    return(traceline)

  }

