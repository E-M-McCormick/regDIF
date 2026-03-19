###############################################################################
## E-step (Expectation Step) for the Penalized EM Algorithm
##
## This file implements the expectation step of an EM algorithm used to
## estimate item response theory (IRT) models that allow for differential
## item functioning (DIF). The E-step computes the posterior distribution
## P(theta | Y, X) of the latent trait (theta) given observed item responses
## (Y) and person-level covariates (X).
##
## Two variants are provided:
##
##   Estep        - Full numerical integration using Gauss-Hermite quadrature.
##                  The latent trait is approximated over a discrete grid of Q
##                  quadrature points, producing an N x Q matrix of posterior
##                  weights ("etable") used in the subsequent M-step.
##
##   Estep_proxy  - A simplified version that replaces numerical integration
##                  with observed proxy scores (e.g., sum scores or external
##                  measures). Each person's latent trait is represented by a
##                  single observed value rather than a distribution over
##                  quadrature points, yielding an N x 1 etable.
##
## The posterior is formed via Bayes' rule:
##
##   P(theta_q | Y_i, X_i) ~ P(Y_i | theta_q, X_i) * P(theta_q | X_i)
##
## where:
##   - P(Y_i | theta_q, X_i) is the likelihood of person i's response pattern
##     at quadrature point q, computed as the product of item response
##     functions (trace lines / ICCs) across all items.
##   - P(theta_q | X_i) is the prior density from the impact model, a normal
##     distribution with mean and variance that depend on covariates X_i.
##
## After normalization, the posterior weights are used to:
##   1. Form the complete-data expected log-likelihood (Q-function) in the
##      M-step.
##   2. Compute EAP (Expected A Posteriori) person scores and their standard
##      deviations when requested.
##
###############################################################################

#' Expectation step.
#'
#' Computes the E-step of the EM algorithm for regularized DIF models. For each
#' person, the posterior distribution over the latent trait is computed by
#' combining the prior (from the impact model) with the likelihood of the
#' observed response pattern (from item trace lines) across a grid of
#' quadrature points. The resulting posterior weights are collected into the
#' \code{etable} matrix, which is passed to the M-step to form the Q-function.
#'
#' @param p List of parameters. Elements 1 through \code{num_items} contain
#'   item parameters; element \code{num_items+1} contains the mean impact
#'   coefficients (alpha); element \code{num_items+2} contains the log-variance
#'   impact coefficients (phi).
#' @param item_data Matrix or dataframe of item responses (N x J). Rows are
#'   persons, columns are items. May contain NA for missing responses.
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#' @param item_type Vector of character values indicating the item type
#'   (e.g., \code{"cfa"}, \code{"2pl"}, \code{"rasch"}, \code{"graded"}).
#' @param mean_predictors Possibly different matrix of predictors for the mean
#'   impact equation. May differ from \code{pred_data} if only a subset of
#'   covariates predict the latent mean.
#' @param var_predictors Possibly different matrix of predictors for the
#'   variance impact equation. May differ from \code{pred_data} if only a
#'   subset of covariates predict the latent variance.
#' @param theta Vector of fixed quadrature points (length Q). These are the
#'   discrete grid values at which the latent trait is evaluated.
#' @param samp_size Sample size in dataset (number of non-missing persons).
#' @param num_items Number of items in dataset (J).
#' @param num_responses Number of responses for each item. For binary items
#'   this is 2; for graded items it can be larger.
#' @param adaptive_quad Logical indicating whether to use adaptive quadrature.
#'   When TRUE, quadrature nodes are re-centered and re-scaled based on the
#'   current impact parameter estimates.
#' @param num_quad Number of quadrature points (Q) used for approximating the
#'   latent variable distribution.
#' @param get_eap Logical indicating whether to compute EAP scores. When TRUE,
#'   Expected A Posteriori point estimates and their standard deviations are
#'   returned.
#' @param NA_cases Logical vector indicating missing observations. Used to map
#'   EAP scores back to their correct position in the full dataset (including
#'   rows that were excluded due to missingness).
#'
#' @return a \code{"list"} of posterior values from the expectation step
#'   containing:
#'   \describe{
#'     \item{etable}{N x Q matrix of normalized posterior weights.}
#'     \item{eap_scores}{N_full x 1 matrix of EAP scores (NA for excluded
#'       persons), or NULL if \code{get_eap = FALSE}.}
#'     \item{eap_sd}{N_full x 1 matrix of posterior standard deviations for
#'       EAP scores, or NULL if \code{get_eap = FALSE}.}
#'     \item{theta}{Vector of quadrature points (possibly updated if adaptive
#'       quadrature was used).}
#'     \item{observed_ll}{Scalar observed-data log-likelihood, computed as the
#'       sum of log-marginal likelihoods across persons.}
#'   }
#'
#' @keywords internal
#'
Estep <-
  function(p,
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
           get_eap,
           NA_cases) {

    # --- Initialize storage ---
    # observed_ll accumulates the observed-data log-likelihood across persons.
    # itemtrace will hold the item characteristic curves (ICCs) / trace lines
    #   for each item -- each element is an N x Q matrix (or list of matrices
    #   for graded items with multiple thresholds).
    # etable is the N x Q matrix of posterior weights: after the E-step,
    #   etable[i, q] = P(theta_q | Y_i, X_i), the normalized posterior
    #   probability that person i has latent trait value theta_q.
    # eap_scores and eap_sd store the Expected A Posteriori estimates and
    #   their posterior standard deviations, indexed by position in the full
    #   dataset (including NA cases).
    observed_ll <- 0
    itemtrace <- rep(list(NA),num_items)
    etable <- matrix(0, nrow = samp_size, ncol = num_quad)
    eap_scores <- if(get_eap) matrix(NA, nrow = length(NA_cases), ncol = 1)
    eap_sd <- if(get_eap) matrix(NA, nrow = length(NA_cases), ncol = 1)

    # --- Impact model ---
    # The impact model specifies how the latent trait distribution varies
    # across persons as a function of covariates. This implements a
    # heteroskedastic normal model:
    #   theta_i ~ N(alpha_i, phi_i)
    # where:
    #   alpha_i = mean_predictors_i %*% beta_mean  (conditional mean)
    #   phi_i   = exp(var_predictors_i %*% beta_var) (conditional variance,
    #             exponentiated to ensure positivity)
    alpha <- mean_predictors %*% p[[num_items+1]]
    phi <- exp(var_predictors %*% p[[num_items+2]])

    # --- Adaptive quadrature (optional) ---
    # When adaptive quadrature is enabled, the fixed quadrature nodes are
    # replaced by nodes centered at the current population mean and scaled by
    # the current population standard deviation. This improves numerical
    # accuracy when the latent distribution is far from standard normal.
    # Gauss-Hermite quadrature nodes are designed for the weight function
    # exp(-x^2), so we apply the change of variable:
    #   theta_q = mu + sqrt(2 * sigma^2) * z_q
    # where z_q are the standard Gauss-Hermite nodes.
    if(adapt_quad == TRUE) {
      theta <- mean(alpha) +
        sqrt(2*mean(phi))*statmod::gauss.quad(n = num_quad,
                                              kind = "hermite")$nodes
    }

    # --- Compute item trace lines (ICCs) ---
    # For each item, compute the probability (or density) of each possible
    # response at each quadrature point for each person. The trace line
    # functions are item-type-specific:
    #   - "cfa": Continuous (Gaussian) responses -- returns density values.
    #   - "2pl"/"rasch": Binary (Bernoulli) responses -- returns P(Y=0|theta).
    #   - "graded": Ordered categorical responses -- returns cumulative
    #     probabilities at each threshold.
    # Each trace line function returns an N x Q matrix (or list of N x Q
    # matrices for graded items), where element [i, q] gives the trace line
    # value for person i at quadrature point q.
    for (item in 1:num_items) {
      if(item_type[item] == "cfa") {
        itemtrace[[item]] <- gaussian_traceline_pts(p[[item]],
                                                    theta,
                                                    item_data[,item],
                                                    pred_data,
                                                    samp_size)
      } else if (item_type[item] %in% c("2pl","rasch")) {                       # Add Rasch items here
        itemtrace[[item]] <- bernoulli_traceline_pts(p[[item]],
                                                     theta,
                                                     pred_data,
                                                     samp_size)
      } else {
        itemtrace[[item]] <- cumulative_traceline_pts(p[[item]],
                                                      theta,
                                                      pred_data,
                                                      samp_size,
                                                      num_responses[item],
                                                      num_quad)
      }
    }

    # --- Compute posterior weights (the core of the E-step) ---
    # For each person i, we build up the posterior P(theta_q | Y_i, X_i) by:
    #   1. Starting with the prior density from the impact model:
    #        P(theta_q | X_i) = dnorm(theta_q, mean=alpha_i, sd=sqrt(phi_i))
    #   2. Sequentially multiplying by the likelihood contribution from each
    #      item j (local independence assumption):
    #        P(Y_ij | theta_q, X_i)
    #   3. Normalizing so the posterior sums to 1 across quadrature points.
    #
    # The marginal likelihood for person i is the normalizing constant:
    #   P(Y_i | X_i) = sum_q [ P(Y_i | theta_q, X_i) * P(theta_q | X_i) ]
    # and the observed log-likelihood is the sum of log-marginals.
    for(i in 1:samp_size) {

      # Initialize the posterior with the prior density evaluated at each
      # quadrature point. This is the person-specific normal density from the
      # impact model, which serves as the "weight" for each theta value.
      posterior <- dnorm(theta,
                         mean = alpha[i],
                         sd = sqrt(phi[i]))

      # Multiply the prior by the likelihood contribution from each item.
      # Under local independence, the joint likelihood factors as a product
      # over items: P(Y_i | theta_q) = prod_j P(Y_ij | theta_q).
      # Missing responses (NA) are skipped, effectively marginalizing over
      # the missing data (i.e., they contribute a factor of 1).
      for(j in 1:num_items) {

        x <- item_data[i,j]
        if(is.na(x)) next

        if(item_type[j] == "cfa") { # Continuous responses.
          # For continuous items, the trace line directly gives the Gaussian
          # density f(Y_ij | theta_q), so we multiply it in.
          posterior <- posterior*itemtrace[[j]][i,]
        } else if(item_type[item] %in% c("2pl","rasch")) { # Binary responses.  # Add Rasch items here
          # For binary items, the trace line gives P(Y_ij=0 | theta_q).
          # When x=1 (endorsed), the probability is 1 - P(Y=0) = P(Y=1).
          # When x=0 (not endorsed), the probability is P(Y=0) directly.
          if(x == 1) {
            posterior <- posterior*(1-itemtrace[[j]][i,])
          } else {
            posterior <- posterior*itemtrace[[j]][i,]
          }
        } else if(item_type[j] == "graded") { # Ordered categorical responses.
          # For graded response items, the trace lines give cumulative
          # probabilities P(Y >= k | theta) at each threshold. The category
          # probability is obtained by differencing adjacent cumulative
          # probabilities:
          #   P(Y=k) = P(Y >= k) - P(Y >= k+1)
          # Special cases: lowest category uses 1 - P(Y >= 2), and highest
          # category uses P(Y >= K) directly.
          if(x == 1) {
            posterior <- posterior*(1-itemtrace[[j]][[1]][i,])
          } else if(x == num_responses[j]) {
            posterior <- posterior*itemtrace[[j]][[num_responses[j]-1]][i,]
          } else {
            posterior <- posterior*(itemtrace[[j]][[x-1]][i,]-
                                        itemtrace[[j]][[x]][i,])
          }
        }

      }

      # --- Normalize the posterior ---
      # The marginal likelihood P(Y_i | X_i) is the sum of the unnormalized
      # posterior across quadrature points. This serves as both the
      # normalizing constant and a contribution to the observed log-likelihood.
      # If the marginal is zero (numerical underflow), we set it to 1 to avoid
      # division by zero; this effectively gives a uniform posterior for that
      # person.
      marginal <- sum(posterior, na.rm = TRUE)
      observed_ll <- observed_ll + log(marginal)
      if(marginal == 0) marginal <- 1
      etable[i,] <- posterior/marginal

      # --- EAP (Expected A Posteriori) scores ---
      # When requested, compute the posterior mean and standard deviation of
      # theta for each person. The EAP score is the expected value of theta
      # under the posterior:
      #   EAP_i = sum_q [ P(theta_q | Y_i) * theta_q ]
      # The posterior SD measures the uncertainty in the point estimate:
      #   SD_i = sqrt( sum_q [ P(theta_q | Y_i) * (theta_q - EAP_i)^2 ] )
      # Note: we use the unnormalized posterior divided by the marginal
      # (equivalent to the normalized posterior) for the computation.
      if(get_eap) {
        eap_scores[which(!NA_cases)[i]] <- sum(posterior*theta)/marginal
        eap_sd[which(!NA_cases)[i]] <-
          sqrt(sum(posterior*(theta-eap_scores[which(!NA_cases)[i]])**2)/marginal)
      }

    }

    # Return the E-step results:
    #   etable       - N x Q posterior weight matrix for the M-step Q-function
    #   eap_scores   - person-level EAP theta estimates (if requested)
    #   eap_sd       - posterior SDs of the EAP estimates (if requested)
    #   theta        - quadrature nodes (possibly updated by adaptive quad)
    #   observed_ll  - observed-data log-likelihood for convergence monitoring
    return(list(etable=etable,eap_scores=eap_scores,eap_sd=eap_sd,
                theta=theta,observed_ll=observed_ll))


  }

###############################################################################
## Estep_proxy: Proxy-Based E-step
##
## This is a simplified variant of the E-step that substitutes observed proxy
## scores (e.g., standardized sum scores, external criterion measures) for the
## latent trait. Instead of integrating over a grid of Q quadrature points,
## each person's latent trait is represented by a single observed proxy value.
## This eliminates the need for numerical integration and produces an N x 1
## etable.
##
## The "posterior" in this case is not a full distribution but rather the
## density of the proxy score under the impact model times the likelihood of
## the response pattern at that single proxy value. Because there is only one
## "quadrature point" per person (the proxy score), no normalization step is
## needed -- the posterior reduces to a scalar weight.
##
## This approach trades statistical efficiency for computational speed and
## simplicity, and can be useful as an approximation or when a reliable
## external measure of the latent trait is available.
###############################################################################

#' Expectation step with proxy data.
#'
#' A simplified E-step that uses observed proxy scores in place of numerical
#' integration over quadrature points. Each person's latent trait is fixed at
#' their proxy value (e.g., a sum score or external measure), eliminating the
#' need for quadrature. The resulting etable is N x 1, with each entry
#' representing the un-normalized posterior weight for that person.
#'
#' @param p List of parameters. Elements 1 through \code{num_items} contain
#'   item parameters; element \code{num_items+1} contains the mean impact
#'   coefficients; element \code{num_items+2} contains the log-variance impact
#'   coefficients.
#' @param item_data Matrix or dataframe of item responses (N x J).
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#' @param item_type Vector of character values indicating the item type
#'   (e.g., \code{"cfa"}, \code{"2pl"}, \code{"rasch"}, \code{"graded"}).
#' @param mean_predictors Possibly different matrix of predictors for the mean
#'   impact equation.
#' @param var_predictors Possibly different matrix of predictors for the
#'   variance impact equation.
#' @param prox_data Vector of observed proxy scores (length N). These replace
#'   the quadrature grid -- each person's theta is fixed at their proxy value.
#' @param samp_size Sample size in dataset.
#' @param num_items Number of items in dataset.
#' @param num_responses Number of responses for each item.
#' @param get_eap Logical indicating whether to compute EAP scores. In the
#'   proxy case, the "EAP" is simply the proxy score weighted by the posterior.
#' @param NA_cases Logical vector indicating missing observations.
#'
#' @return a \code{"list"} of posterior values from the expectation step
#'   containing:
#'   \describe{
#'     \item{etable}{N x 1 matrix of posterior weights.}
#'     \item{eap_scores}{N_full x 1 matrix of weighted proxy scores, or NULL.}
#'     \item{eap_sd}{N_full x 1 matrix of posterior SDs, or NULL.}
#'     \item{observed_ll}{Scalar observed-data log-likelihood.}
#'   }
#'
#' @keywords internal
#'
Estep_proxy <-
  function(p,
           item_data,
           pred_data,
           item_type,
           mean_predictors,
           var_predictors,
           prox_data,
           samp_size,
           num_items,
           num_responses,
           get_eap,
           NA_cases) {

    # --- Initialize storage ---
    # Same structure as the full E-step, but etable is N x 1 because each
    # person has only a single "quadrature point" (their proxy score).
    observed_ll <- 0
    itemtrace <- rep(list(NA),num_items)
    etable <- matrix(0, nrow = samp_size, ncol = 1)
    eap_scores <- if(get_eap) matrix(NA, nrow = length(NA_cases), ncol = 1)
    eap_sd <- if(get_eap) matrix(NA, nrow = length(NA_cases), ncol = 1)

    # --- Impact model ---
    # Same conditional normal model as the full E-step. The impact parameters
    # define the person-specific prior for the latent trait, now evaluated at
    # the single proxy score rather than across a grid of quadrature points.
    alpha <- mean_predictors %*% p[[num_items+1]]
    phi <- exp(var_predictors %*% p[[num_items+2]])


    # --- Compute item trace lines at proxy scores ---
    # Unlike the full E-step which evaluates trace lines at Q quadrature
    # points, the proxy version evaluates them at each person's single proxy
    # score. The "_proxy" variants of the trace line functions handle this
    # by accepting the prox_data vector instead of a theta grid.
    for (item in 1:num_items) {
      if(item_type[item] == "cfa") {
        itemtrace[[item]] <- gaussian_traceline_pts_proxy(p[[item]],
                                                          prox_data,
                                                          item_data[,item],
                                                          pred_data,
                                                          samp_size)
      } else if (item_type[item] %in% c("2pl","rasch")) {
        itemtrace[[item]] <- bernoulli_traceline_pts_proxy(p[[item]],
                                                           prox_data,
                                                           pred_data)
      } else {
        itemtrace[[item]] <- cumulative_traceline_pts_proxy(p[[item]],
                                                      prox_data,
                                                      pred_data,
                                                      samp_size,
                                                      num_responses[item])
      }
    }

    # --- Compute posterior weights ---
    # The logic mirrors the full E-step, but with a key difference: the
    # posterior is a scalar per person (not a vector over quadrature points),
    # so no normalization is needed. The "posterior" here is:
    #   w_i = P(proxy_i | X_i) * prod_j P(Y_ij | proxy_i, X_i)
    for(i in 1:samp_size) {

      # Evaluate the prior density at the person's proxy score. This is the
      # density of the proxy under the impact model's normal distribution.
      posterior <- dnorm(prox_data[i],
                         mean = alpha[i],
                         sd = sqrt(phi[i]))

      # Multiply by the likelihood contribution from each item, evaluated at
      # the proxy score. Same local independence factorization as the full
      # E-step, but each trace line value is a scalar (not a vector).
      for(j in 1:num_items) {

        x <- item_data[i,j]
        if(is.na(x)) next

        if(item_type[j] == "cfa") { # Continuous responses.
          posterior <- posterior*itemtrace[[j]][i,]
        } else if(item_type[item] %in% c("2pl","rasch")) { # Binary responses.
          if(x == 1) {
            posterior <- posterior*(1-itemtrace[[j]][i,])
          } else {
            posterior <- posterior*itemtrace[[j]][i,]
          }
        } else if(item_type[j] == "graded") { # Ordered categorical responses.
          if(x == 1) {
            posterior <- posterior*(1-itemtrace[[j]][[1]][i,])
          } else if(x == num_responses[j]) {
            posterior <- posterior*itemtrace[[j]][[num_responses[j]-1]][i,]
          } else {
            posterior <- posterior*(itemtrace[[j]][[x-1]][i,]-
                                      itemtrace[[j]][[x]][i,])
          }
        }

      }

      # No normalization needed in the proxy case -- the posterior is a scalar
      # weight that will be used directly in the M-step Q-function.
      # The observed log-likelihood is accumulated as log(w_i).
      observed_ll <- observed_ll + log(posterior)
      etable[i,] <- posterior

      # --- EAP scores (proxy version) ---
      # In the proxy case, the "EAP" is simply the proxy score weighted by the
      # posterior. Because there is only one "quadrature point," the posterior
      # mean is just the proxy score itself (weighted). The posterior SD
      # reflects the discrepancy between the proxy and the weighted estimate.
      if(get_eap) {
        eap_scores[which(!NA_cases)[i]] <- posterior*prox_data[i]
        eap_sd[which(!NA_cases)[i]] <-
          sqrt(posterior*(prox_data[i]-eap_scores[which(!NA_cases)[i]])**2)
      }

    }

    # Return the proxy E-step results. Note: no theta is returned because the
    # proxy scores are used directly in place of quadrature points.
    return(list(etable=etable,eap_scores=eap_scores,eap_sd=eap_sd,observed_ll=observed_ll))


  }
