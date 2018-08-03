
# logistic regression decomposition
#' @param x model matrix or data.frame
#' @param y target
#' @param w weights (count of observations)
#' @param value value of each success in y
#' @param coefficients logistic regression coefficients
#' @param interval interval for evaluating incremental probability
#'   scaled to target variable).
#' @return A data.frame with "ID": row number from x, "V":
#'   feature number from x, "Variable": feature name from x, "feature": feature
#'   value from x, "p_y": attributed probability scaled to target y value,
#'   "p_yhat": attributed probability scaled to predicted value, "value_y":
#'   p_y * value and "w": number of occurrences of each observation.
#' @examples
#' library(titanic)
#' library(dplyr)
#' y = titanic_train[["Survived"]]
#' x = titanic_train %>%
#'   transmute(intercept = 1,
#'          young = ifelse(Age >= 20 | is.na(Age), 0, 1),
#'          female = Sex == "female")
#' model = glm.fit(x, y, w = rep(1, length(y)), family = binomial(link = "logit"), intercept = FALSE)
#' decomp_logistic(x, y, coefficients = model$coefficients)
decomp_logistic = function(x, y, w = rep(1, nrow(x)),
                           value = y,
                           coefficients,
                           interval = rep(1, length(coefficients))){

  stopifnot(ncol(x) == length(coefficients))
  stopifnot(ncol(x) == length(interval))
  stopifnot(nrow(x) == length(w))

  coefficients[is.na(coefficients)] = 0

  interval = unname(ifelse(coefficients < 0, floor(interval) + 0.5, interval))

  X = as.data.frame(x)
  feat_names = names(X)
  names(X) = paste0("V", seq_len(ncol(X)))

  base_ind = which(sapply(X, function(x) all(x == 1)))[1]
  if (is.na(base_ind)) stop("intercept column required")

  X = X %>%
    mutate(ID = row_number()) %>%
    tidyr::gather(V, feature, -ID) %>%
    mutate(V = factor(V, names(X)))

  table = data.frame(V = factor(levels(X$V), levels(X$V)),
                     Variable = feat_names,
                     Coefficient = unname(coefficients),
                     interval = interval,
                     stringsAsFactors = FALSE) %>%
    mutate(interval = ifelse(V == names(base_ind), Inf, interval))

  X = X %>%
    left_join(table, "V") %>%
    mutate(Value = feature * Coefficient)

  decomp = X %>%
    arrange(ID, desc(interval), Coefficient, V) %>%
    group_by(ID) %>%
    mutate(z = cumsum(Value),
           yhat = 1 / (1 + exp(-sum(Value)))) %>%
    group_by(ID, interval) %>%
    mutate(z_start = z[1] - Value[1],
           z_end = z[n()],
           z_diff = z_end - z_start) %>%
    ungroup %>%
    mutate(p_diff = 1 / (1 + exp(-z_end)) - 1 / (1 + exp(-z_start)),
           p_diff = ifelse(interval == Inf, p_diff + 0.5, p_diff)) %>%
    group_by(ID, interval) %>%
    mutate(p_attr = ifelse(p_diff == 0, 0, p_diff * Value / sum(Value))) %>%
    group_by(ID) %>%
    mutate(y_attr = sum(p_attr)) %>%
    ungroup %>%
    left_join(data_frame(ID = seq_len(nrow(x)),
                         y = y,
                         w = w,
                         value = value), "ID") %>%
    ungroup %>%
    mutate(p_y = p_attr * y / yhat,
           p_yhat = p_attr,
           value_y = p_y * value)

  decomp = decomp[c("ID", "V", "Variable", "feature",
                    "p_y", "p_yhat", "value_y", "w")]

  stopifnot(abs(sum(decomp$p_y) - sum(y)) < 1e-6)
  stopifnot(abs(sum(value * w) - sum(decomp$value_y * decomp$w)) < 1e-6)

  decomp

}

