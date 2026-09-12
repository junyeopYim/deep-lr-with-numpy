def my_affine(X, W, b):
    output = np.zeros((X.shape[0], W.shape[1]))
    for row in range(X.shape[0]):
        for target in range(W.shape[1]):
            output[row, target] = b[target]
            for feature in range(X.shape[1]):
                output[row, target] += X[row, feature] * W[feature, target]
    return output


def my_mse_backward(residual):
    gradient = np.zeros_like(residual, dtype=float)
    denominator = residual.size
    for index in np.ndindex(residual.shape):
        gradient[index] = residual[index] / denominator
    return gradient
