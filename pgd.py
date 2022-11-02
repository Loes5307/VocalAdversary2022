########################################################################################
#
# Projected Gradient Descent (PGD) to iteratively perturb audio
# From https://adversarial-ml-tutorial.org/
#
# Author(s): Zhuoran Liu and Loes van Bemmel
########################################################################################


"""
The PGD
Model is the gender classification model
X is the audio to be perturbed
y is the true label of the audio
epsilon is the clipping value (we use 0.1)
alpha is the perturbation rate (we use 0.0005)
num_iter is the number of iterations the PGD will run for (we use 10 or 100)

Outputs the perturbation, that then has to be added to the original audio to obtain the Adversarial Example
"""
def pgd(model, X, y, epsilon, alpha, num_iter):
    delta = t.zeros_like(X, requires_grad=True)
    for tx in range(num_iter):
        output = model(X+delta)
        loss = F.nll_loss(output.squeeze(), y.long())
        loss.backward()
        delta.data = (delta + X.shape[0]*alpha*delta.grad.data).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()