import torch


def calc_gradient_penalty(discriminator, real_data, fake_data, batch_size, lmbda, device):

    # Compute interpolation factors
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    # Interpolate between real and fake data.
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    # Evaluate discriminator
    discriminator_interpolates = discriminator(interpolates)

    # Obtain gradients of the discriminator with respect to the inputs
    d_interpolates_size = discriminator_interpolates.size()
    gradients = torch.autograd.grad(
        outputs=discriminator_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates_size).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    # Compute MSE between 1.0 and the gradient of the norm penalty to make discriminator
    # to be a 1-Lipschitz function.
    gradient_penalty = lmbda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty