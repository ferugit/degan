import matplotlib.pyplot as plt
import numpy as np
import random
import time

from torchinfo import summary
from utils import utils
import torch


def right_predictions(out, label):
    """
    From a given set of labes and predictions 
    it uses a softmax function to determine the
    output label
    """
    predictions_index = out.data.cpu().max(1, keepdim=True)[1]
    counter = predictions_index.eq(label.view_as(predictions_index)).sum().item()
    return counter


def train(models, train_loader, validation_loader, optimizers, criterion, device, 
    epochs, batch_size, model_path, patience=10, critic_iters=5, lmbda=10.0, net_class='rnn'):
    """
    Trainer for wuw detection models
    """
    # Models
    generator = models[0]
    discriminator = models[1]

    # Optimizers
    g_optimizer = optimizers[0]
    d_optimizer = optimizers[1]

    # Metrics
    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []
    epoch_times = []
    
    # Early stopping
    best_loss_validation = np.inf
    patience_counter = patience

    # Print model information
    print(generator)
    print(discriminator)

    # Get trainable parameters
    g_trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    d_trainable_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print('Number of trainable parameters of the generator: ' + str(g_trainable_params))
    print('Number of trainable parameters of the discriminator: ' + str(d_trainable_params))

    # For present intermediate information
    n_intermediate_steps = int(len(train_loader)/3)
    
    # Start training loop
    print('Starting trainig...')
    for epoch in range(1, epochs+1):
        start_time = time.process_time()
        
        # Train model
        train_loss = 0.0
        train_accuracy = 0.0
        counter = 0


        one = torch.tensor(1, dtype=torch.float)
        n_one = one * -1
        one.to(device)
        n_one.to(device)

        generator.train()
        discriminator.train()

        for _, x in train_loader:
            counter =+ 1

            #############################
            # (1) Train Discriminator
            #############################

            for lap in range(critic_iters):
                discriminator.zero_grad()

                # Get latent data:
                noise = torch.Tensor(batch_size, 100).uniform_(-1, 1)
                noise.to(device)

                # 1. Compute loss contribution from real training data
                d_real = discriminator(x)
                d_real = d_real.mean()
                d_real.backward(n_one)

                # 2. Compute loss contribution from generated data, then backprop
                fake = torch.autograd.Variable(generator(noise).data)
                d_fake = discriminator(fake)
                d_fake = d_fake.mean()
                d_fake.backward(one)

                # 3. Compute gradient penalty and backprop
                gradient_penalty = utils.calc_gradient_penalty(
                    discriminator,
                    x.data,
                    fake.data,
                    batch_size,
                    lmbda,
                    device
                    )
                gradient_penalty.backward(one)

                # Compute cost * Wassertein loss..
                d_cost_train = d_fake - d_real + gradient_penalty
                d_wass_train = d_real - d_fake

                # Update gradient of discriminator.
                d_optimizer.step()


            #############################
            # (3) Train Generator
            #############################

            generator.zero_grad()

            # Noise
            noise = torch.Tensor(batch_size, 100).uniform_(-1, 1)
            noise.to(device)

            fake = generator(noise)
            G = discriminator(fake)
            G = G.mean()

            # Update gradients.
            G.backward(n_one)
            g_cost =- G

            g_optimizer.step()


             # Present intermediate results
            if (counter%n_intermediate_steps == 0):
                print("Epoch {}......Step: {}/{}....... Discriminator cost: {} | Discriminator wass: {} | Generator cost: {}".format(
                    epoch,
                    counter,
                    d_cost_train,
                    d_wass_train,
                    g_cost
                    ))

    generator.save_entire_model(model_path + '_generator')
    generator.save(model_path + '_generator')

    discriminator.save_entire_model(model_path + '_discriminator')
    discriminator.save(model_path + '_discriminator')

        """
        model.train()
        for _, x, target in train_loader:
            counter += 1
            model.zero_grad()

            # Model forward
            if(net_class == 'rnn'):
                h = model.init_hidden(x.shape[0], device) # Memory reset
                h = h.data
                out, h = model(x.to(device).float(), h)
            elif(net_class == 'cnn'):
                out = model(x.to(device).float())

            # Backward and optimization
            loss = criterion(out, target.to(device))
            loss.backward()
            optimizer.step()

            # Store metrics
            train_loss += loss.item()
            train_accuracy += right_predictions(torch.nn.functional.softmax(out, 1), target)

            # Present intermediate results
            if (counter%n_intermediate_steps == 0):
                print("Epoch {}......Step: {}/{}....... Average Loss for Step: {} | Acurracy: {}".format(
                    epoch,
                    counter,
                    len(train_loader),
                    round(train_loss/counter, 4),
                    round(train_accuracy/(counter*batch_size), 4)
                    ))

        # Validate model
        validation_loss = 0.0
        validation_accuracy = 0.0

        with torch.no_grad():
            model.eval()
            for _, x, target in validation_loader:

                # Model forward
                if(net_class == 'rnn'):
                    h = model.init_hidden(x.shape[0], device) # Memory reset
                    h = h.data
                    out, h = model(x.to(device).float(), h)
                elif(net_class == 'cnn'):
                    out = model(x.to(device).float())
                
                # Store metrics: loss
                loss = criterion(out, target.to(device))
                validation_loss += loss.item()
                validation_accuracy += right_predictions(torch.nn.functional.softmax(out, 1), target)
        
        # Calculate average losses
        train_loss = train_loss/len(train_loader)
        train_accuracy = train_accuracy/len(train_loader.sampler)
        validation_loss = validation_loss/len(validation_loader)
        validation_accuracy = validation_accuracy/len(validation_loader.sampler)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        # Print epoch information
        current_time = time.process_time()
        print("")
        print("Epoch {}/{} Done.".format(epoch, epochs))
        print("\t Tain Loss: {} |  Train Accuracy: {}".format(train_loss, train_accuracy))
        print("\t Validation Loss: {} | Validation Accuracy: {}".format(validation_loss, validation_accuracy))
        print("\t Time Elapsed for Epoch: {} seconds".format(str(current_time-start_time)))
        print("")
        epoch_times.append(current_time-start_time)

        # Early stopping
        if(best_loss_validation <= validation_loss):
            patience_counter += 1

            print('Validation loss did not improve {:.3f} vs {:.3f}. Patience {}/{}.'.format(
                validation_loss,
                best_loss_validation,
                patience_counter,
                patience
                ))
            print("")
            
            if(patience_counter == patience):
                print('Breaking train loop: out of patience')
                print("")
                break
        else:
            # Reinitialize patience counter and save model
            patience_counter = 0
            best_loss_validation = validation_loss
            model.save_entire_model(model_path)
            model.save(model_path)

    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    metrics = {}
    metrics['training_time_s'] = sum(epoch_times)
    metrics['number_of_epochs'] = len(epoch_times)
    metrics['train_loss'] = train_losses
    metrics['train_accuracy'] = train_accuracies
    metrics['validation_loss'] = validation_losses
    metrics['validation_accuracy'] = validation_accuracies

    """
    metrics = {}

    return metrics


def plot_train_metrics(metrics):
    """
    Plot loss and accuracy metrics generated 
    on a training
    """
    plt.figure(figsize=(16, 4))
    plt.subplot(121)
    plt.plot(metrics[0], label='training loss')
    plt.plot(metrics[1], label='validation loss')
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.subplot(122)
    plt.plot(metrics[2], label='training accuracy')
    plt.plot(metrics[3], label='validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()


def save_train_metrics_plot(metrics, figure_path):
    """
    Plot loss and accuracy metrics generated 
    on a training
    """
    plt.figure(figsize=(16, 4))
    plt.subplot(121)
    plt.plot(metrics['train_loss'], label='training loss')
    plt.plot(metrics['validation_loss'], label='validation loss')
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.subplot(122)
    plt.plot(metrics['train_accuracy'], label='training accuracy')
    plt.plot(metrics['validation_accuracy'], label='validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig(figure_path)


def test_model(models, test_loader, criterion, device, batch_size,
    net_class='rnn'):
    
    # Metrics initialization
    test_loss = 0.0
    test_accuracy = 0.0

    labels = []
    predictions = []

    # For present intermediate information
    n_intermediate_steps = int(len(test_loader)/5)
    counter = 0

    with torch.no_grad():

        model.eval()
        
        for ID, x, target in test_loader:
            counter += 1

            # Model forward
            if(net_class == 'rnn'):
                h = model.init_hidden(x.shape[0], device) # Memory reset
                h = h.data
                out, h = model(x.to(device).float(), h)
            elif(net_class == 'cnn'):
                out = model(x.to(device).float())

            # Store metrics: loss and accuracy
            loss = criterion(out, target.to(device))
            test_loss += loss.item()
            test_accuracy += right_predictions(torch.nn.functional.softmax(out, 1), target)

            # Present intermediate results
            if (counter%n_intermediate_steps == 0):
                print("Epoch {}......Step: {}/{}....... Average Loss for Step: {} | Accuracy: {}".format(
                    1,
                    counter,
                    len(test_loader),
                    round(test_loss/counter, 4),
                    round(test_accuracy/(counter*batch_size), 4)
                    ))

            # Store labels and predictions
            labels += target.squeeze().tolist()
            predictions += torch.nn.functional.softmax(out.squeeze(), 1).tolist()

        # Calculate average losses ans accuracies
        test_loss = test_loss/len(test_loader)
        test_accuracy = test_accuracy/len(test_loader.sampler)

        metrics = {}
        metrics['test_loss'] = test_loss
        metrics['test_accuracy'] = test_accuracy

    return labels, predictions, metrics


def set_seed(seed):
    """
    Fix seed of torch, numpy and random.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    random.seed(seed)


def get_forward_time(model, device):
    """
    Returns CPU forward time in seconds
    """
    model.to('cpu')
    dummy_data = torch.rand(1, 24000)
    start = time.time()
    _ = model(dummy_data)
    end = time.time()
    forward_time = end - start
    model.to(device)
    return forward_time


def get_model_summary(model, input_shape, net_class='cnn'):
    """
    Get model summary
    Args:
        model
        input_shape
        net_class
    Returns:
        n_params: number of total parameters
        ops: number of operations
        size: model size
    """
    # Generate random input
    input_shape = (1, ) + input_shape
    model_input = torch.randn(input_shape)

    # Model forward
    if (net_class == 'rnn'):
        hidden_size =  model.init_hidden(1, torch.device('cpu')).data.numpy().shape
        h = torch.randn(hidden_size)
        info = summary(model, input_data=[model_input, h], verbose=0)
    else:
        info = summary(model, input_data=[model_input], verbose=0)
    
    # Store metrics
    n_params = info.total_params
    ops = info.total_mult_adds
    size = round(info.to_bytes(info.total_input + info.total_output + info.total_params), 2)

    model_summary = {
        'number_of_parameters': n_params,
        'number_of_operations': ops,
        'model_size_MB' : size
    }

    return model_summary