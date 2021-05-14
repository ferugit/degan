import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from degan.arguments import parse_arguments
import degan.loader as loader
import degan.models as models
import degan.train as train
from degan.reporter import Reporter

import datetime
import os

def check_path(folder_name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

def main():

    args = parse_arguments()

    # Model and experiment identification
    current_time = datetime.datetime.now()
    current_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    model_id = 'degan'
    experiment_path = os.path.join(args.checkpoint, args.arc)
    check_path(args.checkpoint)
    check_path(experiment_path)

    # Set Report
    reporter = Reporter(experiment_path, model_id  + '_report.json')
    reporter.report('arguments', vars(args))
    reporter.report('experiment_date', current_time)

    # Set seed
    train.set_seed(args.seed)

    # Select device
    if (torch.cuda.is_available() and args.cuda):
        device = torch.device("cuda")
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    # Read train and dev sets
    train_dataset, validation_dataset = loader.load_train_partitions(
        args.partition_path,
        window_size=int(args.time_window*args.sampling_rate),
        fs=args.sampling_rate
    )
    print("Train data information")
    print(train_dataset)

    # Generate DataLoaders for training
    sampler = None
    shuffle_train = True # Indicated in the dataset

    # Set batch size
    batch_size = args.batch_size
    
    # Generate data loaders
    train_loader = DataLoader(
        train_dataset, 
        shuffle=shuffle_train, 
        batch_size=batch_size, 
        drop_last=False, 
        sampler=sampler
        )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=True,
        batch_size=batch_size,
        drop_last=False
        )

    # Build models
    generator = models.SerializableModule().create('generator')
    discriminator = models.SerializableModule().create('discriminator')
    generator.to(device)
    discriminator.to(device)

    # Get models summary
    generator_summary = train.get_model_summary(generator, (100,), 'cnn')
    discriminator_summary = train.get_model_summary(discriminator, (16317,), 'cnn')
    reporter.report('generator_summary', generator_summary)
    reporter.report('discriminator_summary', discriminator_summary)

    epochs = args.epochs

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Model path
    model_path = os.path.join(experiment_path, model_id)

    # Train model
    metrics = train.train(
        [generator, discriminator],
        train_loader,
        validation_loader,
        [g_optimizer, d_optimizer],
        criterion,
        device,
        epochs,
        batch_size,
        model_path,
        patience=args.patience,
        critic_iters=args.critic_iters,
        lmbda=args.lmbda,
        net_class='cnn'
        )

    # Save plot and metrics
    reporter.report('train_metrics', metrics)
    #figure_path = os.path.join(experiment_path,  'train_metrics.png')
    #train.save_train_metrics_plot(metrics, figure_path)

if __name__ == "__main__":
    main()