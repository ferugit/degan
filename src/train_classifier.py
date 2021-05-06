import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from classifier.arguments import parse_arguments
import classifier.loader as loader
import classifier.models as models
import classifier.train as train
from classifier.reporter import Reporter

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
    model_id = args.arc
    experiment_path = os.path.join(args.checkpoint, args.arc)
    check_path(args.checkpoint)
    check_path(experiment_path)

    # Set Report
    reporter = Reporter(experiment_path, model_id  + '_report.json')
    reporter.report('arguments', vars(args))
    reporter.report('experiment_date', current_time)

    # Augments
    augments = ['white_noise']
    augments = {key: False for key in augments}
    if (args.white_noise):
        augments['white_noise'] = True

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
        fs=args.sampling_rate,
        augments=augments
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

    # Build model
    model = models.SerializableModule().create(args.arc, train_dataset.get_number_of_classes())
    model.to(device)

    # Get model summary
    model_summary = train.get_model_summary(model, (24000,), 'cnn')
    reporter.report('model_summary', model_summary)

    lr = args.lr
    epochs = args.epochs

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif(args.optimizer.lower() == 'rmsprop'):
        optimizer = torch.optim.RMSprop(model.parameters(),lr=lr, weight_decay=args.weight_decay)
    elif('sgd' in args.optimizer.lower()):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Model path
    model_path = os.path.join(experiment_path, model_id)

    # Train model
    model, metrics = train.train(
        model,
        train_loader,
        validation_loader,
        optimizer,
        criterion,
        device,
        epochs,
        batch_size,
        model_path,
        patience=args.patience,
        net_class='cnn'
        )

    # Save plot and metrics
    reporter.report('train_metrics', metrics)
    figure_path = os.path.join(experiment_path,  'train_metrics.png')
    train.save_train_metrics_plot(metrics, figure_path)

if __name__ == "__main__":
    main()