import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from classifier.arguments import parse_arguments
import classifier.loader as loader
import classifier.models as models
import classifier.train as train
from classifier.reporter import Reporter
import utils.metrics as analyzer

import datetime
import json
import os

def check_path(folder_name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

def main():

    args = parse_arguments()

    # Model and experiment identification
    model_id = args.arc
    experiment_path = os.path.join(args.checkpoint, args.arc)
    model_path = os.path.join(experiment_path, model_id + '_entire.pt')
    print("Testing model: " + args.arc + "\n")

    # Reporter
    reporter = Reporter(experiment_path, model_id + '_report.json')
    reporter.load(os.path.join(experiment_path, model_id + '_report.json'))

    # Augments
    augments = ['white_noise']
    augments = {key: False for key in augments}

    # Set seed
    train.set_seed(args.seed)

    # Read test df
    _, test_dataset = loader.load_train_partitions(
        args.partition_path,
        window_size=int(args.time_window*args.sampling_rate),
        fs=args.sampling_rate,
        augments=augments
    )

    print("Test data information")
    print(test_dataset)

    # Generate data loaders
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        drop_last=False
    )

    # Build model
    model = torch.load(model_path)

    if("sgru" in args.arc):
        net_class = 'rnn'
    else:
        net_class = 'cnn'

    # Select device
    if (torch.cuda.is_available() and args.cuda):
        device = torch.device("cuda")
        model.to(device)
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Test model
    labels, predictions, metrics = train.test_model(
        model,
        test_loader,
        criterion,
        device,
        args.batch_size,
        net_class=net_class
    )

    # Save confussión matrix
    target_names = []
    dict_path = os.path.join(args.partition_path, 'classes_index.json')
    classes_index = json.load(open(dict_path, 'r'))
    for index in range(len(predictions[0])):
        target_names.append(classes_index[str(index)])
    
    result = analyzer.get_metrics(labels, predictions, target_names=target_names)
    analyzer.plot_confusion_matrix(result[1], target_names, os.path.join(experiment_path, model_id + '_confusion.png'), normalize=False)

    # Store metrics
    test_metrics = {
        'loss':metrics['test_loss'],
        'accuracy':metrics['test_accuracy'],
        'report': result[0]
    }
    reporter.report('test_metrics', test_metrics)

if __name__ == "__main__":
    main()