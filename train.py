import torch
import argparse
import os

import torch.nn as nn
import torch.optim as optim

from loguru import logger
from datasets.mnist import mnist, fashion_mnist
from utils import train, test, build_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running parameters")
    parser.add_argument('--dataset', default='mnist', type=str, required=True, help='Dataset to train', 
        choices=[
            'mnist',
            'fashion_mnist',
        ]
    ) 
    parser.add_argument('--model', default='Net', type=str, required=True, help='Model to load', 
        choices=[
            'Net',
            'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161',
            'EfficientNetB0',
            'GoogLeNet',
            'LeNet',
            'MobileNet', 'MobileNetV2',
            'RegNetX_200MF', 'RegNetX_400MF', 'RegNetY_400MF',
            'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
            'ShuffleNetG2', 'ShuffleNetG3', 'ShuffleNetV2',
            'VGG11', 'VGG13', 'VGG16', 'VGG19', 'VGG_mnist',
        ]
    )
    parser.add_argument('--epochs', default=2, type=int, required=True, help='total number of epochs')
    parser.add_argument('--lr', default=0.001, type=float, required=False, help='learning rate for model trainig')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='batch size for dataloader')
    parser.add_argument('--num_workers', default=2, type=int, required=False, help='total number of workers')
    parser.add_argument('--net_size', default=2, type=float, required=False, help='Net size for shufflenetv2')
    parser.add_argument('--save_model', action='store_true', help='Save best model checkpoint')
    parser.add_argument('--resume', action='store_true', help='Resume best model checkpoint training')
    args = parser.parse_args()

    start_epoch = 0

    # Check if the checkpoints directory to save best model exists, if not, create it
    checkpoint_dir = './checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Check for gpu availability
    logger.info(f'Checking for device...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Device set to {device}')

    # Load dataset
    logger.info('Loading dataset...')
    input_channels, num_classes = 1, 10
    if args.dataset == 'mnist':
        train_loader, test_loader = mnist(batchsize=args.batch_size, numworkers=args.num_workers)
        logger.info('Dataset Loaded!')
    elif args.dataset == 'fashion_mnist':
        train_loader, test_loader = fashion_mnist(batchsize=args.batch_size)
        logger.info('Dataset Loaded!')
    else:
        logger.info('Dataset is not supported!')

    # Define model
    logger.info('Building model...')
    net = build_model(model_name=args.model, input_channels=input_channels, num_classes=num_classes, net_size=args.net_size)
    logger.info(f'{net}')
    logger.info('Model loaded successfully!')

    # Resume training from saved checkpoints
    if args.resume:
        logger.info(f'Loading {args.model} checkpoint..')
        assert os.path.isdir(checkpoint_dir), 'No checkpoint directory found!'
        ckpt = torch.load(os.path.join(checkpoint_dir, f'{args.model}_best_model.pth'))
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch']

    # Move the model to the specified device
    model = net.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Start training
    logger.info('Training started!')
    train(model=model, device=device, train_loader=train_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, 
        start_epoch=start_epoch, total_epochs=args.epochs, model_name=args.model, save_model=args.save_model)
    logger.info('Finished Training!')

    # Test model
    logger.info('Testing model!')
    test(model, device, test_loader, criterion)
    logger.info("All done!")
