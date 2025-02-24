from os import path
import torch
import torch.utils.tensorboard as tb

import tempfile
log_dir = tempfile.mkdtemp()

def test_logging(train_logger, valid_logger):

    for epoch in range(10):
        total_train_accuracy = 0.0
        total_valid_accuracy = 0.0

        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            train_logger.add_scalar('loss', dummy_train_loss, epoch*20+iteration)
            dummy_train_accuracy = epoch/10.+torch.randn(10)
            total_train_accuracy += dummy_train_accuracy.mean().item()
        train_logger.add_scalar('accuracy', total_train_accuracy/20, epoch*20+19)

        torch.manual_seed(epoch)
        for iteration in range(10):
            dummy_validation_accuracy = epoch/10.+torch.randn(10)
            total_valid_accuracy += dummy_validation_accuracy.mean().item()
        valid_logger.add_scalar('accuracy', total_valid_accuracy/10, epoch*20+19)
        


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
