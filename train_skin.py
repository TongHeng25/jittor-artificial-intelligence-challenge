import os
import jittor as jt
import numpy as np

import argparse
import time
import random

from jittor import nn
from jittor import optim

from dataset.dataset import get_dataloader, transform
from dataset.format import id_to_name
from dataset.sampler import SamplerMix
from dataset.format import num_joints
from models.skin import create_model

from dataset.exporter import Exporter

# Set Jittor flags
jt.flags.use_cuda = 1

def train(args):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    
    def log_message(message):
        """Helper function to log messages to file and print to console"""
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    # Log training parameters
    log_message(f"Starting training with parameters: {args}")
    
    # Create model
    model = create_model(
        model_name=args.model_name,
        num_joints=num_joints,
    )
    
    # Load pre-trained model if specified
    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)
    
    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Create loss function
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    
    # Create dataloaders
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=SamplerMix(num_samples=1024, vertex_samples=512),
        transform=transform,
        random_pose=args.random_pose,
    )
    
    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root,
            data_list=args.val_data_list,
            train=False,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=SamplerMix(num_samples=1024, vertex_samples=512),
            transform=transform,
        )
    else:
        val_loader = None
    
    scheduler = jt.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    # Training loop
    best_loss = 99999999
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss_mse = 0.0
        train_loss_l1 = 0.0
        
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            # Get data and labels
            vertices, joints, skin = data['vertices'], data['joints'], data['skin']

            vertices: jt.Var
            joints: jt.Var
            skin: jt.Var
            outputs = model(vertices, joints)
            loss_mse = criterion_mse(outputs, skin)
            loss_l1 = criterion_l1(outputs, skin)
            loss = loss_mse + loss_l1
            
            # Backward pass and optimize
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            # Calculate statistics
            train_loss_mse += loss_mse.item()
            train_loss_l1 += loss_l1.item()
            
            # Print progress
            if jt.rank == 0:
                if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                    log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                            f"Loss mse: {loss_mse.item():.4f} Loss l1: {loss_l1.item():.4f}")
        if jt.rank == 0:
            # Calculate epoch statistics
            train_loss_mse /= len(train_loader)
            train_loss_l1 /= len(train_loader)
            epoch_time = time.time() - start_time
            
            log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Train Loss mse: {train_loss_mse:.4f} "
                    f"Train Loss l1: {train_loss_l1:.4f} "
                    f"Time: {epoch_time:.2f}s "
                    f"LR: {optimizer.lr:.6f}")

            # Validation phase
            if val_loader is not None and (epoch + 1) % args.val_freq == 0:
                model.eval()
                val_loss_mse = 0.0
                val_loss_l1 = 0.0
                
                show_id = np.random.randint(0, len(val_loader))
                for batch_idx, data in enumerate(val_loader):
                    # Get data and labels
                    vertices, joints, skin = data['vertices'], data['joints'], data['skin']
                    
                    # Forward pass
                    outputs = model(vertices, joints)
                    loss_mse = criterion_mse(outputs, skin)
                    loss_l1 = criterion_l1(outputs, skin)
                    
                    # export render results(which is slow, so you can turn it off)
                    if batch_idx == show_id:
                        exporter = Exporter()
                        for i in id_to_name:
                            name = id_to_name[i]
                            # export every joint's corresponding skinning
                            exporter._render_skin(path=f"tmp/skin/epoch_{epoch}/{name}_ref.png",vertices=vertices.numpy()[0], skin=skin.numpy()[0, :, i], joint=joints[0, i])
                            exporter._render_skin(path=f"tmp/skin/epoch_{epoch}/{name}_pred.png",vertices=vertices.numpy()[0], skin=outputs.numpy()[0, :, i], joint=joints[0, i])

                    val_loss_mse += loss_mse.item()
                    val_loss_l1 += loss_l1.item()
                
                # Calculate validation statistics
                val_loss_mse /= len(val_loader)
                val_loss_l1 /= len(val_loader)
                
                log_message(f"Validation Loss: mse: {val_loss_mse:.4f} l1: {val_loss_l1:.4f}")
                
                # Save best model
                if val_loss_l1 < best_loss:
                    best_loss = val_loss_l1
                    model_path = os.path.join(args.output_dir, 'best_model.pkl')
                    model.save(model_path)
                    log_message(f"Saved best model with l1 loss {best_loss:.4f} to {model_path}")

        scheduler.step()
        '''# Save checkpoint
            if (epoch + 1) % args.save_freq == 0:
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
                model.save(checkpoint_path)
                log_message(f"Saved checkpoint to {checkpoint_path}")'''
    
    '''# Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
    model.save(final_model_path)
    log_message(f"Training completed. Saved final model to {final_model_path}")'''
    
    return model, best_loss

def main():
    """Parse arguments and start training"""
    parser = argparse.ArgumentParser(description='Train a point cloud model')
    
    # Dataset parameters
    parser.add_argument('--train_data_list', type=str, default='data/data/train_list_aug.txt',
                        help='Path to the training data list file')
    parser.add_argument('--val_data_list', type=str, default='data/data/val_list_aug.txt',
                        help='Path to the validation data list file')
    parser.add_argument('--data_root', type=str, default='data/data',
                        help='Root directory for the data files')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='skin',
                        choices=['pct', 'pct2', 'custom_pct', 'skeleton'],
                        help='Model architecture to use')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'enhanced'],
                        help='Model type for skeleton model')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='Path to pretrained model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--random_pose', type=int, default=0,
                        help='Apply random pose to asset')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output/skin_new',
                        help='Directory to save output files')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save frequency')
    parser.add_argument('--val_freq', type=int, default=1,
                        help='Validation frequency')
    
    args = parser.parse_args()
    
    # Start training
    train(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()