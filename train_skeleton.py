import jittor as jt
import numpy as np
import os
import argparse
import time
import random

from jittor import nn
from jittor import optim

from dataset.dataset import get_dataloader, transform
from dataset.sampler import SamplerMix
from dataset.exporter import Exporter
from dataset.format import num_joints
from models.skeleton import create_model

from models.metrics import J2J


def j2j_batch(joints_a: jt.Var, joints_b: jt.Var) -> jt.Var:
    """
    joints_a: (B, J1, 3)
    joints_b: (B, J2, 3)

    Returns:
        scalar jt.Var
    """
    B, J1, _ = joints_a.shape
    _, J2, _ = joints_b.shape

    # dis1: (B, J1, J2)
    diff1 = joints_a.unsqueeze(2) - joints_b.unsqueeze(1)   # (B, J1, 1, 3) - (B, 1, J2, 3) -> (B, J1, J2, 3)
    dis1 = jt.norm(diff1, dim=-1)                           # (B, J1, J2)
    loss1 = dis1.min(dim=2)[0]                              # (B, J1)
    
    # dis2: (B, J2, J1)
    diff2 = joints_b.unsqueeze(2) - joints_a.unsqueeze(1)   # (B, J2, 1, 3) - (B, 1, J1, 3) -> (B, J2, J1, 3)
    dis2 = jt.norm(diff2, dim=-1)                           # (B, J2, J1)
    loss2 = dis2.min(dim=2)[0]                              # (B, J2)

    loss = (loss1.mean() + loss2.mean()) / 2   # (B,)
    return loss                                     # scalar


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
        model_type=args.model_type,
        output_channels=num_joints*3,
    )
    
    sampler = SamplerMix(num_samples=1024, vertex_samples=512)
    
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
    
    scheduler = jt.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    
    # Create loss function
    criterion = nn.MSELoss()
    
    # Create dataloaders
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=sampler,
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
            sampler=sampler,
            transform=transform,
            random_pose=True,
        )
    else:
        val_loader = None
    
    # Training loop
    best_loss = 99999999
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            # Get data and labels
            vertices, joints = data['vertices'], data['joints']
            
            vertices = vertices.permute(0, 2, 1)  # [B, 3, N]

            outputs = model(vertices,None)
            loss = criterion(outputs.reshape(outputs.shape[0], -1), joints.reshape(outputs.shape[0], -1)) \
                + j2j_batch(outputs,joints)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            # Calculate statistics
            train_loss += loss.item()
            
            # Print progress
            if jt.rank == 0:
                if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                    log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                            f"Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        if jt.rank == 0:
            # Calculate epoch statistics
            train_loss /= len(train_loader)
            epoch_time = time.time() - start_time
            
            log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Train Loss: {train_loss:.4f} "
                    f"Time: {epoch_time:.2f}s "
                    f"LR: {optimizer.lr:.6f}")

            # Validation phase
            if val_loader is not None and (epoch + 1) % args.val_freq == 0:
                model.eval()
                val_loss = 0.0
                J2J_loss = 0.0
                
                show_id = np.random.randint(0, len(val_loader))
                for batch_idx, data in enumerate(val_loader):
                    # Get data and labels
                    vertices, joints = data['vertices'], data['joints']
                    joints = joints.reshape(joints.shape[0], -1)
                    
                    # Reshape input if needed
                    if vertices.ndim == 3:  # [B, N, 3]
                        vertices = vertices.permute(0, 2, 1)  # [B, 3, N]
                    
                    # Forward pass
                    outputs = model(vertices,None)
                    loss = criterion(outputs.reshape(outputs.shape[0], -1), joints.reshape(outputs.shape[0], -1))
                    
                    # export render results
                    if batch_idx == show_id:
                        exporter = Exporter()
                        # export every joint's corresponding skinning
                        from dataset.format import parents
                        exporter._render_skeleton(path=f"tmp/skeleton/epoch_{epoch}/skeleton_ref.png", joints=joints[0].numpy().reshape(-1, 3), parents=parents)
                        exporter._render_skeleton(path=f"tmp/skeleton/epoch_{epoch}/skeleton_pred.png", joints=outputs[0].numpy().reshape(-1, 3), parents=parents)
                        exporter._render_pc(path=f"tmp/skeleton/epoch_{epoch}/vertices.png", vertices=vertices[0].permute(1, 0).numpy())

                    val_loss += loss.item()
                    for i in range(outputs.shape[0]):
                        J2J_loss += J2J(outputs[i].reshape(-1, 3), joints[i].reshape(-1, 3)).item() / outputs.shape[0]
                
                # Calculate validation statistics
                val_loss /= len(val_loader)
                J2J_loss /= len(val_loader)
                
                log_message(f"Validation Loss: {val_loss:.4f} J2J Loss: {J2J_loss:.4f}")
                
                # Save best model
                if J2J_loss < best_loss:
                    best_loss = J2J_loss
                    model_path = os.path.join(args.output_dir, 'best_model.pkl')
                    model.save(model_path)
                    log_message(f"Saved best model with loss {best_loss:.4f} to {model_path}")
        
        '''# Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
            model.save(checkpoint_path)
            log_message(f"Saved checkpoint to {checkpoint_path}")'''
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
    model.save(final_model_path)
    log_message(f"Training completed. Saved final model to {final_model_path}")
    
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
    parser.add_argument('--model_name', type=str, default='skeleton',
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
    parser.add_argument('--output_dir', type=str, default='output/skeleton_new',
                        help='Directory to save output files')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=100,
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