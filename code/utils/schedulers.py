import torch
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
import math


class PolyLRWithWarmup(_LRScheduler):
    """
    Polynomial Learning Rate Scheduler with Warm-up.
    
    This scheduler implements a polynomial learning rate decay with an initial warm-up phase.
    During warm-up, the learning rate linearly increases from a small fraction to the base LR.
    After warm-up, the learning rate follows polynomial decay: lr = base_lr * (1 - progress)^power
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_iters (int): Number of warm-up iterations. Default: 1000
        total_iters (int): Total number of training iterations (including warm-up).
        power (float): Polynomial power for decay. Default: 0.9
        warmup_start_factor (float): Starting factor for warm-up (fraction of base LR). Default: 0.01
        last_epoch (int): The index of last epoch. Default: -1
        verbose (bool): If True, prints a message to stdout for each update. Default: False
    
    Example:
        >>> scheduler = PolyLRWithWarmup(optimizer, warmup_iters=1000, total_iters=10000, power=0.9)
        >>> for epoch in range(100):
        >>>     for batch in dataloader:
        >>>         train_batch(...)
        >>>         scheduler.step()
    """
    
    def __init__(self, optimizer, warmup_iters=1000, total_iters=10000, power=0.9, 
                 warmup_start_factor=0.01, last_epoch=-1, verbose=False):
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.power = power
        self.warmup_start_factor = warmup_start_factor
        
        if warmup_iters >= total_iters:
            raise ValueError(f"warmup_iters ({warmup_iters}) should be less than total_iters ({total_iters})")
        
        super(PolyLRWithWarmup, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current step."""
        current_iter = self.last_epoch
        lrs = []
        
        for base_lr in self.base_lrs:
            if current_iter < self.warmup_iters:
                # Warm-up phase: linear increase from warmup_start_factor to 1.0
                lr_scale = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * (current_iter / self.warmup_iters)
                lr = base_lr * lr_scale
            else:
                # Polynomial decay phase
                progress = (current_iter - self.warmup_iters) / (self.total_iters - self.warmup_iters)
                progress = min(progress, 1.0)  # Clamp to 1.0 to avoid negative LR
                lr_scale = (1.0 - progress) ** self.power
                lr = base_lr * lr_scale
            
            lrs.append(lr)
        
        return lrs


class CosineAnnealingLRWithWarmup(_LRScheduler):
    """
    Cosine Annealing Learning Rate Scheduler with Warm-up.
    
    This scheduler implements cosine annealing learning rate decay with an initial warm-up phase.
    During warm-up, the learning rate linearly increases from a small fraction to the base LR.
    After warm-up, the learning rate follows cosine annealing.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_iters (int): Number of warm-up iterations. Default: 1000
        total_iters (int): Total number of training iterations (including warm-up).
        eta_min (float): Minimum learning rate. Default: 0
        warmup_start_factor (float): Starting factor for warm-up (fraction of base LR). Default: 0.01
        last_epoch (int): The index of last epoch. Default: -1
        verbose (bool): If True, prints a message to stdout for each update. Default: False
    """
    
    def __init__(self, optimizer, warmup_iters=1000, total_iters=10000, eta_min=0,
                 warmup_start_factor=0.01, last_epoch=-1, verbose=False):
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.eta_min = eta_min
        self.warmup_start_factor = warmup_start_factor
        
        if warmup_iters >= total_iters:
            raise ValueError(f"warmup_iters ({warmup_iters}) should be less than total_iters ({total_iters})")
        
        super(CosineAnnealingLRWithWarmup, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current step."""
        current_iter = self.last_epoch
        lrs = []
        
        for base_lr in self.base_lrs:
            if current_iter < self.warmup_iters:
                # Warm-up phase: linear increase from warmup_start_factor to 1.0
                lr_scale = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * (current_iter / self.warmup_iters)
                lr = base_lr * lr_scale
            else:
                # Cosine annealing phase
                progress = (current_iter - self.warmup_iters) / (self.total_iters - self.warmup_iters)
                progress = min(progress, 1.0)  # Clamp to 1.0
                lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
            
            lrs.append(lr)
        
        return lrs


class FinetuneLRScheduler(_LRScheduler):
    """
    Custom scheduler for finetuning that handles different learning rates for different parameter groups.
    
    This scheduler uses a base scheduler (exponential, cosine, or warmup) for the decoder,
    and starts encoder training after a certain number of epochs with a separate decay schedule.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer with parameter groups.
        base_scheduler (str): Base scheduler for decoder. Options: 'exponential', 'cosine', 'poly_warmup'. Default: 'exponential'
        finetune_start_epoch (int): Epoch when encoder finetuning begins. Default: 10
        encoder_start_lr (float): Starting learning rate for encoder when finetuning begins. Default: 1e-6
        encoder_decay_factor (float): Factor to reduce encoder LR each epoch. Default: 0.95
        
        # Base scheduler parameters
        exponential_gamma (float): Gamma for exponential decay. Default: 0.95
        cosine_eta_min (float): Minimum LR for cosine annealing. Default: 1e-6
        cosine_T_max (int): Period for cosine annealing. Default: 50
        poly_warmup_iters (int): Warmup iterations for polynomial scheduler. Default: 1000
        poly_power (float): Power for polynomial decay. Default: 0.9
        poly_warmup_factor (float): Starting factor for warmup. Default: 0.01
        
        last_epoch (int): The index of last epoch. Default: -1
        verbose (bool): If True, prints a message to stdout for each update. Default: False
    """
    
    def __init__(self, optimizer, base_scheduler='exponential', finetune_start_epoch=10, 
                 encoder_start_lr=1e-6, encoder_decay_factor=0.95,
                 # Exponential scheduler params
                 exponential_gamma=0.95,
                 # Cosine scheduler params  
                 cosine_eta_min=1e-6, cosine_T_max=50,
                 # Polynomial warmup params
                 poly_warmup_iters=1000, poly_power=0.9, poly_warmup_factor=0.01,
                 last_epoch=-1, verbose=False):
        
        self.base_scheduler = base_scheduler
        self.finetune_start_epoch = finetune_start_epoch
        self.encoder_start_lr = encoder_start_lr
        self.encoder_decay_factor = encoder_decay_factor
        
        # Base scheduler parameters
        self.exponential_gamma = exponential_gamma
        self.cosine_eta_min = cosine_eta_min
        self.cosine_T_max = cosine_T_max
        self.poly_warmup_iters = poly_warmup_iters
        self.poly_power = poly_power
        self.poly_warmup_factor = poly_warmup_factor
        
        super().__init__(optimizer, last_epoch)
        
        # Create single scheduler after parent initialization
        self.scheduler = self._create_scheduler()
    
    def _get_decoder_lr(self, base_lr, current_epoch):
        """Calculate decoder learning rate using the base scheduler."""
        # Ensure base_lr is a float
        if isinstance(base_lr, (list, tuple)):
            base_lr = float(base_lr[0])
        else:
            base_lr = float(base_lr)
            
        if self.base_scheduler == 'exponential':
            return base_lr * (self.exponential_gamma ** current_epoch)
        
        elif self.base_scheduler == 'cosine':
            # Cosine annealing
            progress = current_epoch / self.cosine_T_max
            progress = min(progress, 1.0)
            return self.cosine_eta_min + (base_lr - self.cosine_eta_min) * (1 + math.cos(math.pi * progress)) / 2
        
        elif self.base_scheduler == 'poly_warmup':
            # Polynomial with warmup (step-based, but we'll adapt for epochs)
            # Convert epochs to approximate steps (assuming ~100 steps per epoch)
            current_step = current_epoch * 100
            
            if current_step < self.poly_warmup_iters:
                # Warm-up phase
                lr_scale = self.poly_warmup_factor + (1.0 - self.poly_warmup_factor) * (current_step / self.poly_warmup_iters)
                return base_lr * lr_scale
            else:
                # Polynomial decay phase (assume total of 5000 steps for decay)
                total_steps = 5000
                progress = (current_step - self.poly_warmup_iters) / (total_steps - self.poly_warmup_iters)
                progress = min(progress, 1.0)
                lr_scale = (1.0 - progress) ** self.poly_power
                return base_lr * lr_scale
        
        else:
            # Default to exponential if unknown scheduler
            return base_lr * (self.exponential_gamma ** current_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current epoch."""
        current_epoch = self.last_epoch
        lrs = []
        
        for i, (base_lr, group) in enumerate(zip(self.base_lrs, self.optimizer.param_groups)):
            group_name = group.get('name', f'group_{i}')
            
            # Ensure base_lr is a float
            if isinstance(base_lr, (list, tuple)):
                base_lr = float(base_lr[0])
            else:
                base_lr = float(base_lr)
            
            if 'encoder' in group_name.lower():
                # Encoder learning rate: only active after finetune_start_epoch
                if current_epoch < self.finetune_start_epoch:
                    lr = 0.0  # Encoder not active yet
                else:
                    # Start from encoder_start_lr and decay
                    epochs_since_finetune = current_epoch - self.finetune_start_epoch
                    lr = self.encoder_start_lr * (self.encoder_decay_factor ** epochs_since_finetune)
            else:
                # Decoder learning rate: use base scheduler
                lr = self._get_decoder_lr(base_lr, current_epoch)
            
            lrs.append(lr)
        
        return lrs
    
    def debug_info(self):
        """Print debug information about the scheduler state."""
        print(f"FinetuneLRScheduler Debug Info:")
        print(f"  Current epoch: {self.last_epoch}")
        print(f"  Base scheduler: {self.base_scheduler}")
        print(f"  Finetune start epoch: {self.finetune_start_epoch}")
        print(f"  Encoder start LR: {self.encoder_start_lr}")
        print(f"  Encoder decay factor: {self.encoder_decay_factor}")
        print(f"  Base LRs: {self.base_lrs}")
        print(f"  Parameter groups: {len(self.optimizer.param_groups)}")
        for i, group in enumerate(self.optimizer.param_groups):
            print(f"    Group {i}: {group.get('name', 'unnamed')} - current LR: {group.get('lr', 'N/A')}")
        current_lrs = self.get_lr()
        print(f"  Calculated LRs: {current_lrs}")


class FinetuneDropScheduler(_LRScheduler):
    """
    Scheduler that applies base scheduler in two phases:
    1. Before finetuning: applies base scheduler starting from original LR
    2. After finetuning: resets base LR and continues with same scheduler
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        finetune_start_epoch (int): Epoch when finetuning begins and LR drops. Default: 10
        finetune_lr (float): Learning rate to use after finetuning starts. Default: 1e-6
        base_scheduler (str): Base scheduler to use. Options: 'exponential', 'cosine'. Default: 'exponential'
        exponential_gamma (float): Gamma for exponential scheduler. Default: 0.95
        cosine_eta_min (float): Minimum LR for cosine scheduler. Default: 1e-7
        cosine_T_max (int): Period for cosine scheduler. Default: 50
    """
    
    def __init__(self, optimizer, finetune_start_epoch=10, finetune_lr=1e-6, 
                 base_scheduler='exponential', exponential_gamma=0.95, 
                 cosine_eta_min=1e-7, cosine_T_max=50, last_epoch=-1, verbose=False):
        
        self.finetune_start_epoch = finetune_start_epoch
        self.finetune_lr = finetune_lr
        self.base_scheduler_name = base_scheduler
        self.finetuning_started = False
        
        # Store parameters for base scheduler
        self.exponential_gamma = exponential_gamma
        self.cosine_eta_min = cosine_eta_min
        self.cosine_T_max = cosine_T_max
        
        # Store optimizer reference before creating scheduler
        self.optimizer = optimizer
        
        # Create scheduler before parent initialization
        self.scheduler = self._create_scheduler()
        
        # Initialize parent class
        super().__init__(optimizer, last_epoch)
    
    def _create_scheduler(self):
        """Create the base scheduler"""
        if self.base_scheduler_name == 'exponential':
            return ExponentialLR(
                self.optimizer, 
                gamma=self.exponential_gamma,
                last_epoch=-1
            )
        elif self.base_scheduler_name == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.cosine_T_max,
                eta_min=self.cosine_eta_min,
                last_epoch=-1
            )
        else:
            raise ValueError(f"Unsupported base scheduler: {self.base_scheduler_name}")
    
    def get_lr(self):
        if self.last_epoch < self.finetune_start_epoch:
            # Before finetuning: use scheduler normally
            self.scheduler.last_epoch = self.last_epoch
            return self.scheduler.get_lr()
        else:
            # After finetuning starts
            if not self.finetuning_started:
                # First time entering finetuning phase
                self.finetuning_started = True
                
                # Reset scheduler's base_lrs to finetune_lr
                self.scheduler.base_lrs = [self.finetune_lr] * len(self.optimizer.param_groups)
                
                # Reset scheduler's epoch counter to start fresh from finetune_lr
                self.scheduler.last_epoch = -1
                
                return [self.finetune_lr] * len(self.optimizer.param_groups)
            else:
                # Continue with scheduler from finetune_lr
                epochs_since_finetune = self.last_epoch - self.finetune_start_epoch
                self.scheduler.last_epoch = epochs_since_finetune
                return self.scheduler.get_lr()


# Example usage:
"""
# Example 1: Exponential decay for decoder
scheduler = FinetuneLRScheduler(
    optimizer,
    base_scheduler='exponential',
    finetune_start_epoch=10,
    encoder_start_lr=1e-6,
    encoder_decay_factor=0.95,
    exponential_gamma=0.95
)

# Example 2: Cosine annealing for decoder  
scheduler = FinetuneLRScheduler(
    optimizer,
    base_scheduler='cosine',
    finetune_start_epoch=10,
    encoder_start_lr=1e-6,
    encoder_decay_factor=0.95,
    cosine_eta_min=1e-6,
    cosine_T_max=50
)

# Example 3: Polynomial warmup for decoder
scheduler = FinetuneLRScheduler(
    optimizer,
    base_scheduler='poly_warmup',
    finetune_start_epoch=10,
    encoder_start_lr=1e-6,
    encoder_decay_factor=0.95,
    poly_warmup_iters=1000,
    poly_power=0.9,
    poly_warmup_factor=0.01
)

# How it works:
# - Epochs 0-9: Only decoder trains with chosen base scheduler (exponential/cosine/poly_warmup)
# - Epoch 10+: Encoder starts training at 1e-6 LR, then decays by encoder_decay_factor each epoch
# - Decoder continues with base scheduler throughout training
""" 