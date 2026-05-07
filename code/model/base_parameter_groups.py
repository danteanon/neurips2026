"""
Base classes and utilities for parameter groups functionality.

This module provides a general framework for implementing parameter groups
in PyTorch models, allowing different learning rates for different parts
of the model architecture.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class ParameterGroupsMixin:
    """
    Mixin class that provides parameter groups functionality for PyTorch models.
    
    This mixin can be inherited by any PyTorch model to add support for
    parameter groups with different learning rates. Models should implement
    the _define_parameter_group_mappings method to specify how parameters
    should be grouped.
    
    Example usage:
        class MyModel(nn.Module, ParameterGroupsMixin):
            def __init__(self, ...):
                super().__init__()
                # ... model initialization ...
            
            def _define_parameter_group_mappings(self):
                return [
                    {
                        'name': 'encoder_params',
                        'keywords': ['encoder', 'backbone'],
                        'default_config': {'lr': 1e-5, 'weight_decay': 0.01}
                    },
                    {
                        'name': 'decoder_params', 
                        'keywords': ['decoder', 'head'],
                        'default_config': {'lr': 1e-4, 'weight_decay': 0.01}
                    }
                ]
    """
    
    def get_parameter_groups(self, lr_config: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get parameter groups for different parts of the model with specified learning rates.
        
        Args:
            lr_config: Learning rate configuration for different parameter groups.
                Expected format:
                {
                    'group_name': {'lr': float, 'weight_decay': float, ...},
                    ...
                }
                
        Returns:
            List of parameter groups for optimizer, each containing:
                - 'params': List of parameters
                - 'name': Group name for identification  
                - Additional optimizer parameters (lr, weight_decay, etc.)
        """
        if not hasattr(self, 'named_parameters'):
            raise AttributeError("This mixin requires the class to be a PyTorch nn.Module")
        
        param_groups = []
        all_params = dict(self.named_parameters())
        assigned_params = set()
        
        # Get parameter group mappings from the implementing class
        group_definitions = self._define_parameter_group_mappings()
        
        # Create parameter groups based on definitions
        for group_def in group_definitions:
            group_name = group_def['name']
            keywords = group_def['keywords']
            default_config = group_def['default_config']
            
            # Find parameters matching this group's keywords
            group_params = []
            for param_name, param in all_params.items():
                if any(keyword in param_name for keyword in keywords):
                    group_params.append(param)
                    assigned_params.add(param_name)
            
            # Add parameter group if it has parameters
            if group_params:
                group_config = lr_config.get(group_name, default_config)
                param_groups.append({
                    'params': group_params,
                    'name': group_name,
                    **group_config
                })
        
        # Handle any remaining unassigned parameters
        remaining_params = []
        for param_name, param in all_params.items():
            if param_name not in assigned_params:
                remaining_params.append(param)
        
        if remaining_params:
            # Use default config or fall back to reasonable defaults
            default_config = lr_config.get('default_params', {'lr': 1e-4, 'weight_decay': 0.01})
            param_groups.append({
                'params': remaining_params,
                'name': 'default_params',
                **default_config
            })
        
        return param_groups
    
    def debug_parameter_groups(self, lr_config: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Debug method to print parameter group assignments and statistics.
        
        Args:
            lr_config: Learning rate configuration. If None, uses defaults.
        """
        if lr_config is None:
            lr_config = {}
        
        model_name = self.__class__.__name__
        print("=" * 60)
        print(f"{model_name} Parameter Group Analysis")
        print("=" * 60)
        
        param_groups = self.get_parameter_groups(lr_config)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Total model parameters: {total_params:,}")
        print(f"Number of parameter groups: {len(param_groups)}")
        print()
        
        for i, group in enumerate(param_groups):
            group_params = sum(p.numel() for p in group['params'])
            percentage = (group_params / total_params) * 100
            
            print(f"Group {i+1}: {group['name']}")
            print(f"  Parameters: {group_params:,} ({percentage:.1f}%)")
            print(f"  Learning rate: {group['lr']}")
            print(f"  Weight decay: {group['weight_decay']}")
            
            # Show some example parameter names
            example_names = []
            for param_name, param in self.named_parameters():
                if param in group['params']:
                    example_names.append(param_name)
                if len(example_names) >= 3:  # Show up to 3 examples
                    break
            
            if example_names:
                print(f"  Example parameters:")
                for name in example_names:
                    param_shape = dict(self.named_parameters())[name].shape
                    print(f"    - {name}: {tuple(param_shape)}")
            print()
        
        print("=" * 60)
    
    @abstractmethod
    def _define_parameter_group_mappings(self) -> List[Dict[str, Any]]:
        """
        Define how parameters should be grouped for different learning rates.
        
        This method should be implemented by each model to specify:
        1. Parameter group names
        2. Keywords to identify parameters belonging to each group
        3. Default configurations for each group
        
        Returns:
            List of dictionaries, each containing:
                - 'name': str - Group name
                - 'keywords': List[str] - Keywords to match parameter names
                - 'default_config': Dict - Default lr, weight_decay, etc.
        
        Example:
            return [
                {
                    'name': 'encoder_params',
                    'keywords': ['encoder', 'backbone'],
                    'default_config': {'lr': 1e-5, 'weight_decay': 0.01}
                },
                {
                    'name': 'decoder_params',
                    'keywords': ['decoder', 'head'], 
                    'default_config': {'lr': 1e-4, 'weight_decay': 0.01}
                }
            ]
        """
        pass


class BaseModelWithParameterGroups(nn.Module, ParameterGroupsMixin):
    """
    Base class for models that want to support parameter groups.
    
    This class combines nn.Module with ParameterGroupsMixin to provide
    a complete base class that models can inherit from.
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def _define_parameter_group_mappings(self) -> List[Dict[str, Any]]:
        """
        Define parameter group mappings (must be implemented by subclasses).
        """
        pass
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass (must be implemented by subclasses).
        """
        pass


# Utility functions for common parameter group patterns

def create_encoder_decoder_groups(encoder_keywords: List[str], 
                                decoder_keywords: List[str],
                                encoder_lr: float = 1e-5,
                                decoder_lr: float = 1e-4,
                                weight_decay: float = 0.01) -> List[Dict[str, Any]]:
    """
    Create standard encoder-decoder parameter groups.
    
    Args:
        encoder_keywords: Keywords to identify encoder parameters
        decoder_keywords: Keywords to identify decoder parameters
        encoder_lr: Learning rate for encoder parameters
        decoder_lr: Learning rate for decoder parameters
        weight_decay: Weight decay for all parameters
        
    Returns:
        List of parameter group definitions
    """
    return [
        {
            'name': 'encoder_params',
            'keywords': encoder_keywords,
            'default_config': {'lr': encoder_lr, 'weight_decay': weight_decay}
        },
        {
            'name': 'decoder_params',
            'keywords': decoder_keywords,
            'default_config': {'lr': decoder_lr, 'weight_decay': weight_decay}
        }
    ]


def create_backbone_head_aux_groups(backbone_keywords: List[str],
                                  head_keywords: List[str], 
                                  aux_keywords: List[str],
                                  backbone_lr: float = 1e-5,
                                  head_lr: float = 1e-4,
                                  aux_lr: float = 5e-5,
                                  weight_decay: float = 0.01) -> List[Dict[str, Any]]:
    """
    Create parameter groups for backbone + head + auxiliary outputs architecture.
    
    Args:
        backbone_keywords: Keywords to identify backbone parameters
        head_keywords: Keywords to identify head parameters
        aux_keywords: Keywords to identify auxiliary parameters
        backbone_lr: Learning rate for backbone parameters
        head_lr: Learning rate for head parameters  
        aux_lr: Learning rate for auxiliary parameters
        weight_decay: Weight decay for all parameters
        
    Returns:
        List of parameter group definitions
    """
    return [
        {
            'name': 'backbone_params',
            'keywords': backbone_keywords,
            'default_config': {'lr': backbone_lr, 'weight_decay': weight_decay}
        },
        {
            'name': 'head_params',
            'keywords': head_keywords,
            'default_config': {'lr': head_lr, 'weight_decay': weight_decay}
        },
        {
            'name': 'aux_params',
            'keywords': aux_keywords,
            'default_config': {'lr': aux_lr, 'weight_decay': weight_decay}
        }
    ] 