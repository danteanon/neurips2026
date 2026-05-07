"""
FreezeManager - Utility for managing model component freezing/unfreezing

This module provides a centralized way to manage freezing and unfreezing of
model components (e.g., encoder, decoder) with support for:
- Auto-detection of components via module names
- Component-specific learning rates
- Scheduled unfreezing
- State tracking and reporting
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class FreezeManager:
    """
    Manages freezing/unfreezing of model components with support for
    component-specific learning rates.
    
    This class provides a clean interface for:
    - Detecting encoder/decoder/auxiliary components from model structure
    - Freezing and unfreezing components by name
    - Creating optimizer parameter groups with component-specific LRs
    - Tracking freeze state
    
    Attributes:
        model: The PyTorch model to manage
        components: Dict mapping component names to lists of nn.Module
        component_config: Configuration for each component
    """
    
    def __init__(self, model: nn.Module, component_config: Dict):
        """
        Initialize the FreezeManager.
        
        Args:
            model: PyTorch model to manage
            component_config: Dict with component definitions, e.g.:
                {
                    'encoder': {
                        'module_names': ['backbone', 'encoder'],
                        'initial_state': 'frozen',
                        'lr_multiplier': 0.1,
                        'unfreeze_at_epoch': 10
                    },
                    'decoder': {
                        'module_names': ['head', 'decoder', 'segmentation_head'],
                        'initial_state': 'trainable',
                        'lr_multiplier': 1.0
                    }
                }
        """
        self.model = model
        self.component_config = component_config
        self.components: Dict[str, List[nn.Module]] = {}
        
        # Detect components from model structure
        self._detect_components()
        
        logger.info(f"FreezeManager initialized with components: {list(self.components.keys())}")
        for comp_name, modules in self.components.items():
            total_params = sum(p.numel() for m in modules for p in m.parameters())
            logger.info(f"  {comp_name}: {len(modules)} module(s), {total_params:,} parameters")
    
    def _detect_components(self):
        """
        Detect model components based on module_names in config.
        
        For each component in config, checks if the model has an attribute
        matching any of the specified module_names. If found, adds it to
        the components dict.
        """
        for component_name, config in self.component_config.items():
            module_names = config.get('module_names', [])
            
            # Try to find matching modules in the model
            found_modules = []
            for attr_name in module_names:
                if hasattr(self.model, attr_name):
                    module = getattr(self.model, attr_name)
                    if isinstance(module, nn.Module):
                        found_modules.append(module)
                        logger.debug(f"Found {component_name} component: {attr_name}")
            
            if found_modules:
                self.components[component_name] = found_modules
            else:
                logger.warning(
                    f"Component '{component_name}' not found. "
                    f"Looked for attributes: {module_names}"
                )
    
    def freeze(self, component_name: str):
        """
        Freeze all parameters in a component.
        
        Args:
            component_name: Name of the component to freeze
        """
        if component_name not in self.components:
            logger.warning(f"Cannot freeze unknown component: {component_name}")
            return
        
        for module in self.components[component_name]:
            for param in module.parameters():
                param.requires_grad = False
        
        logger.info(f"Frozen component: {component_name}")
    
    def unfreeze(self, component_name: str):
        """
        Unfreeze all parameters in a component.
        
        Args:
            component_name: Name of the component to unfreeze
        """
        if component_name not in self.components:
            logger.warning(f"Cannot unfreeze unknown component: {component_name}")
            return
        
        for module in self.components[component_name]:
            for param in module.parameters():
                param.requires_grad = True
        
        logger.info(f"Unfrozen component: {component_name}")
    
    def get_parameter_groups(self, base_lr: float) -> List[Dict]:
        """
        Create optimizer parameter groups with component-specific learning rates.
        
        This creates one parameter group per component, applying the lr_multiplier
        from the component config to the base_lr.
        
        Args:
            base_lr: Base learning rate to multiply by component lr_multipliers
            
        Returns:
            List of parameter group dicts suitable for optimizer initialization
        """
        param_groups = []
        assigned_params: Set[int] = set()
        
        # Create parameter groups for known components
        for component_name, modules in self.components.items():
            config = self.component_config[component_name]
            lr_multiplier = config.get('lr_multiplier', 1.0)
            component_lr = base_lr * lr_multiplier
            
            # Collect all parameters from this component
            params = []
            for module in modules:
                for param in module.parameters():
                    param_id = id(param)
                    if param_id not in assigned_params:
                        params.append(param)
                        assigned_params.add(param_id)
            
            if params:
                param_groups.append({
                    'params': params,
                    'lr': component_lr,
                    'name': component_name
                })
                logger.debug(
                    f"Parameter group '{component_name}': "
                    f"{len(params)} params, LR={component_lr:.2e}"
                )
        
        # Handle any remaining parameters not assigned to components
        remaining_params = []
        for param in self.model.parameters():
            if id(param) not in assigned_params:
                remaining_params.append(param)
        
        if remaining_params:
            param_groups.append({
                'params': remaining_params,
                'lr': base_lr,
                'name': 'other'
            })
            logger.info(
                f"Parameter group 'other': {len(remaining_params)} unassigned params"
            )
        
        return param_groups
    
    def get_state(self) -> Dict[str, bool]:
        """
        Get current freeze state of all components.
        
        Returns:
            Dict mapping component names to trainable state (True = trainable, False = frozen)
        """
        state = {}
        for component_name, modules in self.components.items():
            # Check if any parameter in the component requires grad
            trainable = any(
                param.requires_grad
                for module in modules
                for param in module.parameters()
            )
            state[component_name] = trainable
        
        return state
    
    def apply_initial_states(self):
        """
        Apply initial freeze states from component config.
        
        This should be called during initialization to set up the initial
        training state (e.g., freeze encoder, train decoder).
        """
        for component_name, config in self.component_config.items():
            initial_state = config.get('initial_state', 'trainable')
            
            if initial_state == 'frozen':
                self.freeze(component_name)
            elif initial_state == 'trainable':
                self.unfreeze(component_name)
            else:
                logger.warning(
                    f"Unknown initial_state '{initial_state}' for {component_name}. "
                    f"Valid values: 'frozen', 'trainable'"
                )
    
    def check_unfreeze_schedule(self, current_epoch: int) -> List[str]:
        """
        Check if any components should be unfrozen at the current epoch.
        
        Args:
            current_epoch: Current training epoch
            
        Returns:
            List of component names that were unfrozen
        """
        unfrozen = []
        
        for component_name, config in self.component_config.items():
            unfreeze_epoch = config.get('unfreeze_at_epoch')
            
            if unfreeze_epoch is not None and current_epoch == unfreeze_epoch:
                # Check if currently frozen
                state = self.get_state()
                if not state.get(component_name, True):  # False = frozen
                    self.unfreeze(component_name)
                    unfrozen.append(component_name)
        
        return unfrozen
    
    def get_component_info(self) -> Dict[str, Dict]:
        """
        Get detailed information about all components.
        
        Returns:
            Dict mapping component names to info dicts with:
                - num_modules: number of nn.Module objects
                - num_parameters: total parameter count
                - trainable: whether component is trainable
                - lr_multiplier: learning rate multiplier
                - unfreeze_at_epoch: scheduled unfreeze epoch (or None)
        """
        info = {}
        state = self.get_state()
        
        for component_name, modules in self.components.items():
            config = self.component_config[component_name]
            num_params = sum(p.numel() for m in modules for p in m.parameters())
            
            info[component_name] = {
                'num_modules': len(modules),
                'num_parameters': num_params,
                'trainable': state.get(component_name, False),
                'lr_multiplier': config.get('lr_multiplier', 1.0),
                'unfreeze_at_epoch': config.get('unfreeze_at_epoch')
            }
        
        return info
    
    def print_state(self):
        """Print a summary of the current component states."""
        print("\n" + "="*60)
        print("FreezeManager Component State")
        print("="*60)
        
        info = self.get_component_info()
        for component_name, comp_info in info.items():
            status = "TRAINABLE" if comp_info['trainable'] else "FROZEN"
            print(f"\n{component_name.upper()}: {status}")
            print(f"  Modules: {comp_info['num_modules']}")
            print(f"  Parameters: {comp_info['num_parameters']:,}")
            print(f"  LR Multiplier: {comp_info['lr_multiplier']}")
            if comp_info['unfreeze_at_epoch'] is not None:
                print(f"  Unfreeze at Epoch: {comp_info['unfreeze_at_epoch']}")
        
        print("="*60 + "\n")

