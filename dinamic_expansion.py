import torch
import torch.nn as nn
import math
from typing import Dict, List, Tuple
import json
import os
from hrm_create_3 import (
    TransformerBlock, RecurrentModule, CastedLinear, 
    trunc_normal_init_, HierarchicalReasoningModel
)

# ============================================================================
# CHARACTER TRACKING AND MODEL EXPANSION
# ============================================================================

class CharacterTracker:
    """Track unique characters processed for model expansion"""
    def __init__(self, tracker_file='character_tracker.json'):
        self.tracker_file = tracker_file
        self.total_chars_processed = 0
        self.datasets_char_counts = {}
        self.last_expansion_at = 0
        self.expansion_history = []
        self.load()
    
    def load(self):
        """Load tracking data from file"""
        if os.path.exists(self.tracker_file):
            with open(self.tracker_file, 'r') as f:
                data = json.load(f)
                self.total_chars_processed = data.get('total_chars_processed', 0)
                self.datasets_char_counts = data.get('datasets_char_counts', {})
                self.last_expansion_at = data.get('last_expansion_at', 0)
                self.expansion_history = data.get('expansion_history', [])
    
    def save(self):
        """Save tracking data to file"""
        with open(self.tracker_file, 'w') as f:
            json.dump({
                'total_chars_processed': self.total_chars_processed,
                'datasets_char_counts': self.datasets_char_counts,
                'last_expansion_at': self.last_expansion_at,
                'expansion_history': self.expansion_history
            }, f, indent=2)
    
    def add_dataset(self, dataset_name: str, char_count: int) -> int:
        """Add a dataset's character count (only counts new unique data)"""
        if dataset_name not in self.datasets_char_counts:
            self.datasets_char_counts[dataset_name] = char_count
            self.total_chars_processed += char_count
            self.save()
            return char_count  # New characters added
        return 0  # Dataset already processed
    
    def should_expand(self, chars_per_expansion: int = 3_000_000) -> bool:
        """Check if model should expand based on characters processed"""
        chars_since_last = self.total_chars_processed - self.last_expansion_at
        return chars_since_last >= chars_per_expansion
    
    def record_expansion(self, params_added: int):
        """Record that an expansion occurred"""
        self.expansion_history.append({
            'chars_at_expansion': self.total_chars_processed,
            'params_added': params_added,
            'expansion_number': len(self.expansion_history) + 1
        })
        self.last_expansion_at = self.total_chars_processed
        self.save()


class ModelExpander:
    """Handles dynamic model expansion by adding parameters"""
    
    @staticmethod
    def calculate_expansion_config(current_params: int, target_new_params: int = 1_000_000, 
                                  model: HierarchicalReasoningModel = None) -> Dict:
        """Calculate how to expand model to add ~target_new_params parameters"""
        
        # Get current model configuration
        hidden_size = model.hidden_size
        num_heads = model.num_heads
        current_L_layers = len(model.L_module.layers)
        current_H_layers = len(model.H_module.layers)
        
        # Calculate parameters per layer type
        params_per_L_layer = ModelExpander._calculate_transformer_params(hidden_size, num_heads, model.L_module.layers[0].feed_forward.gate_up_proj.weight.shape[0] // 2)
        params_per_H_layer = ModelExpander._calculate_transformer_params(hidden_size, num_heads, model.H_module.layers[0].feed_forward.gate_up_proj.weight.shape[0] // 2)
        
        # Strategy 1: Add equal mix of L and H layers
        # This maintains architectural balance
        new_L_layers = 0
        new_H_layers = 0
        params_added = 0
        
        while params_added < target_new_params:
            if params_added + params_per_L_layer <= target_new_params * 1.1:  # Allow 10% overshoot
                new_L_layers += 1
                params_added += params_per_L_layer
            
            if params_added + params_per_H_layer <= target_new_params * 1.1:
                new_H_layers += 1
                params_added += params_per_H_layer
            
            # Safety check
            if new_L_layers + new_H_layers > 10:  # Max 10 new layers per expansion
                break
        
        return {
            'new_L_layers': new_L_layers,
            'new_H_layers': new_H_layers,
            'estimated_params': params_added,
            'strategy': 'balanced_layer_addition'
        }
    
    @staticmethod
    def _calculate_transformer_params(hidden_size: int, num_heads: int, intermediate_size: int) -> int:
        """Calculate parameters in a transformer block"""
        # Multi-head attention
        qkv_params = hidden_size * (3 * hidden_size)  # Q, K, V projections
        o_params = hidden_size * hidden_size  # Output projection
        
        # SwiGLU feed-forward
        gate_up_params = hidden_size * (intermediate_size * 2)  # Gate and up projections
        down_params = intermediate_size * hidden_size  # Down projection
        
        # RMS Norm (2 per block)
        norm_params = hidden_size * 2
        
        total = qkv_params + o_params + gate_up_params + down_params + norm_params
        return total
    
    @staticmethod
    def expand_model(model: HierarchicalReasoningModel, expansion_config: Dict, 
                    device: str, logger) -> HierarchicalReasoningModel:
        """Expand model by adding new layers"""
        
        logger.log(f"🔧 Expanding model with strategy: {expansion_config['strategy']}")
        logger.log(f"   Adding {expansion_config['new_L_layers']} L-layers")
        logger.log(f"   Adding {expansion_config['new_H_layers']} H-layers")
        logger.log(f"   Estimated new parameters: {expansion_config['estimated_params']:,}")
        
        # Store current state
        old_param_count = model.count_parameters()
        
        # Add new L-layers
        for _ in range(expansion_config['new_L_layers']):
            new_layer = TransformerBlock(
                model.hidden_size, 
                model.num_heads,
                expansion=model.L_module.layers[0].feed_forward.gate_up_proj.weight.shape[0] // (2 * model.hidden_size),
                causal=False
            )
            # Initialize with small random weights
            ModelExpander._init_layer_weights(new_layer)
            model.L_module.layers.append(new_layer)
        
        # Add new H-layers
        for _ in range(expansion_config['new_H_layers']):
            new_layer = TransformerBlock(
                model.hidden_size,
                model.num_heads, 
                expansion=model.H_module.layers[0].feed_forward.gate_up_proj.weight.shape[0] // (2 * model.hidden_size),
                causal=False
            )
            # Initialize with small random weights
            ModelExpander._init_layer_weights(new_layer)
            model.H_module.layers.append(new_layer)
        
        # Move model to device
        model = model.to(device)
        
        # Verify expansion
        new_param_count = model.count_parameters()
        actual_params_added = new_param_count - old_param_count
        
        logger.log(f"✅ Model expanded successfully!")
        logger.log(f"   Old parameters: {old_param_count:,}")
        logger.log(f"   New parameters: {new_param_count:,}")
        logger.log(f"   Parameters added: {actual_params_added:,}")
        logger.log(f"   L-layers: {len(model.L_module.layers)} total")
        logger.log(f"   H-layers: {len(model.H_module.layers)} total")
        
        return model, actual_params_added
    
    @staticmethod
    def _init_layer_weights(layer: TransformerBlock):
        """Initialize weights for a new transformer layer"""
        for module in layer.modules():
            if isinstance(module, CastedLinear):
                if hasattr(module, 'weight'):
                    # Use truncated normal initialization
                    in_features = module.weight.shape[1]
                    trunc_normal_init_(module.weight.data, std=1.0 / math.sqrt(in_features))
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Parameter):
                # For layer norm weights
                if module.shape[0] > 1:
                    module.data.ones_()
                else:
                    module.data.zero_()


def check_and_expand_model(model: HierarchicalReasoningModel, text_path: str, 
                          dataset_name: str, device: str, logger, 
                          chars_per_million_params: int = 3_000_000,
                          params_per_expansion: int = 1_000_000) -> Tuple[HierarchicalReasoningModel, bool]:
    """Check if model should be expanded based on characters processed"""
    
    # Initialize character tracker
    tracker = CharacterTracker()
    
    # Load and count characters in current dataset
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
        char_count = len(text)
    
    logger.log(f"📊 Dataset '{dataset_name}' contains {char_count:,} characters")
    
    # Add dataset to tracker (only counts if new)
    new_chars = tracker.add_dataset(dataset_name, char_count)
    
    if new_chars > 0:
        logger.log(f"📈 Added {new_chars:,} new characters to training corpus")
        logger.log(f"📚 Total unique characters processed: {tracker.total_chars_processed:,}")
    else:
        logger.log(f"🔄 Dataset already processed - no new characters added")
        return model, False
    
    # Check if expansion is needed
    expanded = False
    while tracker.should_expand(chars_per_million_params):
        logger.log(f"🚀 Expansion threshold reached! ({tracker.total_chars_processed:,} chars processed)")
        
        # Calculate expansion configuration
        expansion_config = ModelExpander.calculate_expansion_config(
            model.count_parameters(), 
            params_per_expansion,
            model
        )
        
        # Expand the model
        model, params_added = ModelExpander.expand_model(model, expansion_config, device, logger)
        
        # Record expansion
        tracker.record_expansion(params_added)
        expanded = True
        
        logger.log(f"📝 Expansion #{len(tracker.expansion_history)} completed")
    
    # Show expansion history
    if tracker.expansion_history:
        logger.log(f"\n📊 Model Expansion History:")
        for exp in tracker.expansion_history:
            logger.log(f"   Expansion #{exp['expansion_number']}: "
                      f"+{exp['params_added']:,} params at {exp['chars_at_expansion']:,} chars")
    
    # Calculate next expansion
    chars_until_next = chars_per_million_params - (tracker.total_chars_processed - tracker.last_expansion_at)
    logger.log(f"⏭️  Next expansion in {chars_until_next:,} characters")
    
    return model, expanded


def save_expanded_model_checkpoint(train_state, tokenizer, config: dict, datasets_trained: list,
                                  checkpoint_path: str, logger, tracker: CharacterTracker,
                                  is_best: bool = False):
    """Save checkpoint with expansion tracking information"""
    
    # Get character tracking info
    tracker_data = {
        'total_chars_processed': tracker.total_chars_processed,
        'expansion_history': tracker.expansion_history,
        'datasets_char_counts': tracker.datasets_char_counts
    }
    
    # Update config with current layer counts
    config['L_layers'] = len(train_state.model.L_module.layers)
    config['H_layers'] = len(train_state.model.H_module.layers)
    
    checkpoint_data = {
        'model_state_dict': train_state.model.state_dict(),
        'optimizer_states': [opt.state_dict() for opt in train_state.optimizers],
        'train_state': {
            'step': train_state.step,
            'total_steps': train_state.total_steps,
            'current_epoch': train_state.current_epoch,
            'best_eval_loss': train_state.best_eval_loss,
            'training_start_time': train_state.training_start_time,
            'optimizer_lrs': train_state.optimizer_lrs
        },
        'tokenizer_vocab': tokenizer.vocab,
        'config': config,
        'datasets_trained': datasets_trained,
        'character_tracking': tracker_data,  # Add tracking info
        'timestamp': datetime.now().isoformat(),
        'total_parameters': train_state.model.count_parameters()
    }
    
    torch.save(checkpoint_data, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_path.replace('.pt', '_best.pt')
        torch.save(checkpoint_data, best_path)
        logger.log(f"🏆 New best model saved: {best_path}")
    
    logger.log(f"💾 Checkpoint saved: {checkpoint_path}")
    logger.log(f"   Model size: {train_state.model.count_parameters():,} parameters")
    logger.log(f"   Characters trained on: {tracker.total_chars_processed:,}")
    logger.log(f"   Expansions performed: {len(tracker.expansion_history)}")


# ============================================================================
# INTEGRATION WITH TRAINING SCRIPT
# ============================================================================

def integrate_expansion_in_training(train_hrm_continual_learning_func):
    """Decorator to add model expansion to the training function"""
    
    def wrapped_training_function(*args, **kwargs):
        # Get arguments
        text_path = args[0] if args else kwargs.get('text_path')
        dataset_name = os.path.splitext(os.path.basename(text_path))[0]
        
        # Add expansion check before main training
        # This would be integrated into the main training loop
        
        # ... existing training code ...
        
        # After loading/creating model, before training:
        # model, expanded = check_and_expand_model(
        #     model, text_path, dataset_name, device, logger,
        #     chars_per_million_params=3_000_000,
        #     params_per_expansion=1_000_000
        # )
        
        # If expanded, reset optimizers for new parameters
        # if expanded:
        #     train_state = create_train_state(model, total_steps, lr, embed_lr, weight_decay)
        
        return train_hrm_continual_learning_func(*args, **kwargs)
    
    return wrapped_training_function


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    from datetime import datetime
    
    class DummyLogger:
        def log(self, msg, level="INFO"):
            print(f"[{level}] {msg}")
    
    # Example: Check expansion needs
    logger = DummyLogger()
    tracker = CharacterTracker()
    
    print("=== Model Expansion System Demo ===")
    print(f"Current chars processed: {tracker.total_chars_processed:,}")
    print(f"Expansion history: {len(tracker.expansion_history)} expansions")
    
    # Simulate adding datasets
    test_datasets = [
        ("dataset1.txt", 1_500_000),  # 1.5M chars
        ("dataset2.txt", 2_000_000),  # 2M chars  -> triggers expansion at 3M
        ("dataset3.txt", 3_500_000),  # 3.5M chars -> triggers expansion at 6M
    ]
    
    for name, char_count in test_datasets:
        print(f"\nProcessing {name} with {char_count:,} characters...")
        new_chars = tracker.add_dataset(name, char_count)
        if new_chars > 0:
            print(f"  Added {new_chars:,} new characters")
            print(f"  Total: {tracker.total_chars_processed:,}")
            
            if tracker.should_expand():
                print(f"  🚀 EXPANSION TRIGGERED!")
                tracker.record_expansion(1_000_000)  # Simulate adding 1M params
        else:
            print(f"  Dataset already processed - skipping")
    
    print(f"\n=== Final Stats ===")
    print(f"Total characters: {tracker.total_chars_processed:,}")
    print(f"Expansions performed: {len(tracker.expansion_history)}")
    for exp in tracker.expansion_history:
        print(f"  Expansion #{exp['expansion_number']}: +{exp['params_added']:,} params at {exp['chars_at_expansion']:,} chars")