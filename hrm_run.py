#!/usr/bin/env python3
"""
Fixed script for using the advanced HRM model from hrm_create_3.py
"""

import torch
import torch.nn.functional as F

# Import the model class BEFORE loading the complete model
try:
    from hrm_create_3 import HierarchicalReasoningModel
    print("✓ Successfully imported HierarchicalReasoningModel")
except ImportError:
    print("✗ Could not import from hrm_create_3.py")
    print("Make sure hrm_create_3.py is in the same directory")
    HierarchicalReasoningModel = None

def load_and_test_hrm():
    """Load the advanced HRM model and run basic tests"""
    
    # Load the saved model with correct filename
    print("Loading advanced HRM model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try different loading methods in order of preference
    model = None
    
    # Method 1: Try complete model (requires class import)
    if HierarchicalReasoningModel is not None:
        try:
            model = torch.load('hrm_model_advanced_complete.pth', map_location=device)
            print("✓ Loaded complete model")
        except FileNotFoundError:
            print("Complete model file not found, trying config method...")
        except Exception as e:
            print(f"Error loading complete model: {e}, trying config method...")
    
    # Method 2: Try loading from config (most robust)
    if model is None:
        try:
            print("Loading model from config...")
            if HierarchicalReasoningModel is None:
                print("Error: HierarchicalReasoningModel class not available")
                raise ImportError("Cannot import HierarchicalReasoningModel")
            
            model_data = torch.load('hrm_model_advanced_with_config.pth', map_location=device)
            config = model_data['config']
            model = HierarchicalReasoningModel(**config)
            model.load_state_dict(model_data['model_state_dict'])
            print("✓ Loaded model from config")
        except FileNotFoundError:
            print("Config file not found, trying state dict...")
        except Exception as e:
            print(f"Error loading from config: {e}")
    
    # Method 3: Try state dict only (requires knowing config)
    if model is None:
        try:
            print("Loading state dict only...")
            if HierarchicalReasoningModel is None:
                print("Error: HierarchicalReasoningModel class not available")
                raise ImportError("Cannot import HierarchicalReasoningModel")
            
            # Use default config if no config file
            default_config = {
                'vocab_size': 1000,
                'hidden_size': 256,
                'num_heads': 8,
                'L_layers': 1,
                'H_layers': 1,
                'expansion': 2.67,
                'halt_max_steps': 8,
                'halt_exploration_prob': 0.1
            }
            
            model = HierarchicalReasoningModel(**default_config)
            state_dict = torch.load('hrm_model_advanced_state_dict.pth', map_location=device)
            model.load_state_dict(state_dict)
            print("✓ Loaded model from state dict with default config")
        except Exception as e:
            print(f"Error loading state dict: {e}")
    
    if model is None:
        raise FileNotFoundError("Could not load model from any of the expected files")
    
    print("Available model files in directory:")
    import os
    for file in os.listdir('.'):
        if file.startswith('hrm_model') and file.endswith('.pth'):
            print(f"  - {file}")
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Create sample input
    batch_size = 2
    seq_length = 16
    sample_input = torch.randint(0, model.vocab_size, (batch_size, seq_length), device=device)
    
    print(f"\nInput shape: {sample_input.shape}")
    print(f"Sample input: {sample_input[0].tolist()}")
    
    # Test different reasoning configurations
    configurations = [
        {"N": 1, "T": 1, "name": "Minimal reasoning"},
        {"N": 2, "T": 2, "name": "Standard reasoning"},
        {"N": 3, "T": 2, "name": "Extended reasoning"},
        {"N": 2, "T": 3, "name": "Deep low-level"},
    ]
    
    print("\n" + "="*60)
    print("TESTING DIFFERENT REASONING CONFIGURATIONS")
    print("="*60)
    
    results = {}
    
    with torch.no_grad():
        for config in configurations:
            N, T, name = config["N"], config["T"], config["name"]
            
            print(f"\n{name} (N={N}, T={T}):")
            
            # Forward pass with correct API
            start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            
            if device.type == 'cuda':
                start_time.record()
            
            # Use correct model API
            outputs = model(sample_input, N=N, T=T, use_deep_supervision=False)
            output = outputs['logits']
            
            if device.type == 'cuda':
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)
                print(f"  Time: {elapsed_time:.2f} ms")
            
            # Analyze outputs
            print(f"  Output shape: {output.shape}")
            print(f"  Number of segments: {outputs.get('num_segments', 'N/A')}")
            if 'steps_taken' in outputs:
                steps = outputs['steps_taken']
                if steps.dtype in [torch.int32, torch.int64]:
                    steps = steps.float()  # Convert to float for mean calculation
                print(f"  Steps taken: {steps.mean().item():.2f}")
            else:
                print(f"  Steps taken: N/A")
            
            # Get top predictions for first sequence
            top_k = 5
            last_token_logits = output[0, -1, :]  # Last position of first sequence
            top_tokens = torch.topk(last_token_logits, top_k)
            
            print(f"  Top {top_k} predictions: {top_tokens.indices.tolist()}")
            print(f"  Prediction confidence: {torch.softmax(last_token_logits, dim=0)[top_tokens.indices[0]].item():.4f}")
            
            results[name] = {
                'output': output,
                'full_outputs': outputs,
                'top_prediction': top_tokens.indices[0].item()
            }
    
    # Compare predictions across configurations
    print("\n" + "="*60)
    print("PREDICTION COMPARISON")
    print("="*60)
    
    base_config = "Standard reasoning"
    base_pred = results[base_config]['top_prediction']
    
    print(f"Base prediction ({base_config}): {base_pred}")
    
    for name, result in results.items():
        if name != base_config:
            pred = result['top_prediction']
            match = "✓" if pred == base_pred else "✗"
            print(f"{name}: {pred} {match}")
    
    # Simple sequence generation example
    print("\n" + "="*60)
    print("SEQUENCE GENERATION EXAMPLE")
    print("="*60)
    
    def generate_sequence(model, prompt_tokens, max_length=20, temperature=1.0):
        """Simple greedy generation with correct API"""
        sequence = prompt_tokens.copy()
        
        for _ in range(max_length - len(sequence)):
            # Prepare input
            input_tensor = torch.tensor([sequence], device=device)
            
            # Get prediction using correct API
            with torch.no_grad():
                outputs = model(input_tensor, N=2, T=2, use_deep_supervision=False)
                output = outputs['logits']
                next_logits = output[0, -1, :] / temperature
                
                # Sample next token (greedy for simplicity)
                next_token = torch.argmax(next_logits).item()
                sequence.append(next_token)
                
                # Simple stopping condition
                if next_token == 0:  # Assuming 0 is stop token
                    break
        
        return sequence
    
    # Generate from a simple prompt
    prompt = [42, 157, 23, 891]
    print(f"Prompt: {prompt}")
    
    generated = generate_sequence(model, prompt, max_length=15, temperature=0.8)
    print(f"Generated: {generated}")
    
    # Test adaptive computation features
    print("\n" + "="*60)
    print("ADAPTIVE COMPUTATION TIME TEST")
    print("="*60)
    
    try:
        with torch.no_grad():
            # Test inference mode (should use ACT internally)
            outputs = model(sample_input, N=2, T=2, use_deep_supervision=False)
            print("✓ Inference mode working")
            print(f"  Steps taken: {outputs.get('steps_taken', 'N/A')}")
            print(f"  Q-halt logits: {outputs.get('q_halt_logits', 'N/A')}")
            print(f"  Q-continue logits: {outputs.get('q_continue_logits', 'N/A')}")
            
            # Test training mode with deep supervision
            model.train()
            sample_targets = torch.randint(0, model.vocab_size, (batch_size, seq_length), device=device)
            outputs = model(sample_input, labels=sample_targets, N=2, T=2, use_deep_supervision=True)
            print("✓ Training mode with deep supervision working")
            print(f"  Loss: {outputs.get('loss', 'N/A')}")
            model.eval()
            
    except Exception as e:
        print(f"✗ Error testing adaptive computation: {e}")
    
    # Test model's internal states (if accessible)
    print("\n" + "="*60)
    print("INTERNAL STATE ANALYSIS")
    print("="*60)
    
    try:
        # Create a carry state to examine
        carry = model.initial_carry(batch_size, seq_length, device)
        print("✓ Can create initial carry state")
        print(f"  High-level state shape: {carry.inner_carry.z_H.shape}")
        print(f"  Low-level state shape: {carry.inner_carry.z_L.shape}")
        print(f"  High-level state norm: {torch.norm(carry.inner_carry.z_H).item():.4f}")
        print(f"  Low-level state norm: {torch.norm(carry.inner_carry.z_L).item():.4f}")
        
    except Exception as e:
        print(f"✗ Cannot access internal states: {e}")
    
    print("\n" + "="*60)
    print("TESTING COMPLETE!")
    print("="*60)
    
    return model, results


if __name__ == "__main__":
    try:
        model, results = load_and_test_hrm()
        print("\n✓ All tests completed successfully!")
        
        # Print summary
        print("\nSUMMARY:")
        print(f"- Model type: {type(model).__name__}")
        print(f"- Device: {next(model.parameters()).device}")
        print(f"- Total parameters: {model.count_parameters():,}")
        print(f"- Tested configurations: {len(results)}")
        
    except FileNotFoundError as e:
        print(f"Error: Model file not found!")
        print("\nThis script looks for these files (in order of preference):")
        print("1. 'hrm_model_advanced_complete.pth' (complete model)")
        print("2. 'hrm_model_advanced_with_config.pth' (model + config)")  
        print("3. 'hrm_model_advanced_state_dict.pth' (state dict only)")
        print("\nMake sure hrm_create_3.py is in the same directory and run it first.")
        print("\nCurrent directory contents:")
        import os
        for file in sorted(os.listdir('.')):
            if file.endswith('.py') or file.endswith('.pth'):
                print(f"  - {file}")
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Make sure hrm_create_3.py is in the same directory as this script.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Make sure hrm_create_3.py is in the same directory")
        print("2. Run hrm_create_3.py first to generate the model files")
        print("3. Check that the model files are not corrupted")