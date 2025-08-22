#!/usr/bin/env python3
"""
Fixed HRM chat script that resolves generation issues
"""

import torch
import torch.nn.functional as F
import json
import os
from collections import Counter
from hrm_create_3 import HierarchicalReasoningModel

class TrainingTokenizer:
    """Same tokenizer class used in train_hrm_repo_style.py"""
    def __init__(self, level='char'):
        self.level = level
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0
        self.pad_token_id = 0
        self.unk_token_id = 1

    def build_vocab(self, texts, min_freq=1):
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        counter = Counter()
        for text in texts:
            tokens = list(text) if self.level == 'char' else text.split()
            counter.update(tokens)
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        self.vocab_size = len(self.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        tokens = list(text) if self.level == 'char' else text.split()
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def decode(self, ids):
        tokens = [self.inverse_vocab.get(id, '') for id in ids if id not in [self.pad_token_id]]
        return ''.join(tokens) if self.level == 'char' else ' '.join(tokens)

    @classmethod
    def load_from_training(cls, path):
        """Load tokenizer saved during training"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(level=data['level'])
        tokenizer.vocab = data['vocab']
        tokenizer.vocab_size = data['vocab_size']
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        return tokenizer

class SimpleChatTokenizer:
    """Fallback tokenizer if training tokenizer not found"""
    def __init__(self):
        # Common characters for text generation
        self.chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\n\t"
        
        # Create vocab
        self.char_to_id = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        for i, char in enumerate(self.chars):
            self.char_to_id[char] = i + 4
            
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        
        # Pad vocab to match model's expected size (1000)
        while len(self.char_to_id) < 1000:
            self.char_to_id[f'<UNUSED_{len(self.char_to_id)}>'] = len(self.char_to_id)
            self.id_to_char[len(self.char_to_id) - 1] = f'<UNUSED_{len(self.char_to_id) - 1}>'
    
    def encode(self, text):
        """Convert text to token ids"""
        return [self.char_to_id.get(char, 1) for char in text]  # 1 is <UNK>
    
    def decode(self, ids):
        """Convert token ids to text"""
        chars = []
        for id in ids:
            char = self.id_to_char.get(id, '')
            if char in ['<PAD>', '<UNK>', '<START>', '<END>'] or char.startswith('<UNUSED_'):
                continue
            chars.append(char)
        return ''.join(chars)

class HRMChatBot:
    def __init__(self, 
                 trained_model_path='hrm_continual_learning_best.pt',
                 untrained_model_path='hrm_model_advanced_with_config.pth',
                 tokenizer_path='continual_tokenizer.json'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🔍 Checking for trained model and tokenizer...")
        
        # Check what's available
        has_trained_model = os.path.exists(trained_model_path)
        has_tokenizer = os.path.exists(tokenizer_path)
        has_untrained_model = os.path.exists(untrained_model_path)
        
        print(f"  Trained model: {'✅' if has_trained_model else '❌'} {trained_model_path}")
        print(f"  Training tokenizer: {'✅' if has_tokenizer else '❌'} {tokenizer_path}")
        print(f"  Untrained model: {'✅' if has_untrained_model else '❌'} {untrained_model_path}")
        
        # Determine configuration
        if has_trained_model and has_tokenizer:
            print("\n🎯 Using TRAINED model with training tokenizer")
            self.model_path = trained_model_path
            self.tokenizer = TrainingTokenizer.load_from_training(tokenizer_path)
            self.is_trained = True
            
        elif has_tokenizer:
            print("\n🔄 Using training tokenizer with untrained model")
            self.model_path = untrained_model_path
            self.tokenizer = TrainingTokenizer.load_from_training(tokenizer_path)
            self.is_trained = False
            
        else:
            print("\n⚠️  Using fallback tokenizer (no training tokenizer found)")
            self.model_path = untrained_model_path
            self.tokenizer = SimpleChatTokenizer()
            self.is_trained = False
        
        print(f"   Tokenizer vocab size: {self.tokenizer.vocab_size}")
        if hasattr(self.tokenizer, 'level'):
            print(f"   Tokenizer level: {self.tokenizer.level}")
        
        print("Loading HRM model...")
        self.model = self._load_model()
        self.model.eval()
        
        print(f"✅ Model loaded on {self.device}")
        print(f"   Parameters: {self.model.count_parameters():,}")
        print(f"   Status: {'🎓 TRAINED' if self.is_trained else '🎲 UNTRAINED (random responses)'}")
        
        # Test model quickly
        print("🧪 Testing model generation...")
        test_response = self._test_generation()
        print(f"   Test output length: {len(test_response)} chars")
        
    def _load_model(self):
        """Load the appropriate model"""
        try:
            if self.is_trained:
                # Load trained model checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device)
                config = checkpoint['config']
                
                # Ensure vocab size matches
                if config['vocab_size'] != self.tokenizer.vocab_size:
                    print(f"⚠️  Adjusting model vocab size: {config['vocab_size']} → {self.tokenizer.vocab_size}")
                    config['vocab_size'] = self.tokenizer.vocab_size
                
                model = HierarchicalReasoningModel(**config)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded trained checkpoint (epoch {checkpoint.get('epoch', '?')})")
                
            else:
                # Load untrained model
                model_data = torch.load(self.model_path, map_location=self.device)
                config = model_data['config']
                config['vocab_size'] = self.tokenizer.vocab_size
                
                model = HierarchicalReasoningModel(**config)
                
                # Load structural weights, skip embedding/output layers
                state_dict = model_data['model_state_dict']
                model_dict = model.state_dict()
                
                filtered_dict = {k: v for k, v in state_dict.items() 
                               if k in model_dict and 'embedding' not in k and 'output_head' not in k}
                
                model_dict.update(filtered_dict)
                model.load_state_dict(model_dict)
                print(f"✅ Loaded untrained model structure")
            
            return model.to(self.device)
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _test_generation(self):
        """Quick test to see if model can generate anything"""
        try:
            test_input = "hello"
            result = self.generate_response(test_input, max_length=20, temperature=1.0)
            return result
        except Exception as e:
            print(f"⚠️  Test generation failed: {e}")
            return ""
    
    def generate_response(self, prompt, max_length=100, temperature=0.8, N=2, T=2):
        """Generate a response to the prompt - FIXED VERSION"""
        self.model.eval()
        
        # Encode prompt using the correct tokenizer
        if hasattr(self.tokenizer, 'encode'):
            # Training tokenizer
            prompt_ids = self.tokenizer.encode(prompt)
            if len(prompt_ids) == 0:
                prompt_ids = [2]  # BOS token
        else:
            # Fallback tokenizer
            prompt_ids = [2] + self.tokenizer.encode(prompt)  # Add START token
        
        # Limit prompt length
        if len(prompt_ids) > 50:
            prompt_ids = prompt_ids[-50:]
        
        print(f"🔍 Debug: Prompt tokens: {prompt_ids[:10]}...")  # Debug output
        
        generated_ids = prompt_ids.copy()
        consecutive_pads = 0  # Track consecutive pad tokens
        
        with torch.no_grad():
            for step in range(max_length):
                # Prepare input (last 32 tokens to fit model context)
                input_ids = generated_ids[-32:]
                input_tensor = torch.tensor([input_ids], device=self.device)
                
                try:
                    # Get model prediction with error handling
                    outputs = self.model(input_tensor, N=N, T=T, use_deep_supervision=False)
                    logits = outputs['logits'][0, -1, :] / temperature
                    
                    # Check for NaN or Inf
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print(f"⚠️  Warning: Invalid logits at step {step}")
                        break
                    
                    # Apply temperature and sample with better sampling
                    if temperature > 0:
                        # Filter out very low probability tokens to avoid pad/unk
                        logits = logits.float()
                        
                        # Boost non-special tokens slightly
                        special_tokens = [0, 1, 2, 3]  # PAD, UNK, BOS, EOS
                        for special_id in special_tokens:
                            if special_id < len(logits):
                                logits[special_id] -= 1.0  # Slight penalty
                        
                        probs = F.softmax(logits, dim=-1)
                        
                        # Sample next token
                        next_id = torch.multinomial(probs, num_samples=1).item()
                    else:
                        next_id = torch.argmax(logits).item()
                    
                    print(f"🔍 Debug: Step {step}, Next token: {next_id}")  # Debug output
                    
                    # FIXED: More lenient stopping conditions
                    
                    # Check bounds first
                    max_vocab = getattr(self.tokenizer, 'vocab_size', 1000)
                    if next_id >= max_vocab:
                        print(f"⚠️  Token {next_id} out of vocab bounds ({max_vocab})")
                        break
                    
                    # Handle PAD tokens more gracefully
                    if next_id == 0:  # PAD token
                        consecutive_pads += 1
                        if consecutive_pads >= 3:  # Allow a few pads before stopping
                            print("🛑 Stopping: Too many consecutive PAD tokens")
                            break
                    else:
                        consecutive_pads = 0
                    
                    # EOS token handling
                    if hasattr(self.tokenizer, 'vocab') and next_id == 3:  # EOS
                        if step > 5:  # Only stop on EOS if we've generated some content
                            print("🛑 Stopping: EOS token")
                            break
                    
                    generated_ids.append(next_id)
                    
                    # Early stopping for character-level models (FIXED)
                    if hasattr(self.tokenizer, 'level') and self.tokenizer.level == 'char':
                        if hasattr(self.tokenizer, 'inverse_vocab'):
                            char = self.tokenizer.inverse_vocab.get(next_id, '')
                            # Only stop on sentence endings if we have substantial content
                            if char in '.!?' and step > 20:  # Increased minimum length
                                print(f"🛑 Stopping: Sentence end '{char}' at step {step}")
                                break
                            # Stop on multiple newlines
                            if char == '\n' and len(generated_ids) > 2 and \
                               self.tokenizer.inverse_vocab.get(generated_ids[-2], '') == '\n':
                                print("🛑 Stopping: Double newline")
                                break
                                
                except Exception as e:
                    print(f"❌ Error during generation at step {step}: {e}")
                    break
        
        # Decode response (skip the original prompt)
        response_ids = generated_ids[len(prompt_ids):]
        
        print(f"🔍 Debug: Response token count: {len(response_ids)}")
        print(f"🔍 Debug: Response tokens: {response_ids[:20]}...")  # Show first 20 tokens
        
        if len(response_ids) == 0:
            return "[No response generated - model may need training]"
        
        response = self.tokenizer.decode(response_ids)
        
        # Clean up response
        response = response.strip()
        
        # Handle empty responses
        if not response or len(response) < 2:
            return "[Model generated empty response - may need more training]"
        
        return response
    
    def chat(self):
        """Enhanced chat interface with better debugging"""
        print("\n" + "="*60)
        print("🤖 HRM CHATBOT (FIXED VERSION)")
        print("="*60)
        print(f"Status: {'🎓 Trained Model' if self.is_trained else '🎲 Untrained Model'}")
        print(f"Tokenizer: {self.tokenizer.vocab_size} tokens")
        if hasattr(self.tokenizer, 'level'):
            print(f"Level: {self.tokenizer.level}")
        print()
        print("Commands:")
        print("- 'mode <fast|normal|deep|thorough>': Change reasoning mode")
        print("- 'temp <0.1-2.0>': Change temperature")
        print("- 'debug <on|off>': Toggle debug output")
        print("- 'test': Run generation test")
        print("- 'status': Show detailed status")
        print("- 'quit': Exit")
        print("="*60 + "\n")
        
        reasoning_modes = {
            'fast': (1, 1),
            'normal': (2, 2), 
            'deep': (3, 2),
            'thorough': (2, 3)
        }
        
        current_mode = 'normal'
        temperature = 0.8
        debug_mode = False
        
        if not self.is_trained:
            print("⚠️  WARNING: Using untrained model - responses will be mostly random!")
            print("   To get meaningful responses, please train the model first.")
            print("   Run: python train_hrm_repo_style_2.py")
            print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye! 👋")
                    break
                
                elif user_input.lower() == 'test':
                    print("🧪 Running generation test...")
                    test_prompts = ["hello", "the", "once upon a time"]
                    for prompt in test_prompts:
                        print(f"   Prompt: '{prompt}'")
                        result = self.generate_response(prompt, max_length=30, temperature=1.0)
                        print(f"   Result: '{result}'")
                        print()
                    continue
                
                elif user_input.lower().startswith('debug '):
                    mode = user_input[6:].strip().lower()
                    if mode == 'on':
                        debug_mode = True
                        print("✅ Debug mode enabled")
                    elif mode == 'off':
                        debug_mode = False
                        print("✅ Debug mode disabled")
                    else:
                        print("❌ Use 'debug on' or 'debug off'")
                    continue
                
                elif user_input.lower() == 'status':
                    print(f"\n📊 DETAILED STATUS:")
                    print(f"  Model: {'Trained' if self.is_trained else 'Untrained'}")
                    print(f"  Model path: {self.model_path}")
                    print(f"  Tokenizer type: {type(self.tokenizer).__name__}")
                    print(f"  Vocab size: {self.tokenizer.vocab_size}")
                    if hasattr(self.tokenizer, 'level'):
                        print(f"  Tokenizer level: {self.tokenizer.level}")
                    print(f"  Current mode: {current_mode}")
                    print(f"  Temperature: {temperature}")
                    print(f"  Debug mode: {debug_mode}")
                    print(f"  Device: {self.device}")
                    
                    # Test a few token lookups
                    if hasattr(self.tokenizer, 'vocab'):
                        print(f"  Sample tokens: {list(self.tokenizer.vocab.items())[:5]}")
                    
                    if not self.is_trained:
                        print(f"\n💡 To get better responses:")
                        print(f"  1. Run: python train_hrm_repo_style_2.py")
                        print(f"  2. Restart this chat script")
                    continue
                
                elif user_input.lower().startswith('mode '):
                    mode = user_input[5:].strip().lower()
                    if mode in reasoning_modes:
                        current_mode = mode
                        print(f"✅ Switched to {mode} mode")
                    else:
                        print("❌ Invalid mode. Use: fast, normal, deep, or thorough")
                    continue
                
                elif user_input.lower().startswith('temp '):
                    try:
                        new_temp = float(user_input[5:].strip())
                        if 0.1 <= new_temp <= 2.0:
                            temperature = new_temp
                            print(f"✅ Temperature set to {temperature}")
                        else:
                            print("❌ Temperature must be between 0.1 and 2.0")
                    except:
                        print("❌ Invalid temperature value")
                    continue
                
                elif not user_input:
                    continue
                
                # Generate response
                print("🤖 Thinking...", end="", flush=True)
                
                N, T = reasoning_modes[current_mode]
                
                # FIXED: Better error handling and debug output
                try:
                    response = self.generate_response(
                        user_input, 
                        max_length=150, 
                        temperature=temperature,
                        N=N, 
                        T=T
                    )
                    
                    print(f"\rBot: {response}")
                    
                    # Show warning for untrained model
                    if not self.is_trained and len(response) > 10:
                        print("     ⚠️  Untrained model - response may be random")
                        
                except Exception as e:
                    print(f"\r❌ Generation error: {e}")
                    if debug_mode:
                        import traceback
                        traceback.print_exc()
                    print("Try adjusting temperature or mode, or check model training.")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                if debug_mode:
                    import traceback
                    traceback.print_exc()

def main():
    print("🚀 Starting FIXED HRM Chatbot...")
    print("This version includes debugging and better generation logic!")
    
    try:
        chatbot = HRMChatBot()
        chatbot.chat()
        
    except FileNotFoundError as e:
        print(f"❌ Error: Required model files not found!")
        print("Make sure you have either:")
        print("  - hrm_model_advanced_with_config.pth (untrained)")
        print("  - hrm_continual_learning_best.pt + continual_tokenizer.json (trained)")
        print("Run hrm_create_3.py and/or train_hrm_repo_style_2.py first.")
        
    except Exception as e:
        print(f"❌ Error starting chatbot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()