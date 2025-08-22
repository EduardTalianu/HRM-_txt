#!/usr/bin/env python3
"""
IMPROVED HRM chat script with no artificial response limits
- Matches training context window (64 tokens)
- Removes aggressive early stopping
- Longer default responses
- Better parameter matching with training
- Uses hrm_continual_learning_latest.pt as default
- Interactive mode uses same token limit as unlimited mode
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
        
        # Pad vocab to match model's expected size if needed
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

class ImprovedHRMChatBot:
    def __init__(self, 
                 trained_model_path='hrm_continual_learning_latest.pt',
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
                
                # Get training info if available
                datasets_trained = checkpoint.get('datasets_trained', ['Unknown'])
                print(f"✅ Loaded trained checkpoint")
                print(f"   Datasets learned: {', '.join(datasets_trained)}")
                if 'train_state' in checkpoint:
                    epoch = checkpoint['train_state'].get('current_epoch', 'Unknown')
                    print(f"   Training epoch: {epoch}")
                
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
    
    def generate_response(self, prompt, max_length=1000, temperature=0.8, N=2, T=2, 
                         stop_on_double_newline=True, natural_stopping=True):
        """
        IMPROVED: Generate response with no artificial limits
        
        Args:
            prompt: Input text
            max_length: Maximum response length (increased default)
            temperature: Sampling temperature
            N: Number of high-level reasoning cycles
            T: Number of low-level steps per cycle
            stop_on_double_newline: Stop on paragraph breaks
            natural_stopping: Use intelligent stopping heuristics
        """
        # Encode prompt using the correct tokenizer
        if hasattr(self.tokenizer, 'encode'):
            prompt_ids = self.tokenizer.encode(prompt)
            if len(prompt_ids) == 0:
                prompt_ids = [2]  # BOS token
        else:
            prompt_ids = [2] + self.tokenizer.encode(prompt)
        
        # IMPROVED: Allow longer prompts but still manageable
        if len(prompt_ids) > 60:  # Keep some room for response in 64-token window
            prompt_ids = prompt_ids[-60:]
        
        generated_ids = prompt_ids.copy()
        consecutive_repeats = 0
        last_token = None
        
        with torch.no_grad():
            for step in range(max_length):
                # IMPROVED: Use 64-token context to match training seq_length
                input_ids = generated_ids[-64:]
                input_tensor = torch.tensor([input_ids], device=self.device)
                
                # Get model prediction
                outputs = self.model(input_tensor, N=N, T=T, use_deep_supervision=False)
                logits = outputs['logits'][0, -1, :] / temperature
                
                # Apply top-k sampling for better quality
                top_k = min(50, self.tokenizer.vocab_size)
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered[top_k_indices] = top_k_logits
                    logits = logits_filtered
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
                
                # Basic stopping conditions (only essential ones)
                if hasattr(self.tokenizer, 'pad_token_id'):
                    if next_id in [0, 3]:  # PAD or EOS
                        break
                else:
                    if next_id == 3:  # <END> token
                        break
                    if next_id == 0:  # <PAD> token
                        break
                
                # Check vocabulary bounds
                max_vocab = getattr(self.tokenizer, 'vocab_size', len(getattr(self.tokenizer, 'id_to_char', {})))
                if next_id >= max_vocab:
                    break
                
                # Anti-loop protection: prevent getting stuck repeating same token
                if next_id == last_token:
                    consecutive_repeats += 1
                    if consecutive_repeats > 10:  # Stop if repeating same token too much
                        break
                else:
                    consecutive_repeats = 0
                last_token = next_id
                    
                generated_ids.append(next_id)
                
                # MUCH MORE LENIENT: Only stop after very long responses
                if natural_stopping and hasattr(self.tokenizer, 'level') and self.tokenizer.level == 'char':
                    if hasattr(self.tokenizer, 'inverse_vocab'):
                        char = self.tokenizer.inverse_vocab.get(next_id, '')
                        
                        # Only consider stopping after very substantial content (1500+ tokens)
                        if stop_on_double_newline and char == '\n' and step > 1500:
                            # Only stop on clear paragraph breaks (multiple newlines)
                            recent_text = self.tokenizer.decode(generated_ids[-15:])
                            if recent_text.count('\n') >= 3:  # Multiple newlines indicating clear break
                                break
                        
                        # Emergency stop only for extremely long responses (much higher threshold)
                        elif step > max_length * 0.9 and char in '.!?' and step % 50 == 0:
                            # Only occasionally check for sentence endings near the limit
                            break
        
        # Decode response (skip the original prompt)
        response_ids = generated_ids[len(prompt_ids):]
        response = self.tokenizer.decode(response_ids)
        
        return response.strip()
    
    def interactive_generation(self, prompt, temperature=0.8, N=2, T=2, natural_stopping=True):
        """
        Interactive generation that shows tokens as they're generated
        MODIFIED: Now uses same token limit as unlimited mode (4000 tokens)
        """
        print(f"🔄 Generating with N={N}, T={T}, temp={temperature}...")
        print("Response: ", end="", flush=True)
        
        # Encode prompt
        if hasattr(self.tokenizer, 'encode'):
            prompt_ids = self.tokenizer.encode(prompt)
            if len(prompt_ids) == 0:
                prompt_ids = [2]
        else:
            prompt_ids = [2] + self.tokenizer.encode(prompt)
        
        if len(prompt_ids) > 60:
            prompt_ids = prompt_ids[-60:]
        
        generated_ids = prompt_ids.copy()
        response_start_idx = len(prompt_ids)
        
        with torch.no_grad():
            for step in range(4000):  # MODIFIED: Now uses 4000 tokens like unlimited mode
                input_ids = generated_ids[-64:]
                input_tensor = torch.tensor([input_ids], device=self.device)
                
                outputs = self.model(input_tensor, N=N, T=T, use_deep_supervision=False)
                logits = outputs['logits'][0, -1, :] / temperature
                
                # Apply top-k sampling for better quality
                top_k = min(50, self.tokenizer.vocab_size)
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered[top_k_indices] = top_k_logits
                    logits = logits_filtered
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
                
                # Essential stopping only
                if hasattr(self.tokenizer, 'pad_token_id'):
                    if next_id in [0, 3]:
                        break
                else:
                    if next_id in [0, 3]:
                        break
                
                max_vocab = getattr(self.tokenizer, 'vocab_size', len(getattr(self.tokenizer, 'id_to_char', {})))
                if next_id >= max_vocab:
                    break
                
                # Anti-loop protection: prevent getting stuck repeating same token
                consecutive_repeats = 0
                if hasattr(self, '_last_interactive_token'):
                    if next_id == self._last_interactive_token:
                        consecutive_repeats = getattr(self, '_consecutive_repeats', 0) + 1
                        if consecutive_repeats > 10:
                            break
                    else:
                        consecutive_repeats = 0
                self._last_interactive_token = next_id
                self._consecutive_repeats = consecutive_repeats
                
                generated_ids.append(next_id)
                
                # Show character as it's generated
                if hasattr(self.tokenizer, 'decode'):
                    new_char = self.tokenizer.decode([next_id])
                    print(new_char, end="", flush=True)
                
                # MUCH MORE LENIENT: Only stop on very obvious end patterns after very long responses
                if natural_stopping and hasattr(self.tokenizer, 'level') and self.tokenizer.level == 'char':
                    if hasattr(self.tokenizer, 'inverse_vocab'):
                        char = self.tokenizer.inverse_vocab.get(next_id, '')
                        # Only stop after very substantial content (2000+ tokens) and only on clear paragraph breaks
                        if char == '\n' and step > 2000:
                            # Check for multiple newlines (clear paragraph break)
                            recent_text = self.tokenizer.decode(generated_ids[-10:])
                            if recent_text.count('\n') >= 2:  # Multiple newlines in recent text
                                break
                        # Emergency stop only for extremely long responses (3500+ tokens)
                        elif step > 3500 and char in '.!?' and step % 100 == 0:  # Only check occasionally
                            # Check if we're at end of sentence with some whitespace following
                            if len(generated_ids) > 5:
                                next_chars = self.tokenizer.decode(generated_ids[-5:])
                                if any(c in next_chars for c in ['\n', ' ', '\t']):
                                    break
        
        print()  # New line after generation
        
        # Return just the response part
        response_ids = generated_ids[response_start_idx:]
        return self.tokenizer.decode(response_ids).strip()
    
    def chat(self):
        """Enhanced chat interface with no artificial limits"""
        print("\n" + "="*70)
        print("🤖 HRM CHATBOT - UNLIMITED RESPONSE VERSION")
        print("="*70)
        print(f"Status: {'🎓 Trained Model' if self.is_trained else '🎲 Untrained Model'}")
        print(f"Tokenizer: {self.tokenizer.vocab_size} tokens")
        if hasattr(self.tokenizer, 'level'):
            print(f"Level: {self.tokenizer.level}")
        print()
        print("🚀 IMPROVEMENTS:")
        print("  ✅ Context window: 64 tokens (matches training)")
        print("  ✅ No aggressive early stopping")
        print("  ✅ Much longer default responses")
        print("  ✅ Natural stopping points only")
        print("  ✅ Interactive generation mode with 4000 token limit (DEFAULT ON)")
        print("  ✅ Uses hrm_continual_learning_latest.pt as default")
        print("  ✅ Shows model parameter count in status")
        print()
        print("Commands:")
        print("- 'mode <fast|normal|deep|thorough|extreme>': Change reasoning intensity")
        print("- 'temp <0.1-2.0>': Change creativity (0.1=conservative, 2.0=very creative)")
        print("- 'interactive': Watch response generate token by token (4000 tokens) [DEFAULT ON]")
        print("- 'batch': Normal response generation (faster)")
        print("- 'natural': Natural stopping (stops at clear paragraph breaks after 1500+ tokens)")
        print("- 'unlimited': Disables ALL stopping - only token limit applies (Ctrl+C to stop)")
        print("- 'status': Show detailed configuration")
        print("- 'quit': Exit")
        print("="*70 + "\n")
        
        reasoning_modes = {
            'fast': (1, 1),      # Quick responses
            'normal': (2, 2),    # Default balanced
            'deep': (3, 2),      # More L-module reasoning
            'thorough': (4, 3),  # Deep reasoning
            'extreme': (6, 4)    # Maximum reasoning (may be slow)
        }
        
        current_mode = 'normal'
        temperature = 0.8
        max_length = 1000  # Much longer default
        interactive_mode = True  # MODIFIED: Interactive mode ON by default
        natural_stopping = True
        
        if not self.is_trained:
            print("💡 NOTICE: Using untrained model - responses will be mostly random")
            print("   Run 'python train_hrm_repo_style.py' first for meaningful chat!\n")
        
        print("🎬 Interactive mode is ON by default - you'll see tokens generate live!")
        print("   Use 'batch' command for faster non-interactive generation\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("Goodbye! 👋")
                    break
                
                elif user_input.lower() == 'interactive':
                    interactive_mode = True
                    print("✅ Interactive mode ON - watch tokens generate live (4000 token limit)")
                    continue
                    
                elif user_input.lower() == 'batch':
                    interactive_mode = False
                    print("✅ Batch mode ON - faster generation")
                    continue
                
                elif user_input.lower() == 'natural':
                    natural_stopping = True
                    print("✅ Natural stopping ON - stops at clear paragraph breaks (after 1500+ tokens)")
                    continue
                    
                elif user_input.lower() == 'unlimited':
                    natural_stopping = False
                    max_length = 4000
                    print("✅ Unlimited mode ON - disables all natural stopping (Ctrl+C to stop)")
                    print("   ⚠️  Responses may be very long - use with caution!")
                    continue
                
                elif user_input.lower() == 'status':
                    print(f"\n📊 DETAILED STATUS:")
                    print(f"  Model: {'Trained' if self.is_trained else 'Untrained'}")
                    print(f"  Model path: {self.model_path}")
                    print(f"  Model parameters: {self.model.count_parameters():,}")
                    print(f"  Tokenizer type: {type(self.tokenizer).__name__}")
                    print(f"  Vocab size: {self.tokenizer.vocab_size}")
                    if hasattr(self.tokenizer, 'level'):
                        print(f"  Tokenizer level: {self.tokenizer.level}")
                    print(f"  Current mode: {current_mode} {reasoning_modes[current_mode]}")
                    print(f"  Temperature: {temperature}")
                    print(f"  Max length: {max_length}")
                    print(f"  Interactive: {interactive_mode} {'(4000 tokens)' if interactive_mode else ''}")
                    print(f"  Natural stopping: {natural_stopping} {'(lenient - 1500+ tokens)' if natural_stopping else '(DISABLED - only token limit)'}")
                    print(f"  Device: {self.device}")
                    
                    if not self.is_trained:
                        print(f"\n💡 To get meaningful responses:")
                        print(f"  1. Run: python train_hrm_repo_style.py")
                        print(f"  2. Train on your text files")
                        print(f"  3. Restart this chat script")
                    continue
                
                elif user_input.lower().startswith('mode '):
                    mode = user_input[5:].strip().lower()
                    if mode in reasoning_modes:
                        current_mode = mode
                        N, T = reasoning_modes[mode]
                        print(f"✅ Switched to {mode} mode (N={N}, T={T})")
                        if mode == 'extreme':
                            print("   ⚠️  Extreme mode may be slow but very thorough")
                    else:
                        print("❌ Invalid mode. Use: fast, normal, deep, thorough, or extreme")
                    continue
                
                elif user_input.lower().startswith('temp '):
                    try:
                        new_temp = float(user_input[5:].strip())
                        if 0.1 <= new_temp <= 2.0:
                            temperature = new_temp
                            print(f"✅ Temperature set to {temperature}")
                            if temperature > 1.5:
                                print("   🔥 High temperature - responses will be very creative")
                            elif temperature < 0.3:
                                print("   ❄️  Low temperature - responses will be very conservative")
                        else:
                            print("❌ Temperature must be between 0.1 and 2.0")
                    except:
                        print("❌ Invalid temperature value")
                    continue
                
                elif not user_input:
                    continue
                
                # Generate response
                if interactive_mode:
                    try:
                        N, T = reasoning_modes[current_mode]
                        response = self.interactive_generation(
                            user_input, 
                            temperature=temperature,
                            N=N, 
                            T=T,
                            natural_stopping=natural_stopping
                        )
                    except KeyboardInterrupt:
                        print("\n🛑 Generation stopped by user")
                        continue
                else:
                    print("🤖 Generating...", end="", flush=True)
                    
                    N, T = reasoning_modes[current_mode]
                    try:
                        response = self.generate_response(
                            user_input, 
                            max_length=max_length,
                            temperature=temperature,
                            N=N, 
                            T=T,
                            natural_stopping=natural_stopping
                        )
                        print(f"\rBot: {response}")
                    except KeyboardInterrupt:
                        print("\n🛑 Generation stopped by user")
                        continue
                
                # Response quality feedback
                if self.is_trained:
                    if len(response) < 10:
                        print("     💡 Very short response - try 'unlimited' mode or higher temperature")
                    elif len(response) > 2000:
                        print(f"     🚀 Very long response generated ({len(response)} chars)")
                    elif len(response) > 500:
                        print(f"     ✨ Long response generated ({len(response)} chars)")
                    
                    # Show stopping reason for better understanding
                    if natural_stopping and len(response) > 100:
                        if len(response) < 1000:
                            print("     💭 Stopped early - try 'unlimited' mode for longer responses")
                else:
                    print("     ⚠️  Untrained model - train first for meaningful responses")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error during generation: {e}")
                print("💡 Try adjusting temperature or reasoning mode")

def main():
    print("🚀 Starting IMPROVED HRM Chatbot...")
    print("✨ This version removes artificial limits and matches training parameters!")
    print("🎯 Now uses hrm_continual_learning_latest.pt as default model")
    
    # MODIFIED: Look for different model names, prioritizing 'latest' first
    possible_models = [
        'hrm_continual_learning_latest.pt',  # MOVED TO FIRST PRIORITY
        'hrm_continual_learning_best.pt',
        'hrm_continual_learning.pt'
    ]
    
    found_models = [model for model in possible_models if os.path.exists(model)]
    
    if found_models:
        print(f"\n🎯 Found trained models: {', '.join(found_models)}")
        print("Will automatically use the best available model...")
        
        # MODIFIED: Prioritize latest model first
        if 'hrm_continual_learning_latest.pt' in found_models:
            model_path = 'hrm_continual_learning_latest.pt'
            print("🕐 Using LATEST model for most recent training state")
        elif 'hrm_continual_learning_best.pt' in found_models:
            model_path = 'hrm_continual_learning_best.pt'
            print("🏆 Using BEST model for highest quality responses")
        else:
            model_path = found_models[0]
            print(f"📂 Using: {model_path}")
    else:
        print("\n⚠️  No trained models found - will use untrained model")
        print("💡 Run 'python train_hrm_repo_style.py' first for better chat!")
        model_path = 'hrm_continual_learning_latest.pt'  # Will fallback to untrained
    
    try:
        chatbot = ImprovedHRMChatBot(trained_model_path=model_path)
        chatbot.chat()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Required model files not found!")
        print("\nMake sure you have either:")
        print("  📚 TRAINED: hrm_continual_learning_latest.pt + continual_tokenizer.json")
        print("  🎲 UNTRAINED: hrm_model_advanced_with_config.pth")
        print("\nTo get trained models:")
        print("  1. Run: python hrm_create_3.py")
        print("  2. Run: python train_hrm_repo_style.py")
        print("  3. Train on your text files")
        print("  4. Restart this chat script")
        
    except Exception as e:
        print(f"❌ Error starting chatbot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()