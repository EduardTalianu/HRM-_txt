#!/usr/bin/env python3
"""
PROPER HRM Chat - Uses the SAME tokenizer that was used during training
This will show actual learned words after training, not random mappings!
"""

import torch
import torch.nn.functional as F
import json
import os
import time
from hrm_create_3 import HierarchicalReasoningModel

class SimpleTokenizer:
    """Same tokenizer class used in training"""
    def __init__(self, level='char'):
        self.level = level
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0
        self.pad_token_id = 0
        self.unk_token_id = 1

    def build_vocab(self, texts, min_freq=1):
        from collections import Counter
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
    def load(cls, path):
        """Load tokenizer from training"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(level=data['level'])
        tokenizer.vocab = data['vocab']
        tokenizer.vocab_size = data['vocab_size']
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        return tokenizer

class ProperHRMChat:
    def __init__(self, 
                 model_path='hrm_alice_repository_style.pt',
                 tokenizer_path='alice_tokenizer_repo.json'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🔍 Looking for trained model and tokenizer...")
        
        # Check if files exist
        model_exists = os.path.exists(model_path)
        tokenizer_exists = os.path.exists(tokenizer_path)
        
        print(f"  Model file ({model_path}): {'✅ Found' if model_exists else '❌ Missing'}")
        print(f"  Tokenizer file ({tokenizer_path}): {'✅ Found' if tokenizer_exists else '❌ Missing'}")
        
        if not model_exists:
            print(f"\n❌ TRAINED MODEL NOT FOUND!")
            print(f"   Expected: {model_path}")
            print(f"   This file is created by running: python train_hrm_repo_style.py")
            print(f"   Using untrained model instead (responses will be random)")
            model_path = 'hrm_model_advanced_with_config.pth'
            self.is_trained = False
        else:
            self.is_trained = True
            
        if not tokenizer_exists:
            print(f"\n❌ TOKENIZER NOT FOUND!")
            print(f"   Expected: {tokenizer_path}")
            print(f"   This file is created by training script")
            print(f"   Will create a basic tokenizer")
            self.tokenizer = self._create_basic_tokenizer()
        else:
            print(f"✅ Loading trained tokenizer...")
            self.tokenizer = SimpleTokenizer.load(tokenizer_path)
            print(f"   Vocabulary size: {self.tokenizer.vocab_size}")
            print(f"   Level: {self.tokenizer.level}")
        
        print(f"\n🤖 Loading model...")
        self.model = self._load_model(model_path)
        
        status = "TRAINED" if self.is_trained else "UNTRAINED"
        print(f"✅ Model loaded: {status}")
        
        if not self.is_trained:
            print(f"\n⚠️  IMPORTANT: This is an untrained model!")
            print(f"   Responses will be random/meaningless")
            print(f"   To get proper responses:")
            print(f"   1. Run: python train_hrm_repo_style.py")
            print(f"   2. Then run this chat script again")
    
    def _create_basic_tokenizer(self):
        """Create basic character tokenizer as fallback"""
        tokenizer = SimpleTokenizer(level='char')
        # Create minimal character set
        chars = " .,!?abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\n"
        tokenizer.vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        for i, char in enumerate(chars):
            tokenizer.vocab[char] = len(tokenizer.vocab)
        tokenizer.vocab_size = len(tokenizer.vocab)
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        return tokenizer
    
    def _load_model(self, model_path):
        """Load model with proper tokenizer compatibility"""
        
        if self.is_trained:
            # Load trained model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            config = checkpoint['config']
            
            # Ensure vocab size matches tokenizer
            if config['vocab_size'] != self.tokenizer.vocab_size:
                print(f"⚠️  Vocab size mismatch: model={config['vocab_size']}, tokenizer={self.tokenizer.vocab_size}")
                config['vocab_size'] = self.tokenizer.vocab_size
            
            model = HierarchicalReasoningModel(**config)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded trained checkpoint from epoch {checkpoint.get('epoch', '?')}")
            
        else:
            # Load untrained model
            model_data = torch.load(model_path, map_location=self.device)
            config = model_data['config']
            config['vocab_size'] = self.tokenizer.vocab_size
            
            model = HierarchicalReasoningModel(**config)
            
            # Load structure but skip embedding/output (different vocab size)
            state_dict = model_data['model_state_dict']
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() 
                           if k in model_dict and 'embedding' not in k and 'output_head' not in k}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
        
        return model.to(self.device)
    
    def generate_response(self, prompt, max_length=50, temperature=0.8, N=2, T=2):
        """Generate response using the CORRECT tokenizer"""
        
        # Encode using the training tokenizer
        prompt_tokens = self.tokenizer.encode(prompt)
        
        print(f"📝 Prompt: '{prompt}'")
        print(f"🔢 Tokens: {prompt_tokens[:10]}{'...' if len(prompt_tokens) > 10 else ''}")
        
        if len(prompt_tokens) == 0:
            prompt_tokens = [2]  # BOS token
        
        generated_tokens = prompt_tokens.copy()
        
        start_time = time.time()
        
        with torch.no_grad():
            for step in range(max_length):
                # Prepare input
                input_tokens = generated_tokens[-64:]  # Last 64 tokens
                input_tensor = torch.tensor([input_tokens], device=self.device)
                
                # Generate
                outputs = self.model(input_tensor, N=N, T=T, use_deep_supervision=False)
                logits = outputs['logits'][0, -1, :]
                
                if temperature > 0:
                    logits = logits / temperature
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = torch.argmax(logits).item()
                
                # Check bounds
                if next_token >= self.tokenizer.vocab_size:
                    break
                
                # Check for stop tokens
                if next_token in [0, 3]:  # PAD or EOS
                    break
                
                generated_tokens.append(next_token)
                
                # Stop at sentence end
                if self.tokenizer.level == 'char':
                    char = self.tokenizer.inverse_vocab.get(next_token, '')
                    if char in '.!?\n' and step > 10:
                        break
        
        # Decode using the training tokenizer
        response_tokens = generated_tokens[len(prompt_tokens):]
        response = self.tokenizer.decode(response_tokens)
        
        elapsed = time.time() - start_time
        
        print(f"🎯 Generated {len(response_tokens)} tokens in {elapsed:.1f}s")
        print(f"🔢 Response tokens: {response_tokens[:10]}{'...' if len(response_tokens) > 10 else ''}")
        
        return response.strip(), elapsed
    
    def chat(self):
        print(f"\n{'='*60}")
        print("🎯 PROPER HRM CHAT")
        print("Uses the SAME tokenizer as training!")
        print(f"{'='*60}")
        print(f"Status: {'✅ TRAINED MODEL' if self.is_trained else '❌ UNTRAINED MODEL'}")
        print(f"Tokenizer: {self.tokenizer.vocab_size} tokens, {self.tokenizer.level}-level")
        print()
        print("Commands:")
        print("  'mode fast/normal/deep' - Change reasoning depth")
        print("  'temp <0.1-2.0>' - Change temperature")
        print("  'info' - Show model status")
        print("  'quit' - Exit")
        print(f"{'='*60}")
        
        reasoning_modes = {
            'fast': (1, 1),
            'normal': (1, 2),
            'deep': (2, 2)
        }
        current_mode = 'fast'
        temperature = 0.8
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 🎯")
                    break
                
                elif user_input.lower().startswith('mode '):
                    mode = user_input[5:].strip()
                    if mode in reasoning_modes:
                        current_mode = mode
                        N, T = reasoning_modes[mode]
                        print(f"🧠 Mode: {mode} (N={N}, T={T})")
                    else:
                        print("❌ Use: fast, normal, or deep")
                    continue
                
                elif user_input.lower().startswith('temp '):
                    try:
                        new_temp = float(user_input[5:].strip())
                        if 0.1 <= new_temp <= 2.0:
                            temperature = new_temp
                            print(f"🌡️ Temperature: {temperature}")
                        else:
                            print("❌ Temperature must be 0.1-2.0")
                    except:
                        print("❌ Invalid temperature")
                    continue
                
                elif user_input.lower() == 'info':
                    print(f"\n📊 MODEL STATUS:")
                    print(f"  Trained: {'Yes' if self.is_trained else 'No'}")
                    print(f"  Vocab size: {self.tokenizer.vocab_size}")
                    print(f"  Tokenizer level: {self.tokenizer.level}")
                    print(f"  Current mode: {current_mode}")
                    print(f"  Temperature: {temperature}")
                    if not self.is_trained:
                        print(f"\n💡 To train the model:")
                        print(f"  python train_hrm_repo_style.py")
                    continue
                
                elif user_input.lower() == 'help':
                    print(f"\nThis chat uses the EXACT tokenizer from training")
                    print(f"So if model is trained, it will show learned words!")
                    print(f"If untrained, responses will still be random.")
                    continue
                
                elif not user_input:
                    continue
                
                # Generate response
                N, T = reasoning_modes[current_mode]
                
                print("🤖 Generating...")
                response, elapsed = self.generate_response(
                    user_input, 
                    max_length=30, 
                    temperature=temperature,
                    N=N, T=T
                )
                
                if not response:
                    response = "[empty response]"
                
                print(f"\nBot: {response}")
                print(f"     ⏱️ {elapsed:.1f}s | 🧠 {current_mode} | 🌡️ {temperature}")
                
                if not self.is_trained and len(response) > 5:
                    print(f"     ⚠️ Model is untrained - response is likely random")
                
            except KeyboardInterrupt:
                print("\n\nBye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    print("🎯 Proper HRM Chat - Uses Training Tokenizer!")
    print("This will show ACTUAL learned words after training")
    
    try:
        chatbot = ProperHRMChat()
        chatbot.chat()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()