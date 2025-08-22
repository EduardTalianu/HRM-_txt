#!/usr/bin/env python3
"""
HRM Conversation script with memory AND compatible tokenizer
Combines conversation history with proper tokenizer detection
"""

import torch
import torch.nn.functional as F
import json
import os
import time
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

class FallbackTokenizer:
    """Fallback tokenizer if training tokenizer not found"""
    def __init__(self):
        # Simple character tokenizer
        self.chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\n\t"
        self.char_to_id = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        for i, char in enumerate(self.chars):
            self.char_to_id[char] = i + 4
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        
        # Pad to 1000 vocab size
        while len(self.char_to_id) < 1000:
            self.char_to_id[f'<UNUSED_{len(self.char_to_id)}>'] = len(self.char_to_id)
            self.id_to_char[len(self.char_to_id) - 1] = f'<UNUSED_{len(self.char_to_id) - 1}>'
        
        self.vocab_size = len(self.char_to_id)
    
    def encode(self, text):
        return [self.char_to_id.get(char, 1) for char in text]
    
    def decode(self, ids):
        chars = []
        for id in ids:
            char = self.id_to_char.get(id, '')
            if char in ['<PAD>', '<UNK>', '<START>', '<END>'] or char.startswith('<UNUSED_'):
                continue
            chars.append(char)
        return ''.join(chars)

class ConversationHRMChat:
    def __init__(self, 
                 trained_model_path='hrm_continual_learning.pt',
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
            self.tokenizer = FallbackTokenizer()
            self.is_trained = False
        
        print(f"   Tokenizer vocab size: {self.tokenizer.vocab_size}")
        if hasattr(self.tokenizer, 'level'):
            print(f"   Tokenizer level: {self.tokenizer.level}")
        
        # Load model
        print("Loading HRM model...")
        self.model = self._load_model()
        self.model.eval()
        
        # Conversation state
        self.conversation_history = []
        self.max_context_length = 200  # Maximum tokens to keep in context
        
        print(f"✅ Model loaded on {self.device}")
        print(f"   Parameters: {self.model.count_parameters():,}")
        print(f"   Status: {'🎓 TRAINED' if self.is_trained else '🎲 UNTRAINED (random responses)'}")
        print(f"   Memory: ✅ Conversation history enabled")
    
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
    
    def encode(self, text):
        """Encode text using the correct tokenizer"""
        return self.tokenizer.encode(text)
    
    def decode(self, ids):
        """Decode token ids using the correct tokenizer"""
        return self.tokenizer.decode(ids)
    
    def add_to_conversation(self, user_text, bot_text):
        """Add exchange to conversation history"""
        # Format depends on tokenizer type
        if hasattr(self.tokenizer, 'level') and self.tokenizer.level == 'char':
            # Character-level: compact format
            exchange = f"User: {user_text}\nBot: {bot_text}\n"
        else:
            # Word-level: more structured format
            exchange = f"User: {user_text} Bot: {bot_text} "
            
        self.conversation_history.append(exchange)
        
        # Keep context manageable - adaptive based on tokenizer
        full_context = "".join(self.conversation_history)
        max_chars = self.max_context_length * (4 if hasattr(self.tokenizer, 'level') and self.tokenizer.level == 'char' else 8)
        
        if len(full_context) > max_chars:
            # Remove oldest exchanges, keep last 3-4
            keep_count = 3 if len(self.conversation_history) > 5 else 2
            self.conversation_history = self.conversation_history[-keep_count:]
    
    def get_context_prompt(self, new_user_input):
        """Build prompt with conversation context"""
        if not self.conversation_history:
            if hasattr(self.tokenizer, 'level') and self.tokenizer.level == 'char':
                return f"User: {new_user_input}\nBot:"
            else:
                return f"User: {new_user_input} Bot:"
        
        context = "".join(self.conversation_history)
        
        if hasattr(self.tokenizer, 'level') and self.tokenizer.level == 'char':
            return f"{context}User: {new_user_input}\nBot:"
        else:
            return f"{context}User: {new_user_input} Bot:"
    
    def generate_response(self, user_input, temperature=0.8, N=2, T=2):
        """Generate response with conversation context"""
        
        start_time = time.time()
        
        # Build context-aware prompt
        full_prompt = self.get_context_prompt(user_input)
        
        # Encode using the correct tokenizer
        if hasattr(self.tokenizer, 'encode'):
            # Training tokenizer
            prompt_ids = self.tokenizer.encode(full_prompt)
            if len(prompt_ids) == 0:
                prompt_ids = [2]  # BOS token
        else:
            # Fallback tokenizer  
            prompt_ids = [2] + self.tokenizer.encode(full_prompt)  # Add START
        
        # Limit context to fit model
        if len(prompt_ids) > 64:
            prompt_ids = prompt_ids[-64:]  # Keep last 64 tokens
        
        generated_ids = prompt_ids.copy()
        
        with torch.no_grad():
            for step in range(100):  # Max response length
                # Use sliding window
                input_ids = generated_ids[-64:]
                input_tensor = torch.tensor([input_ids], device=self.device)
                
                # Generate next token
                outputs = self.model(input_tensor, N=N, T=T, use_deep_supervision=False)
                logits = outputs['logits'][0, -1, :] / temperature
                
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
                
                # Stop conditions based on tokenizer type
                if hasattr(self.tokenizer, 'pad_token_id'):
                    # Training tokenizer
                    if next_id in [0, 3]:  # PAD or EOS
                        break
                else:
                    # Fallback tokenizer
                    if next_id in [0, 3]:  # PAD or END
                        break
                
                # Check bounds
                if next_id >= self.tokenizer.vocab_size:
                    break
                
                generated_ids.append(next_id)
                
                # Stop at natural break points - depends on tokenizer
                if hasattr(self.tokenizer, 'level') and self.tokenizer.level == 'char':
                    # Character-level: stop at sentence endings
                    if hasattr(self.tokenizer, 'inverse_vocab'):
                        char = self.tokenizer.inverse_vocab.get(next_id, '')
                        if char == '\n' and step > 10:  # Natural line break
                            break
                else:
                    # Word-level or fallback: stop after reasonable length
                    if step > 20:
                        break
        
        # Extract just the bot's response
        response_ids = generated_ids[len(prompt_ids):]
        response = self.decode(response_ids).strip()
        
        # Clean up response (remove any leftover context)
        if "User:" in response:
            response = response.split("User:")[0].strip()
        if "Bot:" in response:
            response = response.replace("Bot:", "").strip()
        
        generation_time = time.time() - start_time
        
        return response, generation_time
    
    def chat(self):
        print("\n" + "="*70)
        print("🧠💬 HRM CONVERSATIONAL CHATBOT (TOKENIZER-COMPATIBLE)")
        print("="*70)
        print(f"Status: {'🎓 Trained Model' if self.is_trained else '🎲 Untrained Model'}")
        print(f"Memory: ✅ Conversation history enabled")
        print(f"Tokenizer: {self.tokenizer.vocab_size} tokens")
        if hasattr(self.tokenizer, 'level'):
            print(f"Level: {self.tokenizer.level}")
        print()
        print("Commands:")
        print("- 'mode <fast|normal|deep>': Change reasoning")
        print("- 'temp <0.1-2.0>': Change creativity")  
        print("- 'clear': Clear conversation memory")
        print("- 'history': Show conversation history")
        print("- 'status': Show detailed status")
        print("- 'quit': Exit")
        print("="*70 + "\n")
        
        reasoning_modes = {
            'fast': (1, 1),
            'normal': (2, 2), 
            'deep': (3, 2),
        }
        
        current_mode = 'normal'
        temperature = 0.8
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye! 🌟")
                    break
                
                elif user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("🗑️ Conversation history cleared!")
                    continue
                
                elif user_input.lower() == 'history':
                    if self.conversation_history:
                        print("\n📚 Conversation History:")
                        print("".join(self.conversation_history))
                    else:
                        print("📭 No conversation history yet.")
                    continue
                
                elif user_input.lower() == 'status':
                    print(f"\n📊 DETAILED STATUS:")
                    print(f"  Model: {'Trained' if self.is_trained else 'Untrained'}")
                    print(f"  Model path: {self.model_path}")
                    print(f"  Tokenizer type: {type(self.tokenizer).__name__}")
                    print(f"  Vocab size: {self.tokenizer.vocab_size}")
                    if hasattr(self.tokenizer, 'level'):
                        print(f"  Tokenizer level: {self.tokenizer.level}")
                    print(f"  Conversation history: {len(self.conversation_history)} exchanges")
                    print(f"  Current mode: {current_mode}")
                    print(f"  Temperature: {temperature}")
                    
                    if not self.is_trained:
                        print(f"\n💡 To get better responses:")
                        print(f"  1. Run: python train_hrm_repo_style.py")
                        print(f"  2. Restart this chat script")
                    continue
                
                elif user_input.lower() == 'help':
                    print("\nThis chat script combines:")
                    print("  ✅ Conversation memory (remembers context)")
                    print("  ✅ Compatible tokenizer (works with trained models)")
                    print("  ✅ Smart model detection (trained vs untrained)")
                    print(f"\nCurrent: {current_mode} mode, temp={temperature}")
                    print(f"History: {len(self.conversation_history)} exchanges stored")
                    continue
                
                elif user_input.lower().startswith('mode '):
                    mode = user_input[5:].strip().lower()
                    if mode in reasoning_modes:
                        current_mode = mode
                        print(f"🧠 Switched to {mode} reasoning mode")
                    else:
                        print("❌ Use: fast, normal, or deep")
                    continue
                
                elif user_input.lower().startswith('temp '):
                    try:
                        new_temp = float(user_input[5:].strip())
                        if 0.1 <= new_temp <= 2.0:
                            temperature = new_temp
                            print(f"🎛️ Temperature: {temperature}")
                        else:
                            print("❌ Temperature: 0.1-2.0")
                    except:
                        print("❌ Invalid temperature")
                    continue
                
                elif not user_input:
                    continue
                
                # Generate response
                print("🤖 Thinking", end="", flush=True)
                
                N, T = reasoning_modes[current_mode]
                response, gen_time = self.generate_response(
                    user_input, temperature=temperature, N=N, T=T
                )
                
                if not response.strip():
                    response = "..." if not self.is_trained else "I'm not sure how to respond to that."
                
                print(f"\rBot: {response}")
                print(f"     ⏱️ {gen_time:.2f}s | 🧠 {current_mode} | 🌡️ {temperature} | 💬 {len(self.conversation_history)} exchanges")
                
                # Show warning for untrained model
                if not self.is_trained and len(response) > 5:
                    print("     ⚠️  Untrained model - response likely random")
                
                # Add to conversation memory
                self.add_to_conversation(user_input, response)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def main():
    print("🚀 Starting Compatible HRM Conversational Chatbot...")
    print("This version has conversation memory AND uses the correct tokenizer!")
    
    try:
        chatbot = ConversationHRMChat()
        chatbot.chat()
    except FileNotFoundError as e:
        print(f"❌ Error: Required model files not found!")
        print("Make sure you have either:")
        print("  - hrm_model_advanced_with_config.pth (untrained)")
        print("  - hrm_alice_repository_style.pt + alice_tokenizer_repo.json (trained)")
        print("Run hrm_create_3.py and/or train_hrm_repo_style.py first.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()