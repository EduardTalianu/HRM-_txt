import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import json
from tqdm import tqdm
import os
import math
import time
import signal
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Any, Sequence
import tkinter as tk
from tkinter import filedialog, messagebox

# Import the advanced model from hrm_create_3.py
from hrm_create_3 import HierarchicalReasoningModel, create_hrm_model_advanced, CastedEmbedding, CastedLinear, trunc_normal_init_, HRMCarry, HRMInnerCarry


# ============================================================================
# GUI FILE SELECTION
# ============================================================================

def select_text_file():
    """GUI file selector for training text"""
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    print("🎨 Opening file selection dialog...")
    
    file_path = filedialog.askopenfilename(
        title="Select Text File for Training",
        filetypes=[
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ],
        initialdir=os.getcwd()
    )
    
    root.destroy()
    
    if file_path:
        print(f"📚 Selected: {file_path}")
        return file_path
    else:
        print("❌ No file selected. Using default alice.txt")
        return "alice.txt"


def select_training_settings():
    """Simple GUI for training settings"""
    root = tk.Tk()
    root.title("HRM Training Settings")
    root.geometry("400x250")
    root.resizable(False, False)
    
    # Center the window
    root.eval('tk::PlaceWindow . center')
    
    selected_settings = {"cancelled": False, "settings": {}}
    
    # Title
    title_label = tk.Label(root, text="🧠 HRM Continual Learning", font=("Arial", 16, "bold"))
    title_label.pack(pady=10)
    
    subtitle_label = tk.Label(root, text="Simple Mode (No ACT - Stable Training)", font=("Arial", 10))
    subtitle_label.pack(pady=2)
    
    # Settings
    settings_frame = tk.Frame(root)
    settings_frame.pack(pady=15)
    
    tk.Label(settings_frame, text="Epochs:", font=("Arial", 10)).grid(row=0, column=0, sticky="e", padx=5)
    epochs_var = tk.StringVar(value="3")
    tk.Entry(settings_frame, textvariable=epochs_var, width=10).grid(row=0, column=1, padx=5)
    
    tk.Label(settings_frame, text="Batch Size:", font=("Arial", 10)).grid(row=1, column=0, sticky="e", padx=5)
    batch_var = tk.StringVar(value="16")
    tk.Entry(settings_frame, textvariable=batch_var, width=10).grid(row=1, column=1, padx=5)
    
    tk.Label(settings_frame, text="Save every N epochs:", font=("Arial", 10)).grid(row=2, column=0, sticky="e", padx=5)
    save_var = tk.StringVar(value="1")
    tk.Entry(settings_frame, textvariable=save_var, width=10).grid(row=2, column=1, padx=5)
    
    tk.Label(settings_frame, text="Evaluate every N epochs:", font=("Arial", 10)).grid(row=3, column=0, sticky="e", padx=5)
    eval_var = tk.StringVar(value="1")
    tk.Entry(settings_frame, textvariable=eval_var, width=10).grid(row=3, column=1, padx=5)
    
    def start_training():
        try:
            selected_settings["settings"] = {
                "epochs": int(epochs_var.get()),
                "batch_size": int(batch_var.get()),
                "save_every": int(save_var.get()),
                "eval_every": int(eval_var.get())
            }
            root.quit()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for all settings")
    
    def cancel():
        selected_settings["cancelled"] = True
        root.quit()
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=15)
    
    start_btn = tk.Button(button_frame, text="🚀 Start Training", 
                         command=start_training, bg="#4CAF50", fg="white",
                         font=("Arial", 12, "bold"))
    start_btn.pack(side=tk.LEFT, padx=10)
    
    cancel_btn = tk.Button(button_frame, text="❌ Cancel", 
                          command=cancel, bg="#f44336", fg="white")
    cancel_btn.pack(side=tk.LEFT, padx=10)
    
    root.mainloop()
    root.destroy()
    
    return selected_settings


# ============================================================================
# ENHANCED DATASET AND TOKENIZER
# ============================================================================

class SimpleTokenizer:
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

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                'level': self.level,
                'vocab': self.vocab,
                'vocab_size': self.vocab_size
            }, f)


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length=80, stride=1):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride
        self.tokens = tokenizer.encode(text)
        self.num_sequences = max(1, (len(self.tokens) - seq_length) // stride + 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_length
        input_seq = self.tokens[start:end]
        target_seq = self.tokens[start + 1:end + 1]

        # Ensure both sequences are exactly seq_length
        input_seq = input_seq[:self.seq_length] + [self.tokenizer.pad_token_id] * max(0, self.seq_length - len(input_seq))
        target_seq = target_seq[:self.seq_length] + [self.tokenizer.pad_token_id] * max(0, self.seq_length - len(target_seq))

        return {
            'inputs': torch.tensor(input_seq),
            'targets': torch.tensor(target_seq)
        }


# ============================================================================
# TRAINING LOGGER AND PAUSE/RESUME FUNCTIONALITY
# ============================================================================

class TrainingLogger:
    """Comprehensive training logger with file output"""
    def __init__(self, log_file: str = None):
        self.log_file = log_file or f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.start_time = time.time()
        
        # Create log file and write header with UTF-8 encoding
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== HRM Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        
        print(f"📝 Logging to: {self.log_file}")
    
    def log(self, message: str, level: str = "INFO"):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        elapsed = time.time() - self.start_time
        
        formatted_msg = f"[{timestamp}] [{level}] {message}"
        print(formatted_msg)
        
        # Write to file with UTF-8 encoding to handle emojis
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{formatted_msg} (Elapsed: {elapsed:.1f}s)\n")
    
    def log_metrics(self, epoch: int, step: int, metrics: dict):
        """Log training metrics"""
        metrics_str = ", ".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" 
                                for k, v in metrics.items()])
        self.log(f"Epoch {epoch}, Step {step} - {metrics_str}")
    
    def log_config(self, config: dict):
        """Log training configuration"""
        self.log("Training Configuration:")
        for key, value in config.items():
            self.log(f"  {key}: {value}")
        self.log("")


class TrainingState:
    """Enhanced training state with pause/resume support"""
    def __init__(self):
        self.should_pause = False
        self.should_stop = False
        self.last_checkpoint_path = None
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print(f"\n🛑 Training interruption signal received...")
        print("Options:")
        print("1. Pause and save checkpoint (resume later)")
        print("2. Stop training completely")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            self.should_pause = True
            print("⏸️  Will pause after current batch and save checkpoint...")
        else:
            self.should_stop = True
            print("🛑 Will stop training after current batch...")


training_state = TrainingState()


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(train_state, tokenizer, config: dict, datasets_trained: list,
                   checkpoint_path: str, logger: TrainingLogger, 
                   is_best: bool = False, is_pause: bool = False):
    """Save comprehensive checkpoint with continual learning data"""
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
        'datasets_trained': datasets_trained,  # Track learning history
        'timestamp': datetime.now().isoformat(),
        'total_parameters': train_state.model.count_parameters()
    }
    
    # Save main checkpoint
    torch.save(checkpoint_data, checkpoint_path)
    
    # Save best model separately
    if is_best:
        best_path = checkpoint_path.replace('.pt', '_best.pt')
        torch.save(checkpoint_data, best_path)
        logger.log(f"🏆 New best model saved: {best_path}")
    
    # Log checkpoint type
    checkpoint_type = "PAUSE" if is_pause else "BEST" if is_best else "REGULAR"
    logger.log(f"💾 {checkpoint_type} checkpoint saved: {checkpoint_path}")
    logger.log(f"   Datasets trained: {', '.join(datasets_trained)}")
    logger.log(f"   Vocabulary size: {len(tokenizer.vocab)}")


def load_checkpoint_data(checkpoint_path: str, device: str, logger: TrainingLogger, current_dataset: str = None):
    """Load checkpoint data and return training state with continual learning fixes"""
    if not os.path.exists(checkpoint_path):
        return None, None
    
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        
        # Get training state data
        train_state_data = checkpoint_data['train_state']
        
        # CONTINUAL LEARNING FIX: Check if this is a new dataset
        previous_datasets = checkpoint_data.get('datasets_trained', [])
        is_new_dataset = current_dataset and current_dataset not in previous_datasets
        
        if is_new_dataset:
            logger.log(f"🆕 New dataset detected: {current_dataset}")
            logger.log("🔄 Resetting epoch and step counters for new dataset training")
            # Reset epoch and step for new dataset, but preserve model weights
            train_state_data['current_epoch'] = 0
            train_state_data['step'] = 0
            # Keep best_eval_loss to track overall model performance
        else:
            logger.log(f"🔄 Resuming training on same dataset: {current_dataset}")
        
        logger.log(f"📂 Checkpoint loaded from: {checkpoint_path}")
        logger.log(f"   Epoch: {train_state_data['current_epoch']}")
        logger.log(f"   Step: {train_state_data['step']}")
        logger.log(f"   Best eval loss: {train_state_data['best_eval_loss']:.4f}")
        
        return train_state_data, checkpoint_data
        
    except Exception as e:
        logger.log(f"❌ Error loading checkpoint: {e}", "ERROR")
        return None, None


# ============================================================================
# INCREMENTAL VOCABULARY AND MODEL EXPANSION
# ============================================================================

def expand_vocabulary(existing_tokenizer: SimpleTokenizer, new_text: str, logger: TrainingLogger):
    """Expand existing vocabulary with new tokens from new text"""
    logger.log("🔤 Expanding vocabulary with new text...")
    
    old_vocab_size = existing_tokenizer.vocab_size
    old_vocab = existing_tokenizer.vocab.copy()
    
    # Get new tokens
    if existing_tokenizer.level == 'char':
        new_tokens = list(new_text)
    else:
        new_tokens = new_text.split()
    
    # Count new tokens
    counter = Counter(new_tokens)
    new_tokens_added = 0
    
    # Add new tokens to vocabulary
    for token, freq in counter.items():
        if token not in existing_tokenizer.vocab:
            existing_tokenizer.vocab[token] = len(existing_tokenizer.vocab)
            new_tokens_added += 1
    
    # Update vocab size and inverse mapping
    existing_tokenizer.vocab_size = len(existing_tokenizer.vocab)
    existing_tokenizer.inverse_vocab = {v: k for k, v in existing_tokenizer.vocab.items()}
    
    logger.log(f"   Old vocabulary: {old_vocab_size} tokens")
    logger.log(f"   New vocabulary: {existing_tokenizer.vocab_size} tokens")
    logger.log(f"   Added: {new_tokens_added} new tokens")
    
    return new_tokens_added > 0


def expand_model_for_new_vocab(model: HierarchicalReasoningModel, old_vocab_size: int, 
                              new_vocab_size: int, device: str, logger: TrainingLogger):
    """Expand model's embedding and output layers for new vocabulary"""
    if old_vocab_size == new_vocab_size:
        logger.log("📐 No vocabulary expansion needed")
        return model
    
    logger.log(f"🔧 Expanding model layers from {old_vocab_size} to {new_vocab_size} tokens...")
    
    # Get current weights
    old_input_weight = model.input_embedding.embedding_weight.data.clone()
    old_output_weight = model.output_head.weight.data.clone()
    
    # Create new layers with expanded vocabulary
    hidden_size = model.hidden_size
    embed_init_std = 1.0 / math.sqrt(hidden_size)
    
    # New input embedding
    new_input_embedding = CastedEmbedding(new_vocab_size, hidden_size, embed_init_std, model.forward_dtype)
    
    # Copy old weights and initialize new tokens
    with torch.no_grad():
        new_input_embedding.embedding_weight.data[:old_vocab_size] = old_input_weight
        # Initialize new tokens with small random values
        if new_vocab_size > old_vocab_size:
            trunc_normal_init_(
                new_input_embedding.embedding_weight.data[old_vocab_size:], 
                std=embed_init_std
            )
    
    # New output head
    new_output_head = CastedLinear(hidden_size, new_vocab_size, bias=False)
    
    # Copy old weights and initialize new tokens
    with torch.no_grad():
        new_output_head.weight.data[:old_vocab_size] = old_output_weight
        # Initialize new token predictions with small random values
        if new_vocab_size > old_vocab_size:
            trunc_normal_init_(
                new_output_head.weight.data[old_vocab_size:], 
                std=1.0 / math.sqrt(hidden_size)
            )
    
    # Replace layers in model
    model.input_embedding = new_input_embedding
    model.output_head = new_output_head
    model.vocab_size = new_vocab_size
    
    # Move to device
    model = model.to(device)
    
    logger.log(f"✅ Model expanded successfully!")
    logger.log(f"   Input embedding: {old_vocab_size} → {new_vocab_size}")
    logger.log(f"   Output head: {old_vocab_size} → {new_vocab_size}")
    
    return model


def check_for_continual_learning(base_model_path: str, current_dataset: str, logger: TrainingLogger):
    """Check for existing model to continue learning from"""
    possible_paths = [
        base_model_path,
        base_model_path.replace('.pt', '_latest.pt'),
        base_model_path.replace('.pt', '_best.pt'),
        'checkpoint.pt'  # Generic checkpoint file
    ]
    
    existing_models = [path for path in possible_paths if os.path.exists(path)]
    
    if not existing_models:
        logger.log("🆕 No existing model found - starting fresh continual learning")
        return None, False
    
    logger.log(f"🔍 Found existing continual learning models: {existing_models}")
    
    # Check if any model has been trained on the current dataset
    models_with_current_dataset = []
    models_without_current_dataset = []
    
    for model_path in existing_models:
        try:
            data = torch.load(model_path, map_location='cpu')
            datasets_trained = data.get('datasets_trained', [])
            if current_dataset in datasets_trained:
                models_with_current_dataset.append((model_path, data))
            else:
                models_without_current_dataset.append((model_path, data))
        except Exception as e:
            logger.log(f"⚠️  Could not read {model_path}: {e}", "WARNING")
    
    print("\n" + "="*60)
    print("CONTINUAL LEARNING - EXISTING MODEL FOUND")
    print("="*60)
    
    # Show models that have been trained on current dataset
    if models_with_current_dataset:
        print(f"\n🔄 Models already trained on '{current_dataset}':")
        for i, (model_path, data) in enumerate(models_with_current_dataset, 1):
            vocab_size = data['config']['vocab_size']
            total_params = data.get('total_parameters', 'Unknown')
            datasets_trained = data.get('datasets_trained', ['Unknown'])
            timestamp = data.get('timestamp', 'Unknown')
            current_epoch = data.get('train_state', {}).get('current_epoch', 0)
            
            print(f"  {i}. {model_path}")
            print(f"     Vocabulary: {vocab_size} tokens")
            print(f"     Parameters: {total_params:,}" if isinstance(total_params, int) else f"     Parameters: {total_params}")
            print(f"     Datasets: {', '.join(datasets_trained)}")
            print(f"     Last epoch: {current_epoch}")
            print(f"     Last updated: {timestamp}")
            print()
    
    # Show models that haven't been trained on current dataset
    if models_without_current_dataset:
        print(f"\n🆕 Models NOT yet trained on '{current_dataset}':")
        for i, (model_path, data) in enumerate(models_without_current_dataset, len(models_with_current_dataset) + 1):
            vocab_size = data['config']['vocab_size']
            total_params = data.get('total_parameters', 'Unknown')
            datasets_trained = data.get('datasets_trained', ['Unknown'])
            timestamp = data.get('timestamp', 'Unknown')
            
            print(f"  {i}. {model_path}")
            print(f"     Vocabulary: {vocab_size} tokens")
            print(f"     Parameters: {total_params:,}" if isinstance(total_params, int) else f"     Parameters: {total_params}")
            print(f"     Datasets: {', '.join(datasets_trained)}")
            print(f"     Last updated: {timestamp}")
            print()
    
    total_existing_models = len(models_with_current_dataset) + len(models_without_current_dataset)
    print(f"{total_existing_models + 1}. Start completely fresh model (lose all previous learning)")
    
    # Special handling for same dataset
    if models_with_current_dataset:
        print(f"\n⚠️  NOTICE: Found model(s) already trained on '{current_dataset}'")
        print("Options:")
        print("- Continue from checkpoint (resume previous training)")
        print("- Start fresh training from epoch 0 (lose progress on this dataset)")
        print()
    
    while True:
        choice = input(f"\nChoose model to continue learning (1-{total_existing_models + 1}): ").strip()
        try:
            choice_idx = int(choice) - 1
            if choice_idx == total_existing_models:
                logger.log("🆕 User chose to start completely fresh")
                return None, False
            elif 0 <= choice_idx < len(models_with_current_dataset):
                # Selected a model already trained on current dataset
                selected_model = models_with_current_dataset[choice_idx][0]
                print(f"\n⚠️  You selected a model already trained on '{current_dataset}'")
                
                # Make the consequences VERY explicit
                print("="*70)
                print("⚠️  IMPORTANT: WHAT EACH CHOICE MEANS")
                print("="*70)
                print()
                print("1️⃣  RESUME TRAINING FROM CHECKPOINT:")
                print("   ✅ Keep all learned patterns and knowledge from previous training")
                print("   ✅ Continue from where you left off (same epoch/step)")
                print("   ✅ Preserve all model weights and optimizer state")
                print("   ✅ Keep existing vocabulary optimizations")
                print("   📈 BUILD UPON existing knowledge")
                print()
                print("2️⃣  START FRESH TRAINING ON THIS DATASET:")
                print("   ❌ LOSE ALL learned patterns from previous training on this dataset")
                print("   ❌ RESET model back to base pre-trained weights")
                print("   ❌ DISCARD all vocabulary optimizations for this dataset")
                print("   ❌ OVERWRITE your existing checkpoint files")
                print("   ❌ Reset epoch counter back to 0")
                print("   🔥 THROW AWAY all previous learning on this dataset")
                print()
                print("💡 Think of it like:")
                print("   Option 1: Continue reading where you bookmarked")
                print("   Option 2: Burn the book and start reading from page 1 again")
                print()
                print("="*70)
                
                while True:
                    resume_choice = input("Do you want to:\n1. Resume training from checkpoint (RECOMMENDED)\n2. Start fresh training (LOSE ALL PROGRESS)\nChoice (1 or 2): ").strip()
                    if resume_choice == "1":
                        logger.log(f"🔄 Resuming training from: {selected_model}")
                        return selected_model, True
                    elif resume_choice == "2":
                        # Extra confirmation for destructive action
                        print("\n🚨 FINAL WARNING 🚨")
                        print("You are about to PERMANENTLY LOSE all learning progress on this dataset!")
                        print("This action CANNOT be undone!")
                        print()
                        final_confirm = input("Type 'YES DELETE MY PROGRESS' to confirm: ").strip()
                        
                        if final_confirm == "YES DELETE MY PROGRESS":
                            logger.log(f"🔥 User confirmed: Starting fresh training (DISCARDING all progress for '{current_dataset}')")
                            logger.log("⚠️  All previous learning on this dataset will be lost!")
                            return None, False
                        else:
                            print("❌ Confirmation failed. Returning to main menu...")
                            break  # Go back to the choice menu
                    else:
                        print("Invalid choice. Please enter 1 or 2.")
                
            elif choice_idx < total_existing_models:
                # Selected a model not yet trained on current dataset
                adjusted_idx = choice_idx - len(models_with_current_dataset)
                selected_model = models_without_current_dataset[adjusted_idx][0]
                logger.log(f"🧠 Continuing learning from: {selected_model}")
                logger.log(f"📚 Will add '{current_dataset}' to model's knowledge")
                return selected_model, True
                
        except (ValueError, IndexError):
            pass
        print("Invalid choice. Please try again.")


# ============================================================================
# ENHANCED TRAINING STATE
# ============================================================================

@dataclass
class TrainState:
    """Simplified training state management"""
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    total_steps: int
    current_epoch: int = 0
    best_eval_loss: float = float('inf')
    training_start_time: float = 0.0


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, 
    num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    """Learning rate scheduling"""
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def compute_lr(base_lr: float, current_step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.1):
    """Compute learning rate for current step"""
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=current_step,
        base_lr=base_lr,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_ratio=min_ratio
    )


def create_train_state(model, total_steps: int, lr: float = 1e-3, embed_lr: float = 5e-4, 
                      weight_decay: float = 0.01, resume_data: dict = None):
    """Create training state with multi-optimizer setup"""
    
    # Separate optimizers for embeddings and main model
    embed_params = []
    main_params = []
    
    for name, param in model.named_parameters():
        if 'embedding' in name:
            embed_params.append(param)
        else:
            main_params.append(param)
    
    optimizers = []
    optimizer_lrs = []
    
    if embed_params:
        embed_optimizer = torch.optim.AdamW(embed_params, lr=0, weight_decay=weight_decay * 0.1)
        optimizers.append(embed_optimizer)
        optimizer_lrs.append(embed_lr)
    
    main_optimizer = torch.optim.AdamW(main_params, lr=0, weight_decay=weight_decay)
    optimizers.append(main_optimizer)
    optimizer_lrs.append(lr)
    
    # Create train state
    train_state = TrainState(
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        step=0,
        total_steps=total_steps,
        current_epoch=0,
        best_eval_loss=float('inf'),
        training_start_time=time.time()
    )
    
    # Load resume data if provided
    if resume_data:
        train_state.step = resume_data.get('step', 0)
        train_state.current_epoch = resume_data.get('current_epoch', 0)
        train_state.best_eval_loss = resume_data.get('best_eval_loss', float('inf'))
        train_state.training_start_time = resume_data.get('training_start_time', time.time())
        train_state.optimizer_lrs = resume_data.get('optimizer_lrs', optimizer_lrs)
        
        # Load optimizer states if available (and compatible)
        if 'optimizer_states' in resume_data:
            for i, (opt, state) in enumerate(zip(optimizers, resume_data['optimizer_states'])):
                try:
                    opt.load_state_dict(state)
                    if 'logger' in locals():
                        logger.log(f"✅ Loaded optimizer {i+1} state")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load optimizer {i+1} state: {e}")
                    print("   Using fresh optimizer (normal after vocabulary expansion)")
    
    return train_state


class SimpleHRMTrainer:
    """Simplified HRM trainer - no ACT mode for stability"""
    def __init__(self, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.tokenizer = tokenizer
        self.device = device

    def train_batch(self, train_state: TrainState, batch: dict, global_batch_size: int, 
                   warmup_steps: int, N: int = 2, T: int = 2):
        """Simple training batch without ACT complexity"""
        train_state.step += 1
        
        if train_state.step > train_state.total_steps:
            return None
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Initialize carry if None
        if train_state.carry is None:
            train_state.carry = train_state.model.initial_carry(
                batch['inputs'].shape[0], 
                batch['inputs'].shape[1], 
                self.device
            )
        
        # Simple forward pass
        loss, metrics = self._train_simple(train_state, batch, N, T)
        
        if loss is not None:
            # Scale loss by global batch size
            ((1.0 / global_batch_size) * loss).backward()
            
            # Apply optimizers with dynamic learning rates
            current_lr = None
            for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
                current_lr = compute_lr(base_lr, train_state.step, warmup_steps, train_state.total_steps)
                
                for param_group in optim.param_groups:
                    param_group['lr'] = current_lr
                
                # Gradient clipping for stability
                if optim == train_state.optimizers[-1]:  # Main optimizer
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in optim.param_groups for p in group['params']], 
                        1.0
                    )
                
                optim.step()
                optim.zero_grad()
            
            # Add learning rate to metrics
            if metrics is None:
                metrics = {}
            metrics['lr'] = current_lr
            
            return metrics
        
        return None

    def _train_simple(self, train_state: TrainState, batch: dict, N: int, T: int):
        """Simplified training without ACT - more stable"""
        # Simple forward pass
        x_emb = train_state.model.input_embedding(batch['inputs']) * train_state.model.embed_scale
        batch_size, seq_len = batch['inputs'].shape
        
        # Get RoPE
        cos, sin = train_state.model.rope(x_emb, seq_len)
        cos_sin = (cos, sin)
        
        # Get states
        zL = train_state.carry.inner_carry.z_L
        zH = train_state.carry.inner_carry.z_H
        
        # Ensure correct shapes
        if zL.shape[0] != batch_size:
            zL = train_state.model.z0_L.expand(batch_size, seq_len, -1).to(self.device)
            zH = train_state.model.z0_H.expand(batch_size, seq_len, -1).to(self.device)
        
        # Forward computation with limited steps for stability
        with torch.no_grad():
            for i in range(N * T - 1):
                zL = train_state.model.L_module(zL, zH, x_emb, cos_sin=cos_sin)
                if (i + 1) % T == 0:
                    zH = train_state.model.H_module(zH, zL, cos_sin=cos_sin)
        
        # Final step with gradients
        zL = train_state.model.L_module(zL, zH, x_emb, cos_sin=cos_sin)
        zH = train_state.model.H_module(zH, zL, cos_sin=cos_sin)
        
        # Output and loss
        logits = train_state.model.output_head(zH)
        loss = F.cross_entropy(
            logits.reshape(-1, train_state.model.vocab_size),
            batch['targets'].reshape(-1),
            ignore_index=self.tokenizer.pad_token_id
        )
        
        # Update carry (detached for next iteration)
        train_state.carry = HRMCarry(
            inner_carry=HRMInnerCarry(z_H=zH.detach(), z_L=zL.detach()),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=self.device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=self.device),
            current_data={}
        )
        
        metrics = {
            'train/loss': loss.item(),
            'train/perplexity': math.exp(min(loss.item(), 10))  # Cap to prevent overflow
        }
        
        return loss, metrics

    def evaluate(self, train_state: TrainState, eval_loader: DataLoader, N: int = 2, T: int = 2):
        """Simple evaluation"""
        train_state.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.inference_mode():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Simple forward pass for evaluation
                x_emb = train_state.model.input_embedding(batch['inputs']) * train_state.model.embed_scale
                batch_size, seq_len = batch['inputs'].shape
                
                cos, sin = train_state.model.rope(x_emb, seq_len)
                cos_sin = (cos, sin)
                
                # Use initial states for evaluation
                zL = train_state.model.z0_L.expand(batch_size, seq_len, -1).to(self.device)
                zH = train_state.model.z0_H.expand(batch_size, seq_len, -1).to(self.device)
                
                # Fixed number of iterations for evaluation
                for i in range(N * T):
                    zL = train_state.model.L_module(zL, zH, x_emb, cos_sin=cos_sin)
                    if (i + 1) % T == 0:
                        zH = train_state.model.H_module(zH, zL, cos_sin=cos_sin)
                
                logits = train_state.model.output_head(zH)
                loss = F.cross_entropy(
                    logits.reshape(-1, train_state.model.vocab_size),
                    batch['targets'].reshape(-1),
                    ignore_index=self.tokenizer.pad_token_id
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        train_state.model.train()
        
        avg_loss = total_loss / num_batches
        return {
            'eval/loss': avg_loss,
            'eval/perplexity': math.exp(min(avg_loss, 10))
        }

    @torch.no_grad()
    def generate(self, train_state: TrainState, prompt: str, max_length: int = 100, 
                temperature: float = 0.8, N: int = 2, T: int = 2):
        """Generate text using simple inference"""
        train_state.model.eval()
        
        tokens = self.tokenizer.encode(prompt)
        generated = tokens.copy()
        
        for _ in range(max_length):
            current_seq = torch.tensor([generated[-64:]]).to(self.device)
            
            # Simple forward pass
            x_emb = train_state.model.input_embedding(current_seq) * train_state.model.embed_scale
            batch_size, seq_len = current_seq.shape
            
            cos, sin = train_state.model.rope(x_emb, seq_len)
            cos_sin = (cos, sin)
            
            zL = train_state.model.z0_L.expand(batch_size, seq_len, -1).to(self.device)
            zH = train_state.model.z0_H.expand(batch_size, seq_len, -1).to(self.device)
            
            for i in range(N * T):
                zL = train_state.model.L_module(zL, zH, x_emb, cos_sin=cos_sin)
                if (i + 1) % T == 0:
                    zH = train_state.model.H_module(zH, zL, cos_sin=cos_sin)
            
            logits = train_state.model.output_head(zH)[0, -1, :] / temperature
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == self.tokenizer.pad_token_id:
                break
                
            generated.append(next_token)
        
        train_state.model.train()
        return self.tokenizer.decode(generated)


def train_hrm_continual_learning(
    text_path,
    model_config_path='hrm_model_advanced_with_config.pth',
    model_save_path='hrm_continual_learning.pt',
    tokenizer_save_path='continual_tokenizer.json',
    epochs=1,
    batch_size=6,
    seq_length=80,
    lr=1e-3,
    embed_lr=5e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    tokenizer_level='char',
    N=2,
    T=2,
    eval_every=1,
    save_every=1,
    save_every_steps=500,
    auto_resume=True
):
    """Simplified continual learning training with fixed epoch logic"""
    
    # === SETUP ===
    dataset_name = os.path.splitext(os.path.basename(text_path))[0]
    logger = TrainingLogger()
    
    logger.log(f"🧠 CONTINUAL LEARNING MODE - SIMPLE TRAINING")
    logger.log(f"📚 Current dataset: {text_path}")
    logger.log(f"🔄 Model will expand and improve with new data")
    logger.log(f"💾 Unified model path: {model_save_path}")
    logger.log(f"🔤 Unified tokenizer path: {tokenizer_save_path}")
    
    # Set up signal handler
    signal.signal(signal.SIGINT, training_state.signal_handler)
    
    # Log configuration
    config = {
        'text_path': text_path,
        'current_dataset': dataset_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'lr': lr,
        'embed_lr': embed_lr,
        'weight_decay': weight_decay,
        'warmup_ratio': warmup_ratio,
        'tokenizer_level': tokenizer_level,
        'N': N,
        'T': T,
        'use_act': False,  # Always False now
        'eval_every': eval_every,
        'save_every': save_every,
        'save_every_steps': save_every_steps
    }
    logger.log_config(config)
    
    # === DEVICE SETUP ===
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    logger.log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.log(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.log(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.log(f"Using device: {device}")
    
    # Check files
    if not os.path.exists(text_path):
        logger.log(f"Error: Text file {text_path} not found!", "ERROR")
        return
    
    if not os.path.exists(model_config_path):
        logger.log(f"Error: Model config file {model_config_path} not found!", "ERROR")
        return

    logger.log(f"=== Simple HRM Continual Learning ===")
    
    # === CHECK FOR CONTINUAL LEARNING RESUME ===
    resume_from_checkpoint = None
    should_resume = False
    
    if auto_resume:
        resume_from_checkpoint, should_resume = check_for_continual_learning(model_save_path, dataset_name, logger)
    
    # === LOAD AND EXPAND TOKENIZER ===
    with open(text_path, 'r', encoding='utf-8') as f:
        current_text = f.read()[:500000]  # Manageable size
    
    logger.log(f"📖 Current text length: {len(current_text)} characters")
    
    # Handle tokenizer for continual learning
    tokenizer = SimpleTokenizer(level=tokenizer_level)
    old_vocab_size = 0
    
    if should_resume and os.path.exists(tokenizer_save_path):
        # Load existing tokenizer and expand with new data
        logger.log("📂 Loading existing tokenizer for continual learning...")
        with open(tokenizer_save_path, 'r') as f:
            tokenizer_data = json.load(f)
            tokenizer.vocab = tokenizer_data['vocab']
            tokenizer.vocab_size = tokenizer_data['vocab_size']
            tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        
        old_vocab_size = tokenizer.vocab_size
        logger.log(f"   Loaded tokenizer with {old_vocab_size} tokens")
        
        # Expand vocabulary with new text
        vocab_expanded = expand_vocabulary(tokenizer, current_text, logger)
        
        if vocab_expanded:
            # Save expanded tokenizer
            tokenizer.save(tokenizer_save_path)
            logger.log(f"💾 Expanded tokenizer saved")
        
    else:
        # Build new tokenizer from scratch
        logger.log("🔤 Building new tokenizer...")
        tokenizer.build_vocab([current_text])
        tokenizer.save(tokenizer_save_path)
        logger.log(f"   Built tokenizer with {tokenizer.vocab_size} tokens")
    
    # Create datasets
    train_text = current_text[:int(len(current_text) * 0.9)]
    val_text = current_text[int(len(current_text) * 0.9):]
    
    train_dataset = TextDataset(train_text, tokenizer, seq_length=seq_length)
    val_dataset = TextDataset(val_text, tokenizer, seq_length=seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.log(f"📊 Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # === LOAD OR CREATE MODEL ===
    model_config = None
    datasets_trained = [dataset_name]
    resume_data = None
    checkpoint_data = None
    
    if should_resume:
        # Load existing model for continual learning
        logger.log("🧠 Loading model for continual learning...")
        resume_data, checkpoint_data = load_checkpoint_data(resume_from_checkpoint, device, logger, dataset_name)
        
        if resume_data is None or checkpoint_data is None:
            logger.log("❌ Failed to load checkpoint, starting fresh", "ERROR")
            should_resume = False
        else:
            model_config = checkpoint_data['config']
            
            # Load tokenizer from checkpoint first
            checkpoint_tokenizer_vocab = checkpoint_data['tokenizer_vocab']
            
            # Update our tokenizer with checkpoint vocab
            tokenizer.vocab.update(checkpoint_tokenizer_vocab)
            tokenizer.vocab_size = len(tokenizer.vocab)
            tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
            
            # Track previous datasets
            datasets_trained = checkpoint_data.get('datasets_trained', [])
            if dataset_name not in datasets_trained:
                datasets_trained.append(dataset_name)
                logger.log(f"📚 Adding new dataset '{dataset_name}' to model's learning history")
                logger.log(f"   Previously trained on: {', '.join(datasets_trained[:-1])}")
            else:
                logger.log(f"🔄 Continuing training on dataset '{dataset_name}'")
            
            # Check if vocabulary expansion is needed
            old_model_vocab_size = model_config['vocab_size']
            new_vocab_size = tokenizer.vocab_size
            
            if new_vocab_size > old_model_vocab_size:
                logger.log(f"📈 Vocabulary expansion needed: {old_model_vocab_size} → {new_vocab_size}")
                
                # Create model with old vocab size first
                temp_config = model_config.copy()
                temp_config['vocab_size'] = old_model_vocab_size
                model = HierarchicalReasoningModel(**temp_config)
                
                # Load existing weights
                model.load_state_dict(checkpoint_data['model_state_dict'])
                
                # Update config for expanded vocabulary
                model_config['vocab_size'] = new_vocab_size
                
                # Expand model for new vocabulary
                model = expand_model_for_new_vocab(model, old_model_vocab_size, new_vocab_size, device, logger)
                
                # Don't use optimizer states after vocab expansion
                if 'optimizer_states' in resume_data:
                    del resume_data['optimizer_states']
                    logger.log("⚠️  Vocabulary expanded - using fresh optimizers for stability")
                
            else:
                # No expansion needed - create model with current vocab size
                model_config['vocab_size'] = tokenizer.vocab_size
                model = HierarchicalReasoningModel(**model_config)
                model.load_state_dict(checkpoint_data['model_state_dict'])
                logger.log("📂 Model loaded - no vocabulary expansion needed")
    
    if not should_resume:
        # Create completely new model
        logger.log("🆕 Creating new model for continual learning...")
        model_data = torch.load(model_config_path, map_location='cpu')
        model_config = model_data['config']
        model_config['vocab_size'] = tokenizer.vocab_size
        
        model = HierarchicalReasoningModel(**model_config)
        
        # Load compatible weights (excluding vocab-dependent layers)
        state_dict = model_data['model_state_dict']
        keys_to_remove = [k for k in state_dict.keys() if 'input_embedding' in k or 'output_head' in k]
        for key in keys_to_remove:
            del state_dict[key]
        
        model.load_state_dict(state_dict, strict=False)
        logger.log("🎯 Fresh model initialized for continual learning")
    
    # === SETUP MODEL ===
    model = model.float()  # Ensure float32
    model = model.to(device)
    
    logger.log(f"🧠 Model parameters: {model.count_parameters():,}")
    logger.log(f"🔢 Model dtype: {next(model.parameters()).dtype}")
    
    # Calculate training steps
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)
    
    logger.log(f"📈 Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    # Create training state
    train_state = create_train_state(
        model, total_steps, lr=lr, embed_lr=embed_lr, 
        weight_decay=weight_decay, resume_data=resume_data
    )
    
    # FIXED: Handle epoch resumption properly for continual learning
    if should_resume and resume_data:
        is_new_dataset = dataset_name not in checkpoint_data.get('datasets_trained', [])[:-1]  # Exclude current dataset
        
        if is_new_dataset:
            logger.log("🆕 New dataset - starting from epoch 0")
            start_epoch = 0
            train_state.current_epoch = 0
            train_state.step = 0  # Reset step counter for clean training
        else:
            logger.log("🔄 Same dataset - resuming from checkpoint")
            start_epoch = train_state.current_epoch
    else:
        start_epoch = 0
    
    # Set training start time
    if not should_resume or train_state.step == 0:
        train_state.training_start_time = time.time()
    
    # Initialize trainer
    trainer = SimpleHRMTrainer(tokenizer, device=device)
    
    logger.log(f"🚀 Starting simple HRM training from epoch {start_epoch}...")
    
    # === MAIN TRAINING LOOP ===
    try:
        for epoch in range(start_epoch, epochs):
            train_state.current_epoch = epoch
            epoch_start_time = time.time()
            
            logger.log(f"\n=== Epoch {epoch+1}/{epochs} ===")
            
            # Training phase
            model.train()
            epoch_metrics = []
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Check for pause/stop signals
                if training_state.should_pause:
                    logger.log("⏸️  Training paused by user")
                    pause_checkpoint_path = model_save_path.replace('.pt', '_pause.pt')
                    save_checkpoint(train_state, tokenizer, model_config, datasets_trained, 
                                  pause_checkpoint_path, logger, is_pause=True)
                    logger.log(f"💾 Pause checkpoint saved. Resume with the same script.")
                    return
                
                if training_state.should_stop:
                    logger.log("🛑 Training stopped by user")
                    return
                
                # Ensure batch is float32 and on correct device
                batch = {k: v.to(device).float() if v.dtype.is_floating_point else v.to(device) 
                        for k, v in batch.items()}
                
                metrics = trainer.train_batch(
                    train_state, batch, len(train_dataset), warmup_steps, N, T
                )
                
                if metrics:
                    epoch_metrics.append(metrics)
                    progress_bar.set_postfix({
                        'loss': f"{metrics['train/loss']:.4f}",
                        'ppl': f"{metrics['train/perplexity']:.2f}",
                        'lr': f"{metrics['lr']:.6f}",
                        'step': train_state.step
                    })
                    
                    # Log detailed metrics every 100 steps
                    if train_state.step % 100 == 0:
                        logger.log_metrics(epoch+1, train_state.step, metrics)
                    
                    # Save checkpoint every N steps (overwrite previous)
                    if save_every_steps > 0 and train_state.step % save_every_steps == 0:
                        checkpoint_path = 'checkpoint.pt'
                        
                        # Remove previous checkpoint
                        if os.path.exists(checkpoint_path):
                            try:
                                os.remove(checkpoint_path)
                            except:
                                pass
                        
                        save_checkpoint(train_state, tokenizer, model_config, datasets_trained, 
                                      checkpoint_path, logger)
            
            # Calculate epoch time and average metrics
            epoch_time = time.time() - epoch_start_time
            
            if epoch_metrics:
                avg_metrics = {
                    key: sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
                    for key in epoch_metrics[0].keys()
                }
                logger.log(f"📊 Epoch {epoch+1} completed in {epoch_time:.1f}s")
                logger.log(f"📈 Train - Loss: {avg_metrics['train/loss']:.4f}, Perplexity: {avg_metrics['train/perplexity']:.2f}")
            
            # === EVALUATION ===
            should_evaluate = (epoch + 1) % eval_every == 0
            is_best = False
            
            if should_evaluate:
                logger.log("🔍 Running evaluation...")
                eval_start_time = time.time()
                
                eval_metrics = trainer.evaluate(train_state, val_loader, N, T)
                eval_time = time.time() - eval_start_time
                
                logger.log(f"📊 Evaluation completed in {eval_time:.1f}s")
                logger.log(f"📉 Eval - Loss: {eval_metrics['eval/loss']:.4f}, Perplexity: {eval_metrics['eval/perplexity']:.2f}")
                
                # Check if this is the best model
                is_best = eval_metrics['eval/loss'] < train_state.best_eval_loss
                if is_best:
                    train_state.best_eval_loss = eval_metrics['eval/loss']
                    logger.log(f"🎯 New best eval loss: {train_state.best_eval_loss:.4f}")
                
                # Generate sample text
                logger.log("🎨 Generating sample text...")
                sample_prompt = train_text[:30]
                sample = trainer.generate(train_state, sample_prompt, max_length=100, N=N, T=T)
                logger.log(f"Generated Sample:\n{sample}")
                
                # Save checkpoint if best
                if is_best:
                    save_checkpoint(train_state, tokenizer, model_config, datasets_trained, 
                                  model_save_path, logger, is_best=True)
            
            # === REGULAR CHECKPOINT SAVING ===
            should_save = (epoch + 1) % save_every == 0 or epoch == epochs - 1
            if should_save and not (should_evaluate and is_best):  # Don't double-save best models
                save_checkpoint(train_state, tokenizer, model_config, datasets_trained, 
                              model_save_path, logger)
            
            # === ALWAYS SAVE LATEST CHECKPOINT ===
            latest_checkpoint_path = model_save_path.replace('.pt', '_latest.pt')
            save_checkpoint(train_state, tokenizer, model_config, datasets_trained, 
                          latest_checkpoint_path, logger)
        
        # === TRAINING COMPLETION ===
        total_training_time = time.time() - train_state.training_start_time
        logger.log(f"\n🎉 Continual learning session completed!")
        logger.log(f"📚 Model has now learned from: {', '.join(datasets_trained)}")
        logger.log(f"🔤 Final vocabulary size: {tokenizer.vocab_size} tokens")
        logger.log(f"⏱️  Session training time: {total_training_time/60:.1f} minutes")
        logger.log(f"📊 Total steps completed: {train_state.step}/{train_state.total_steps}")
        logger.log(f"💾 Model saved to: {model_save_path}")
        logger.log(f"🏆 Best eval loss achieved: {train_state.best_eval_loss:.4f}")
        
        # List all checkpoint files created
        checkpoint_files = []
        base_name = model_save_path.replace('.pt', '')
        for suffix in ['', '_latest', '_best', '_pause']:
            checkpoint_path = f"{base_name}{suffix}.pt"
            if os.path.exists(checkpoint_path):
                checkpoint_files.append(checkpoint_path)
        
        if os.path.exists('checkpoint.pt'):
            checkpoint_files.append('checkpoint.pt')
        
        if checkpoint_files:
            logger.log(f"📁 Checkpoint files available:")
            for cp_file in checkpoint_files:
                logger.log(f"   {cp_file}")
        
        logger.log(f"📝 Complete training log: {logger.log_file}")
        logger.log(f"🚀 Ready for next dataset! Run script again with new text file.")
        
        # Show completion summary
        print(f"\n" + "="*60)
        print("🎓 CONTINUAL LEARNING COMPLETE")  
        print("="*60)
        print(f"✅ Model has learned from: {', '.join(datasets_trained)}")
        print(f"🔤 Vocabulary expanded to: {tokenizer.vocab_size} tokens")
        print(f"💾 Model saved as: {model_save_path}")
        print(f"📈 Final perplexity: {math.exp(min(train_state.best_eval_loss, 10)):.2f}")
        print()
        print("🚀 Next steps:")
        print("   1. Run this script again with a different text file")
        print("   2. The model will expand its vocabulary and learn new patterns")
        print("   3. All previous knowledge will be preserved!")
        print("="*60)
        
        # Clean up temporary checkpoint
        if os.path.exists('checkpoint.pt'):
            try:
                os.remove('checkpoint.pt')
                logger.log("🗑️  Cleaned up temporary checkpoint file")
            except Exception as e:
                logger.log(f"⚠️  Could not clean up checkpoint.pt: {e}", "WARNING")
        
    except KeyboardInterrupt:
        logger.log("⚠️  Training interrupted!", "WARNING")
    except Exception as e:
        logger.log(f"❌ Training error: {e}", "ERROR")
        import traceback
        logger.log(f"Full traceback:\n{traceback.format_exc()}", "ERROR")
        raise


if __name__ == "__main__":
    print("=== 🧠 CONTINUAL LEARNING HRM TRAINING ===")
    print("Simplified Mode - Stable Training Without ACT")
    print()
    print("Features:")
    print("🧠 One model that learns from multiple datasets")
    print("🔤 Vocabulary expands automatically with new tokens")
    print("📈 Model preserves all previous knowledge")
    print("💾 Smart checkpoint management")
    print("⏸️  Pause/resume with Ctrl+C")
    print("📝 Comprehensive training logs")
    print("🎨 GUI file selection")
    print("🛡️  Stable training (no ACT complexity)")
    print()
    
    # GUI file selection
    print("📁 Please select a text file to train on...")
    text_path = select_text_file()
    
    if not os.path.exists(text_path):
        print(f"❌ Error: File {text_path} not found!")
        print("Please make sure the file exists and try again.")
        sys.exit(1)
    
    # GUI training settings selection
    print("🎨 Opening training configuration dialog...")
    settings_config = select_training_settings()
    
    if settings_config["cancelled"]:
        print("❌ Training cancelled by user")
        sys.exit(0)
    
    settings = settings_config["settings"]
    
    print(f"\n🎯 Continual Learning Session:")
    print(f"   📖 Selected file: {os.path.basename(text_path)}")
    print(f"   🧠 Mode: Simple (Stable)")
    print(f"   🔄 Epochs: {settings['epochs']}")
    print(f"   📦 Batch size: {settings['batch_size']}")
    print(f"   💾 Save every: {settings['save_every']} epochs")
    print(f"   📊 Evaluate every: {settings['eval_every']} epochs")
    print()
    print("💡 How continual learning works:")
    print("   - Model will check for existing knowledge to build upon")
    print("   - Vocabulary will expand to include new tokens")
    print("   - Previous learning will be preserved")
    print("   - Press Ctrl+C to pause and resume later")
    print("   - Run script again with different files to keep learning!")
    print()
    
    input("Press Enter to start continual learning...")
    
    train_hrm_continual_learning(
        text_path=text_path,
        epochs=settings['epochs'],
        batch_size=settings['batch_size'],
        lr=1e-3,
        eval_every=settings['eval_every'],
        save_every=settings['save_every'],
        auto_resume=True
    )