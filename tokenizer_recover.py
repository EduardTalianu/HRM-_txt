#!/usr/bin/env python3
"""
TOKENIZER RECOVERY SCRIPT
Extracts the correct tokenizer from trained model checkpoint and saves it properly
"""

import torch
import json
import os
from collections import Counter

def extract_tokenizer_from_checkpoint(checkpoint_path, output_tokenizer_path='continual_tokenizer.json'):
    """
    Extract tokenizer vocabulary from model checkpoint and save as proper tokenizer file
    """
    print(f"🔧 TOKENIZER RECOVERY")
    print(f"=" * 50)
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return False
    
    try:
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Checkpoint loaded successfully")
        
        # Extract tokenizer vocabulary
        if 'tokenizer_vocab' not in checkpoint_data:
            print(f"❌ No tokenizer_vocab found in checkpoint!")
            print(f"Available keys: {list(checkpoint_data.keys())}")
            return False
        
        tokenizer_vocab = checkpoint_data['tokenizer_vocab']
        print(f"📚 Found tokenizer vocabulary with {len(tokenizer_vocab)} tokens")
        
        # Show some example tokens
        print(f"📖 Sample tokens:")
        sample_tokens = list(tokenizer_vocab.items())[:10]
        for token, id in sample_tokens:
            display_token = repr(token) if token in ['\n', '\t', ' '] else token
            print(f"   '{display_token}' → {id}")
        
        # Reconstruct proper tokenizer format
        tokenizer_data = {
            'level': 'char',  # Your training uses character level
            'vocab': tokenizer_vocab,
            'vocab_size': len(tokenizer_vocab)
        }
        
        # Backup existing tokenizer if it exists
        if os.path.exists(output_tokenizer_path):
            backup_path = output_tokenizer_path + '.backup'
            print(f"💾 Backing up existing tokenizer to: {backup_path}")
            os.rename(output_tokenizer_path, backup_path)
        
        # Save corrected tokenizer
        with open(output_tokenizer_path, 'w') as f:
            json.dump(tokenizer_data, f, indent=2)
        
        print(f"✅ Tokenizer saved to: {output_tokenizer_path}")
        print(f"🔤 Vocabulary size: {len(tokenizer_vocab)}")
        
        # Verify the saved tokenizer
        print(f"\n🧪 VERIFICATION:")
        with open(output_tokenizer_path, 'r') as f:
            verify_data = json.load(f)
        
        print(f"✅ Saved tokenizer verification:")
        print(f"   Level: {verify_data['level']}")
        print(f"   Vocab size: {verify_data['vocab_size']}")
        print(f"   Actual vocab length: {len(verify_data['vocab'])}")
        
        if verify_data['vocab_size'] == len(verify_data['vocab']):
            print(f"✅ Vocabulary size matches!")
        else:
            print(f"⚠️  Size mismatch detected")
        
        # Show training info if available
        if 'datasets_trained' in checkpoint_data:
            datasets = checkpoint_data['datasets_trained']
            print(f"\n📚 Model was trained on: {', '.join(datasets)}")
        
        if 'config' in checkpoint_data:
            config = checkpoint_data['config']
            model_vocab_size = config.get('vocab_size', 'Unknown')
            print(f"🧠 Model expects vocab size: {model_vocab_size}")
            
            if model_vocab_size == len(tokenizer_vocab):
                print(f"✅ Perfect match between model and tokenizer!")
            else:
                print(f"⚠️  Model expects {model_vocab_size} but tokenizer has {len(tokenizer_vocab)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error extracting tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenizer_compatibility(tokenizer_path, checkpoint_path):
    """
    Test if the tokenizer is compatible with the model
    """
    print(f"\n🧪 COMPATIBILITY TEST")
    print(f"=" * 30)
    
    try:
        # Load tokenizer
        with open(tokenizer_path, 'r') as f:
            tokenizer_data = json.load(f)
        
        # Load checkpoint  
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        # Compare vocab sizes
        tokenizer_vocab_size = tokenizer_data['vocab_size']
        model_vocab_size = checkpoint_data['config']['vocab_size']
        checkpoint_vocab_size = len(checkpoint_data.get('tokenizer_vocab', {}))
        
        print(f"📊 Vocabulary Size Comparison:")
        print(f"   Tokenizer file: {tokenizer_vocab_size}")
        print(f"   Model config: {model_vocab_size}")
        print(f"   Checkpoint vocab: {checkpoint_vocab_size}")
        
        if tokenizer_vocab_size == model_vocab_size == checkpoint_vocab_size:
            print(f"✅ Perfect compatibility!")
            return True
        else:
            print(f"❌ Size mismatch detected!")
            return False
            
        # Test encode/decode
        print(f"\n🔤 Testing encode/decode:")
        test_text = "Hello world!"
        vocab = tokenizer_data['vocab']
        
        # Simple encode test
        encoded = [vocab.get(char, vocab.get('<UNK>', 1)) for char in test_text]
        print(f"   Text: '{test_text}'")
        print(f"   Encoded: {encoded}")
        
        # Simple decode test
        inverse_vocab = {v: k for k, v in vocab.items()}
        decoded = ''.join([inverse_vocab.get(id, '') for id in encoded])
        print(f"   Decoded: '{decoded}'")
        
        if decoded == test_text:
            print(f"✅ Encode/decode working correctly!")
        else:
            print(f"❌ Encode/decode mismatch!")
            
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False


def find_available_checkpoints():
    """
    Find all available model checkpoints
    """
    possible_checkpoints = [
        'hrm_continual_learning_best.pt',
        'hrm_continual_learning_latest.pt', 
        'hrm_continual_learning.pt',
        'checkpoint.pt'
    ]
    
    found_checkpoints = []
    for cp in possible_checkpoints:
        if os.path.exists(cp):
            try:
                # Try to load and check if it has tokenizer_vocab
                checkpoint_data = torch.load(cp, map_location='cpu')
                if 'tokenizer_vocab' in checkpoint_data:
                    found_checkpoints.append(cp)
                    print(f"✅ {cp} (has tokenizer)")
                else:
                    print(f"⚠️  {cp} (no tokenizer vocab)")
            except Exception as e:
                print(f"❌ {cp} (corrupted: {e})")
    
    return found_checkpoints


def main():
    print("🔧 HRM TOKENIZER RECOVERY TOOL")
    print("=" * 60)
    print("This tool extracts the correct tokenizer from your trained model")
    print("and fixes mismatched vocabulary issues.")
    print()
    
    # Find available checkpoints
    print("🔍 Scanning for model checkpoints...")
    checkpoints = find_available_checkpoints()
    
    if not checkpoints:
        print("\n❌ No compatible checkpoints found!")
        print("\nLooking for files with tokenizer_vocab:")
        print("  - hrm_continual_learning_best.pt")
        print("  - hrm_continual_learning_latest.pt") 
        print("  - hrm_continual_learning.pt")
        print("  - checkpoint.pt")
        print("\nMake sure you have a trained model checkpoint.")
        return
    
    print(f"\n📁 Found {len(checkpoints)} compatible checkpoint(s):")
    for i, cp in enumerate(checkpoints, 1):
        print(f"  {i}. {cp}")
    
    # Let user choose checkpoint
    if len(checkpoints) == 1:
        selected_checkpoint = checkpoints[0]
        print(f"\n🎯 Using: {selected_checkpoint}")
    else:
        print(f"\nWhich checkpoint should I use?")
        while True:
            try:
                choice = input(f"Enter number (1-{len(checkpoints)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(checkpoints):
                    selected_checkpoint = checkpoints[idx]
                    break
                else:
                    print("Invalid choice. Try again.")
            except ValueError:
                print("Please enter a number.")
    
    print(f"\n🔧 Extracting tokenizer from: {selected_checkpoint}")
    
    # Extract tokenizer
    success = extract_tokenizer_from_checkpoint(
        checkpoint_path=selected_checkpoint,
        output_tokenizer_path='continual_tokenizer.json'
    )
    
    if success:
        print(f"\n🧪 Running compatibility test...")
        test_tokenizer_compatibility('continual_tokenizer.json', selected_checkpoint)
        
        print(f"\n🎉 TOKENIZER RECOVERY COMPLETE!")
        print(f"=" * 40)
        print(f"✅ Fixed tokenizer saved as: continual_tokenizer.json")
        print(f"✅ Should now be compatible with your trained model")
        print(f"\n🚀 Next steps:")
        print(f"  1. Test with: python hrm_chat.py")
        print(f"  2. Try prompt: 'The Queen said'")
        print(f"  3. Should now see proper words instead of garbled text!")
        print(f"\n💡 If you still see issues:")
        print(f"  - Make sure you're using the same checkpoint")
        print(f"  - Try different temperature settings (0.3-0.8)")
        print(f"  - Check that vocab sizes match in the chat script")
        
    else:
        print(f"\n❌ TOKENIZER RECOVERY FAILED")
        print(f"Check the error messages above for details.")


if __name__ == "__main__":
    main()