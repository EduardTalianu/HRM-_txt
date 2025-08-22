#!/usr/bin/env python3
"""
HRM Architecture Demo - Shows how the reasoning works WITHOUT fake text generation
This is honest about what an untrained model can and cannot do.
"""

import torch
import torch.nn.functional as F
import numpy as np
from hrm_create_3 import HierarchicalReasoningModel

class HRMArchitectureDemo:
    def __init__(self, model_path='hrm_model_advanced_with_config.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("Loading HRM architecture (untrained model)...")
        self.model = self._load_model(model_path)
        print("✓ Architecture loaded")
        print("⚠️  NOTE: This is an UNTRAINED model - no language knowledge!")
        print("   It can only demonstrate the reasoning ARCHITECTURE.")
    
    def _load_model(self, model_path):
        model_data = torch.load(model_path, map_location=self.device)
        config = model_data['config']
        
        model = HierarchicalReasoningModel(**config)
        
        # Load structural weights (but skip language-specific layers)
        state_dict = model_data['model_state_dict']
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() 
                        if k in model_dict and 'embedding' not in k and 'output_head' not in k}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        
        return model.to(self.device)
    
    def analyze_reasoning_patterns(self, input_sequence, N=2, T=2):
        """Analyze how the model's internal reasoning changes over cycles"""
        
        print(f"\n🧠 REASONING ARCHITECTURE ANALYSIS")
        print(f"Input sequence length: {len(input_sequence)}")
        print(f"Reasoning configuration: N={N} cycles, T={T} steps per cycle")
        print("=" * 60)
        
        # Create input tensor
        x = torch.tensor([input_sequence], device=self.device)
        batch_size, seq_len = x.shape
        
        # Get embeddings
        x_emb = self.model.input_embedding(x) * self.model.embed_scale
        cos, sin = self.model.rope(x_emb, seq_len)
        cos_sin = (cos, sin)
        
        # Initialize states
        zL = self.model.z0_L.expand(batch_size, seq_len, -1).to(self.device)
        zH = self.model.z0_H.expand(batch_size, seq_len, -1).to(self.device)
        
        print(f"📊 INITIAL STATE ANALYSIS:")
        self._analyze_state(zL, "Low-level (zL)")
        self._analyze_state(zH, "High-level (zH)")
        
        state_history = []
        
        with torch.no_grad():
            for cycle in range(N):
                print(f"\n🔄 REASONING CYCLE {cycle + 1}/{N}")
                print("-" * 40)
                
                cycle_states = {'cycle': cycle + 1, 'steps': []}
                
                for step in range(T):
                    # Store pre-update states
                    old_zL_stats = self._get_state_stats(zL)
                    old_zH_stats = self._get_state_stats(zH)
                    
                    # Low-level reasoning step
                    zL = self.model.L_module(zL, zH, x_emb, cos_sin=cos_sin)
                    
                    # Analyze changes
                    new_zL_stats = self._get_state_stats(zL)
                    zL_change = new_zL_stats['norm'] - old_zL_stats['norm']
                    
                    print(f"  📈 L-step {step+1}: norm {old_zL_stats['norm']:.1f} → {new_zL_stats['norm']:.1f} (Δ{zL_change:+.1f})")
                    print(f"       Mean shift: {old_zL_stats['mean']:.3f} → {new_zL_stats['mean']:.3f}")
                    print(f"       Std change: {old_zL_stats['std']:.3f} → {new_zL_stats['std']:.3f}")
                    
                    cycle_states['steps'].append({
                        'step': step + 1,
                        'zL_change': zL_change,
                        'zL_stats': new_zL_stats
                    })
                
                # High-level update at end of cycle
                old_zH_stats = self._get_state_stats(zH)
                zH = self.model.H_module(zH, zL, cos_sin=cos_sin)
                new_zH_stats = self._get_state_stats(zH)
                zH_change = new_zH_stats['norm'] - old_zH_stats['norm']
                
                print(f"  🎯 H-update: norm {old_zH_stats['norm']:.1f} → {new_zH_stats['norm']:.1f} (Δ{zH_change:+.1f})")
                print(f"       Information integration: {self._measure_integration(zL, zH):.3f}")
                
                cycle_states['zH_change'] = zH_change
                cycle_states['zH_stats'] = new_zH_stats
                cycle_states['integration'] = self._measure_integration(zL, zH)
                
                state_history.append(cycle_states)
        
        print(f"\n✨ FINAL STATE ANALYSIS:")
        self._analyze_state(zL, "Final Low-level")
        self._analyze_state(zH, "Final High-level")
        
        # Show reasoning evolution
        print(f"\n📈 REASONING EVOLUTION:")
        print("Cycle | L-norm  | H-norm  | Integration | L-activity | H-activity")
        print("-" * 65)
        for i, cycle_data in enumerate(state_history):
            l_norm = cycle_data['steps'][-1]['zL_stats']['norm']
            h_norm = cycle_data['zH_stats']['norm']
            integration = cycle_data['integration']
            l_activity = sum(abs(s['zL_change']) for s in cycle_data['steps'])
            h_activity = abs(cycle_data['zH_change'])
            
            print(f"{i+1:5d} | {l_norm:7.1f} | {h_norm:7.1f} | {integration:11.3f} | {l_activity:10.1f} | {h_activity:10.1f}")
        
        return state_history
    
    def _get_state_stats(self, state):
        """Get statistical summary of a state tensor"""
        return {
            'norm': torch.norm(state).item(),
            'mean': torch.mean(state).item(), 
            'std': torch.std(state).item(),
            'max': torch.max(state).item(),
            'min': torch.min(state).item()
        }
    
    def _analyze_state(self, state, name):
        """Print analysis of a state tensor"""
        stats = self._get_state_stats(state)
        print(f"  {name}: norm={stats['norm']:.1f}, mean={stats['mean']:.3f}, std={stats['std']:.3f}, range=[{stats['min']:.1f}, {stats['max']:.1f}]")
    
    def _measure_integration(self, zL, zH):
        """Measure how much low and high level states are integrated"""
        # Simple measure: cosine similarity between mean vectors
        zL_mean = torch.mean(zL, dim=1)  # [batch, hidden]
        zH_mean = torch.mean(zH, dim=1)  # [batch, hidden]
        
        cos_sim = F.cosine_similarity(zL_mean, zH_mean, dim=1)
        return torch.mean(cos_sim).item()
    
    def compare_reasoning_modes(self):
        """Compare how different reasoning configurations affect internal dynamics"""
        
        print(f"\n🔬 COMPARING REASONING MODES")
        print("=" * 60)
        
        # Test input sequence
        test_sequence = [1, 5, 10, 3, 8, 15, 2, 9]  # Arbitrary numbers
        
        modes = [
            (1, 1, "Minimal (1×1)"),
            (2, 1, "Wide (2×1)"), 
            (1, 2, "Deep (1×2)"),
            (2, 2, "Balanced (2×2)"),
            (3, 2, "Extended (3×2)")
        ]
        
        results = []
        
        for N, T, name in modes:
            print(f"\n🧪 Testing {name}")
            print("-" * 30)
            
            history = self.analyze_reasoning_patterns(test_sequence, N=N, T=T)
            
            # Compute summary metrics
            final_L_norm = history[-1]['steps'][-1]['zL_stats']['norm']
            final_H_norm = history[-1]['zH_stats']['norm'] 
            avg_integration = np.mean([c['integration'] for c in history])
            total_L_activity = sum(sum(abs(s['zL_change']) for s in c['steps']) for c in history)
            total_H_activity = sum(abs(c['zH_change']) for c in history)
            
            results.append({
                'name': name,
                'N': N, 'T': T,
                'final_L_norm': final_L_norm,
                'final_H_norm': final_H_norm,
                'avg_integration': avg_integration,
                'total_L_activity': total_L_activity,
                'total_H_activity': total_H_activity
            })
        
        # Summary table
        print(f"\n📊 REASONING MODE COMPARISON")
        print("=" * 80)
        print("Mode        | N×T | L-norm | H-norm | Integration | L-activity | H-activity")
        print("-" * 80)
        
        for r in results:
            print(f"{r['name']:11} | {r['N']}×{r['T']} | {r['final_L_norm']:6.1f} | {r['final_H_norm']:6.1f} | {r['avg_integration']:11.3f} | {r['total_L_activity']:10.1f} | {r['total_H_activity']:10.1f}")
        
        return results
    
    def interactive_demo(self):
        """Interactive demo of the reasoning architecture"""
        
        print(f"\n🎮 INTERACTIVE HRM ARCHITECTURE DEMO")
        print("=" * 60)
        print("This shows the INTERNAL REASONING PROCESS (no text generation)")
        print("Commands:")
        print("  'analyze <numbers>' - Analyze reasoning on number sequence")
        print("  'compare' - Compare different reasoning modes")
        print("  'help' - Show help")
        print("  'quit' - Exit")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 🧠")
                    break
                
                elif user_input.lower() == 'compare':
                    self.compare_reasoning_modes()
                
                elif user_input.lower().startswith('analyze '):
                    numbers_str = user_input[8:].strip()
                    try:
                        # Parse numbers
                        numbers = [int(x) for x in numbers_str.split() if x.isdigit()]
                        if numbers:
                            print(f"Analyzing sequence: {numbers}")
                            self.analyze_reasoning_patterns(numbers, N=2, T=2)
                        else:
                            print("❌ Please provide space-separated numbers")
                    except:
                        print("❌ Error parsing numbers")
                
                elif user_input.lower() == 'help':
                    print("\nExamples:")
                    print("  analyze 1 2 3 4 5    - Analyze sequence [1,2,3,4,5]")
                    print("  compare              - Compare reasoning modes")
                    print("\nThis demo shows:")
                    print("  • How internal states evolve during reasoning")
                    print("  • Differences between low-level and high-level processing") 
                    print("  • How N (cycles) and T (steps) affect the dynamics")
                    print("  • Integration between reasoning levels")
                
                else:
                    print("❌ Unknown command. Type 'help' for options.")
                    
            except KeyboardInterrupt:
                print("\n\nBye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    print("🚀 HRM Architecture Demo")
    print("This demonstrates the REASONING ARCHITECTURE of HRM")
    print("(Not trained for language - shows internal dynamics only)")
    
    try:
        demo = HRMArchitectureDemo()
        demo.interactive_demo()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()