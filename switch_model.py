#!/usr/bin/env python
"""
Quick script to switch Whisper model size in config.yaml
"""

import sys
import yaml

VALID_MODELS = ['tiny', 'base', 'small', 'medium', 'large']

def switch_model(model_size: str):
    """Switch Whisper model size in config.yaml"""
    if model_size not in VALID_MODELS:
        print(f"Error: Invalid model size '{model_size}'")
        print(f"Valid options: {', '.join(VALID_MODELS)}")
        sys.exit(1)
    
    try:
        # Load config
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Update model size
        if 'whisper' not in config:
            config['whisper'] = {}
        
        old_model = config['whisper'].get('model_size', 'base')
        config['whisper']['model_size'] = model_size
        
        # Save config
        with open('config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✓ Model changed: {old_model} → {model_size}")
        print(f"\nModel characteristics:")
        
        if model_size == 'tiny':
            print("  Speed: ⚡⚡⚡⚡⚡ (fastest)")
            print("  Accuracy: ⭐⭐ (lowest)")
            print("  Memory: ~1 GB")
        elif model_size == 'base':
            print("  Speed: ⚡⚡⚡⚡ (very fast)")
            print("  Accuracy: ⭐⭐⭐ (good)")
            print("  Memory: ~1 GB")
        elif model_size == 'small':
            print("  Speed: ⚡⚡⚡ (fast)")
            print("  Accuracy: ⭐⭐⭐⭐ (very good)")
            print("  Memory: ~2 GB")
        elif model_size == 'medium':
            print("  Speed: ⚡⚡ (moderate)")
            print("  Accuracy: ⭐⭐⭐⭐⭐ (excellent)")
            print("  Memory: ~5 GB")
        elif model_size == 'large':
            print("  Speed: ⚡ (slow)")
            print("  Accuracy: ⭐⭐⭐⭐⭐ (best)")
            print("  Memory: ~10 GB")
        
        print(f"\nRun: python realtime_app.py")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python switch_model.py <model_size>")
        print(f"Valid models: {', '.join(VALID_MODELS)}")
        print("\nExamples:")
        print("  python switch_model.py small")
        print("  python switch_model.py medium")
        sys.exit(1)
    
    switch_model(sys.argv[1])
