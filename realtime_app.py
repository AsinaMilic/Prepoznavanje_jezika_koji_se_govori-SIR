#!/usr/bin/env python
"""
Real-time Language Recognition Application

Standalone launcher for the real-time language recognition GUI.
"""

import argparse
import sys
from src.realtime_gui import RealtimeLanguageGUI


def main():
    """Main entry point for the real-time language recognition application."""
    parser = argparse.ArgumentParser(
        description='Real-time Language Recognition using Whisper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Launch with default model (base)
  python realtime_app.py

  # Launch with specific Whisper model size
  python realtime_app.py --model-size small
  
  # Available model sizes: tiny, base, small, medium, large
        """
    )
    
    parser.add_argument(
        '--model-size',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size (default: base)'
    )
    
    args = parser.parse_args()
    
    # Update config with model size if specified
    if args.model_size != 'base':
        try:
            import yaml
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            if 'whisper' not in config:
                config['whisper'] = {}
            
            config['whisper']['model_size'] = args.model_size
            
            with open('config.yaml', 'w') as f:
                yaml.dump(config, f)
                
            print(f"Using Whisper model size: {args.model_size}")
        except Exception as e:
            print(f"Warning: Could not update config.yaml: {e}")
            print(f"Proceeding with model size: {args.model_size}")
    
    # Create and run GUI application
    try:
        app = RealtimeLanguageGUI()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
