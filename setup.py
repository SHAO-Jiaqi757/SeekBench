#!/usr/bin/env python3
"""
Setup script for the Search Evaluation Toolkit
"""

import os
import sys
import subprocess

def create_directories():
    """Create necessary directories"""
    dirs = [
        'data',
        'eval_results',
        'annotated_results',
        'analysis_results',
        'logs'
    ]
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created directory: {dir_name}")

def install_dependencies():
    """Install Python dependencies"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    return True

def create_env_template():
    """Create .env template file"""
    env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE=https://api.openai.com/v1

# Optional: Custom model endpoints
# CUSTOM_MODEL_ENDPOINT=https://your-custom-endpoint.com

# GPU Configuration (optional)
# CUDA_VISIBLE_DEVICES=0,1,2,3
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("Created .env template file")
    else:
        print(".env file already exists")

def make_scripts_executable():
    """Make shell scripts executable"""
    scripts = [
        'scripts/run_generate_traces.sh',
        'scripts/run_rerun_traces.sh'
    ]
    
    for script in scripts:
        if os.path.exists(script):
            os.chmod(script, 0o755)
            print(f"Made {script} executable")

def main():
    """Main setup function"""
    print("Setting up Search Evaluation Toolkit...")
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("Warning: Some dependencies may not have installed correctly")
    
    # Create environment template
    create_env_template()
    
    # Make scripts executable
    make_scripts_executable()
    
    print("\nSetup complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Place your data files in the data/ directory")
    print("3. Run evaluations using scripts/run_generate_traces.sh")
    print("4. Annotate results using annotation/main.py")
    print("5. Analyze results using the scripts in analysis/")

if __name__ == "__main__":
    main()
