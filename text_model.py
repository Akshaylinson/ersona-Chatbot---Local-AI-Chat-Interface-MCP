# check_model.py - Fixed for new GPT4All version
import os
from pathlib import Path

def diagnose_model():
    print("🔍 Diagnosing Meta-Llama-3.1-8B model setup...")
    print("=" * 60)
    
    model_path = "models/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf"
    model_file = Path(model_path)
    
    # Check if file exists
    if model_file.exists():
        file_size_gb = model_file.stat().st_size / (1024**3)
        print(f"✅ Model file found: {model_file.name}")
        print(f"📏 File size: {file_size_gb:.2f} GB")
        print(f"📁 Full path: {model_file.absolute()}")
        
        # Check file size (expected ~4.5-5GB for Q4_0)
        expected_min_size = 4.0  # GB
        expected_max_size = 5.5  # GB
        
        if expected_min_size <= file_size_gb <= expected_max_size:
            print("✅ File size looks correct")
        else:
            print(f"⚠️  File size seems unusual for a Q4_0 model (expected ~4.5GB)")
        
    else:
        print(f"❌ Model file NOT found: {model_path}")
        print("💡 Please check the file path")
        return False
    
    # Check Python environment
    print("\n🐍 Python environment:")
    try:
        import gpt4all
        print("✅ GPT4All package is installed")
        # Check version
        version = getattr(gpt4all, '__version__', 'Unknown')
        print(f"📦 GPT4All version: {version}")
    except ImportError:
        print("❌ GPT4All package not installed")
        print("💡 Run: pip install gpt4all")
        return False
    
    # Check if we can load the model with different methods
    print("\n🔄 Testing model loading with different methods...")
    
    # Method 1: New GPT4All version (requires model_name)
    try:
        from gpt4all import GPT4All
        print("Testing Method 1: New GPT4All syntax...")
        model = GPT4All(model_name=model_file.name, model_path=str(model_file.parent))
        print("✅ Method 1: Model loaded successfully!")
        
        # Test a simple generation
        print("Testing generation...")
        response = model.generate("Hello, please say 'OK' if working.", max_tokens=10)
        print(f"✅ Test response: {response.strip()}")
        return True
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Try with just the file path
    try:
        from gpt4all import GPT4All
        print("Testing Method 2: Direct file path...")
        model = GPT4All(str(model_file))
        print("✅ Method 2: Model loaded successfully!")
        
        # Test generation
        response = model.generate("Hello, test.", max_tokens=5)
        print(f"✅ Test response: {response.strip()}")
        return True
        
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    # Method 3: Try different parameter combinations
    try:
        from gpt4all import GPT4All
        print("Testing Method 3: Alternative syntax...")
        model = GPT4All(model_path=str(model_file))
        print("✅ Method 3: Model loaded successfully!")
        
        response = model.generate("Hello.", max_tokens=5)
        print(f"✅ Test response: {response.strip()}")
        return True
        
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    print("\n❌ All loading methods failed")
    print("\n🔧 Troubleshooting tips:")
    print("1. Try installing a specific GPT4All version:")
    print("   pip install gpt4all==1.0.12  # Older version")
    print("   pip install gpt4all>=2.0.0   # Newer version")
    print("2. Check GPT4All documentation for your version")
    print("3. Ensure the model file is not corrupted")
    
    return False

if __name__ == "__main__":
    diagnose_model()
