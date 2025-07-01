#!/usr/bin/env python3
"""
Quick check script to verify Ollama is running and has required model
Run before starting the backend
"""

import requests
import json

def check_ollama():
    """Check if Ollama is running and has the required model"""
    ollama_host = "http://localhost:11434"
    required_model = "mxbai-embed-large"
    
    print("🔍 Checking Ollama status...")
    
    # Check if Ollama is running
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            
            # Check if required model is available
            models = response.json()
            model_names = [model["name"].split(":")[0] for model in models.get("models", [])]
            
            if required_model in model_names:
                print(f"✅ Required model '{required_model}' is available")
                print("🚀 Ready to start backend!")
                return True
            else:
                print(f"❌ Required model '{required_model}' not found")
                print(f"📥 Run: ollama pull {required_model}")
                print(f"Available models: {model_names}")
                return False
                
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama")
        print("🔧 Start Ollama with: ollama serve")
        return False
    except requests.exceptions.Timeout:
        print("❌ Ollama connection timeout")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("AI War Room Backend - Ollama Check")
    print("=" * 50)
    
    if check_ollama():
        print("\n🎯 All prerequisites met!")
        print("👉 You can now run: python run.py")
    else:
        print("\n🚨 Prerequisites not met!")
        print("👉 Fix the issues above before starting backend")
    
    print("=" * 50) 