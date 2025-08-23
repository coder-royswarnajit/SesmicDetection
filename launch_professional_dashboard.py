#!/usr/bin/env python3
"""
Professional Dashboard Launcher
Launch the production-ready seismic detection dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'plotly', 
        'pandas',
        'numpy',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'scikit-learn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Main launcher function"""
    
    print("🌍 Professional Seismic Detection Dashboard")
    print("=" * 60)
    print("Production-ready seismic analysis platform")
    print("Showcasing fully functional features only")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("professional_dashboard.py").exists():
        print("❌ professional_dashboard.py not found in current directory!")
        print("   Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ All dependencies found!")
    
    # Launch dashboard
    print("\n🚀 Launching Professional Seismic Detection Dashboard...")
    print("   Features: Complete workflows only")
    print("   Quality: Production-ready")
    print("   Demo: Synthetic data included")
    print("   Dashboard will open in your default web browser")
    print("   Press Ctrl+C to stop the dashboard")
    print("\n" + "=" * 60)
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "professional_dashboard.py",
            "--server.port", "8502",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light",
            "--theme.primaryColor", "#4ECDC4"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Professional dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
