#!/usr/bin/env python3
"""
Football Predictor Setup Script for Termux
Automated installation and configuration
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import requests
import json

def run_command(command, description=""):
    """Run shell command with error handling"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_termux():
    """Check if running on Termux"""
    return 'com.termux' in os.environ.get('PREFIX', '')

def install_system_packages():
    """Install required system packages"""
    packages = [
        'python',
        'python-pip',
        'git',
        'wget',
        'curl',
        'libxml2',
        'libxslt',
        'libjpeg-turbo',
        'libpng',
        'freetype'
    ]
    
    if check_termux():
        print("🔧 Installing Termux packages...")
        for package in packages:
            run_command(f"pkg install -y {package}", f"Installing {package}")
        
        # Additional Termux-specific packages
        run_command("pkg install -y termux-api", "Installing Termux API")
    else:
        print("⚠️  Not running on Termux, skipping package installation")

def setup_python_environment():
    """Setup Python virtual environment"""
    print("🐍 Setting up Python environment...")
    
    # Upgrade pip
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Create virtual environment
    if not os.path.exists("venv"):
        run_command("python -m venv venv", "Creating virtual environment")
    
    # Activate and install requirements
    if os.path.exists("requirements.txt"):
        if check_termux():
            activate_cmd = "source venv/bin/activate && pip install -r requirements.txt"
        else:
            activate_cmd = "venv/bin/pip install -r requirements.txt"
        
        run_command(activate_cmd, "Installing Python dependencies")
    else:
        print("❌ requirements.txt not found")

def create_directories():
    """Create necessary directories"""
    directories = [
        "storage",
        "output", 
        "data",
        "temp",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def create_config_files():
    """Create configuration files if they don't exist"""
    
    # Create .env file
    if not os.path.exists(".env"):
        env_content = """# Football Predictor Environment Variables
# Replace with your actual API keys

# API-Football (get free key from api-football.com)
API_FOOTBALL_KEY=your_api_football_key_here

# OpenWeather (get free key from openweathermap.org)
OPENWEATHER_KEY=your_openweather_key_here

# News API (get free key from newsapi.org)
NEWS_API_KEY=your_news_api_key_here

# Telegram Bot (create bot with @BotFather)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Database
DATABASE_URL=sqlite:///football_predictor.db

# Debug settings
DEBUG_MODE=false
LOG_LEVEL=INFO
"""
        with open(".env", "w") as f:
            f.write(env_content)
        print("📝 Created .env template")
    
    # Create sample matches.csv
    if not os.path.exists("matches.csv"):
        csv_content = """match_id,home_team,away_team,match_date,league,venue,status
1,Arsenal,Chelsea,2024-12-07 15:00:00,Premier League,Emirates Stadium,scheduled
2,Manchester United,Liverpool,2024-12-07 17:30:00,Premier League,Old Trafford,scheduled
3,Barcelona,Real Madrid,2024-12-08 20:00:00,La Liga,Camp Nou,scheduled
"""
        with open("matches.csv", "w") as f:
            f.write(csv_content)
        print("📝 Created sample matches.csv")

def setup_cron_job():
    """Setup automated execution using Termux cron"""
    if not check_termux():
        print("⚠️  Cron setup only available on Termux")
        return
    
    # Install cronie if not available
    run_command("pkg install -y cronie", "Installing cron")
    
    # Create cron script
    script_content = f"""#!/data/data/com.termux/files/usr/bin/bash
cd {os.getcwd()}
source venv/bin/activate
python run.py > logs/cron.log 2>&1
"""
    
    script_path = "run_cron.sh"
    with open(script_path, "w") as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"📝 Created cron script: {script_path}")
    
    print("\n🕐 To setup daily predictions, run:")
    print("   crontab -e")
    print("   Add line: 0 9 * * * /path/to/football_predictor/run_cron.sh")

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    # Test Python imports
    test_imports = [
        ("aiohttp", "HTTP client"),
        ("pandas", "Data processing"),
        ("numpy", "Numerical computing"),
        ("matplotlib", "Plotting"),
        ("yaml", "Configuration"),
        ("dotenv", "Environment variables")
    ]
    
    failed_imports = []
    
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"✅ {description} ({module})")
        except ImportError:
            print(f"❌ {description} ({module}) - MISSING")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Missing modules: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
    
    # Test configuration
    if os.path.exists(".env"):
        print("✅ Environment file exists")
    else:
        print("❌ Environment file missing")
    
    if os.path.exists("config.yaml"):
        print("✅ Configuration file exists")
    else:
        print("❌ Configuration file missing")
    
    # Test directories
    for directory in ["storage", "output", "data"]:
        if os.path.exists(directory):
            print(f"✅ Directory {directory} exists")
        else:
            print(f"❌ Directory {directory} missing")

def download_sample_data():
    """Download sample historical data"""
    print("📊 Downloading sample data...")
    
    # Create sample historical data
    sample_data = {
        "matches": [
            {
                "match_id": "sample_1",
                "home_team": "Manchester United",
                "away_team": "Liverpool", 
                "home_score": 2,
                "away_score": 1,
                "match_date": "2024-11-01T15:00:00",
                "league": "Premier League",
                "status": "FT"
            }
        ]
    }
    
    with open("data/sample_historical.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print("✅ Created sample historical data")

def show_next_steps():
    """Show next steps to user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 NEXT STEPS:")
    print("1. Edit .env file with your API keys:")
    print("   nano .env")
    
    print("\n2. Get your API keys:")
    print("   • API-Football: https://api-football.com")
    print("   • OpenWeather: https://openweathermap.org/api")
    print("   • News API: https://newsapi.org")
    print("   • Telegram Bot: Chat with @BotFather")
    
    print("\n3. Test the installation:")
    print("   python run.py --csv matches.csv")
    
    print("\n4. Run daily predictions:")
    print("   python run.py")
    
    print("\n5. Setup automation (optional):")
    print("   Setup cron job for daily runs")
    
    print("\n📚 USAGE EXAMPLES:")
    print("   python run.py                    # Fetch and predict")
    print("   python run.py --csv custom.csv   # Use custom CSV")
    print("   python run.py --output csv json  # Specific outputs")
    print("   python run.py --analyze-only     # Just analyze results")
    
    print("\n💡 TIPS:")
    print("   • Start with free API tiers")
    print("   • Check logs in storage/ directory")
    print("   • Use --debug for troubleshooting")
    
    print("\n🆘 SUPPORT:")
    print("   • Check logs in storage/")
    print("   • Run with --debug flag")
    print("   • Ensure all API keys are valid")
    
    print("="*60)

def main():
    """Main setup function"""
    print("🚀 Football Predictor Setup for Termux")
    print("="*50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check if Termux
    if check_termux():
        print("✅ Running on Termux")
    else:
        print("⚠️  Not running on Termux - some features may not work")
    
    try:
        # Run setup steps
        install_system_packages()
        setup_python_environment()
        create_directories()
        create_config_files()
        download_sample_data()
        
        if check_termux():
            setup_cron_job()
        
        test_installation()
        show_next_steps()
        
        print("\n✅ Setup completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
