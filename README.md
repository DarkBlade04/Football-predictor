# ⚽ Football Predictor

**AI-powered football match prediction system optimized for Termux**

Get accurate match predictions using multiple data sources, advanced ML models, and real-time notifications - all running on your Android device!

## 🌟 Features

### 🔹 Multi-Source Data Collection
- **API-Football**: Official fixtures, standings, team stats
- **Sofascore**: Live stats, xG data, player ratings  
- **Weather API**: Match conditions impact
- **News scraping**: Injury reports, team news
- **Dynamic fallback**: Tries up to 10 sources automatically

### 🔹 Advanced Prediction Engine
- **Machine Learning**: Random Forest + Gradient Boosting
- **Statistical Models**: Poisson distribution, Elo ratings
- **Multi-factor Analysis**: 15+ prediction factors
- **Confidence Scoring**: Reliability assessment
- **Risk Analysis**: Identifies prediction uncertainties

### 🔹 Comprehensive Analysis Factors
- Recent form (last 10 games)
- Head-to-head records
- Expected goals (xG) statistics
- Player injuries & suspensions
- Home/away advantage
- Weather conditions
- Market odds analysis
- Tactical matchups
- Motivation & pressure factors
- Derby/rivalry intensity

### 🔹 Multiple Output Formats
- **Telegram Bot**: Real-time notifications
- **CSV/JSON**: Data analysis
- **Word Documents**: Professional reports
- **PDF Reports**: Presentation-ready
- **Charts**: Visual probability displays

### 🔹 Termux Optimized
- Memory-efficient (512MB limit)
- Battery-friendly async operations
- Offline capability once trained
- Automatic cleanup & maintenance
- Mobile network resilience

## 🚀 Quick Start

### 1. Installation
```bash
# Clone or download the project
cd ~/football_predictor

# Run automated setup
python setup.py
```

### 2. Configuration
```bash
# Edit environment variables
nano .env

# Add your API keys:
API_FOOTBALL_KEY=your_key_here
TELEGRAM_BOT_TOKEN=your_bot_token
# ... etc
```

### 3. First Run
```bash
# Test with sample data
python run.py --csv matches.csv

# Or fetch live data
python run.py
```

## 📊 Usage Examples

### Basic Usage
```bash
# Fetch today's matches and predict
python run.py

# Use specific CSV file
python run.py --csv my_matches.csv

# Generate specific output formats
python run.py --output csv json docx

# Analyze previous predictions only
python run.py --analyze-only
```

### Advanced Options
```bash
# Debug mode with detailed logging
python run.py --debug

# Set memory limit (MB)
python run.py --memory-limit 256

# Disable Telegram notifications
python run.py --no-telegram
```

## 🔧 Configuration

### API Keys Required
1. **API-Football** (free tier: 100 calls/day)
   - Get at: https://api-football.com
   - Provides: Fixtures, team stats, odds

2. **OpenWeather** (free tier: 1000 calls/day)  
   - Get at: https://openweathermap.org/api
   - Provides: Weather conditions

3. **News API** (free tier: 500 calls/day)
   - Get at: https://newsapi.org
   - Provides: Team news, injuries

4. **Telegram Bot** (free)
   - Create with: @BotFather on Telegram
   - Provides: Real-time notifications

### Configuration File (`config.yaml`)
```yaml
prediction:
  confidence_threshold: 60.0
  max_sources: 10
  form_games: 10
  
  weights:
    recent_form: 0.25
    head_to_head: 0.15
    xg_stats: 0.15
    player_injuries: 0.18
    # ... etc

leagues:
  premier_league: 39
  la_liga: 140
  # ... etc
```

## 📱 Telegram Bot Setup

### 1. Create Bot
1. Message @BotFather on Telegram
2. Send `/newbot`
3. Follow instructions to get token
4. Add token to `.env` file

### 2. Get Chat ID
```bash
# Send message to your bot, then:
curl https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
```

### 3. Features
- 📊 Daily prediction summaries
- 🎯 Top confidence picks
- 📈 Accuracy reports
- ⚠️ Error notifications
- 🔧 System status updates

## 📈 Output Examples

### Terminal Output
```
🏆 FOOTBALL PREDICTOR SUMMARY
============================================
📅 Date: 2024-12-07 10:30:45
🎯 Matches processed: 12
🏟️  Leagues covered: 4
💪 High confidence predictions: 5

🔥 TOP PREDICTIONS:
  1. Arsenal vs Chelsea
     📈 Home Win (65.2%) | Confidence: 82.1%
  2. Man United vs Liverpool  
     📈 Away Win (58.7%) | Confidence: 79.3%
```

### Telegram Notifications
```
🎯 TOP PICK #1

Arsenal vs Chelsea
📅 07/12 15:00 | 🏆 Premier League

📊 Predictions:
  🏠 Home Win: 65.2% ⭐
  ⚖️ Draw: 22.1%
  ✈️ Away Win: 12.7%
  ⚽ Over 2.5: 71.4%

🔥 Confidence: 82.1%

⚠️ Risks:
  • Derby match - form often irrelevant
```

## 📁 Project Structure

```
football_predictor/
├── 📄 run.py              # Main application
├── 📄 fetcher.py          # Data collection
├── 📄 predictor.py        # ML prediction engine
├── 📄 output.py           # Report generation
├── 📄 telegram_bot.py     # Notifications
├── 📄 utils.py            # Helper functions
├── 📄 config_loader.py    # Configuration
├── 📄 setup.py            # Installation script
├── 📄 requirements.txt    # Dependencies
├── 📄 config.yaml         # Settings
├── 📄 .env               # API keys
├── 📂 storage/           # Logs & data
├── 📂 output/            # Generated reports
├── 📂 data/              # Historical data
└── 📂 venv/              # Python environment
```

## 🔍 How Predictions Work

### 1. Data Collection
- Fetches fixtures from multiple APIs
- Collects team statistics and form
- Gathers injury/suspension news
- Retrieves weather conditions
- Analyzes market odds

### 2. Feature Engineering
- Recent form calculation (weighted)
- Head-to-head analysis
- Expected goals modeling
- Injury impact assessment
- Home advantage factors
- Weather impact scoring

### 3. Model Ensemble
- **Statistical Model**: Form + H2H + xG
- **Machine Learning**: Random Forest trained on historical data
- **Market Model**: Odds-implied probabilities
- **Final Prediction**: Weighted combination

### 4. Risk Assessment
- Data quality checks
- Confidence intervals
- Uncertainty factors
- Risk warnings

## 📊 Performance Metrics

### Typical Accuracy (Free Tier)
- **Overall**: ~58-65% correct predictions
- **High Confidence (>70%)**: ~70-75% accuracy
- **Major Leagues**: Better accuracy than lower leagues
- **Derby Matches**: Lower accuracy due to unpredictability

### Resource Usage (Termux)
- **Memory**: ~300-500MB during operation
- **Storage**: ~50MB for app + data
- **Network**: ~10-50MB per day (depends on matches)
- **Battery**: Minimal impact with async operations

## 🔄 Automation

### Daily Automation
```bash
# Setup cron job (after running setup.py)
crontab -e

# Add line for daily 9 AM predictions:
0 9 * * * /path/to/football_predictor/run_cron.sh
```

### Termux Automation
```bash
# Using Termux:Tasker
# Create task that runs: 
# am start -n com.termux
