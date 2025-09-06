"""
Utility functions for Football Predictor
Optimized for Termux environment
"""

import os
import logging
import asyncio
import time
import psutil
import gc
from functools import wraps
from typing import Dict, List, Any, Callable
from datetime import datetime
import json
import glob
import shutil
from pathlib import Path

def setup_logging(name: str) -> logging.Logger:
    """Setup logging with proper formatting"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create storage directory
        os.makedirs('storage', exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(f'storage/{name}.log')
        file_handler.setLevel(logging.INFO)
        
        # Console handler  
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
    
    return logger

def rate_limit(calls: int = 1, period: float = 1.0):
    """Rate limiting decorator"""
    def decorator(func):
        last_called = [0.0]
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = period - elapsed
            
            if left_to_wait > 0:
                await asyncio.sleep(left_to_wait)
                
            last_called[0] = time.time()
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def handle_errors(default_return=None):
    """Error handling decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger = logging.getLogger(func.__module__)
                logger.error(f"Error in {func.__name__}: {e}")
                return default_return
        return wrapper
    return decorator

def memory_monitor(limit_mb: int = 512):
    """Monitor memory usage - important for Termux"""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > limit_mb:
            logging.warning(f"⚠️  Memory usage: {memory_mb:.1f}MB (limit: {limit_mb}MB)")
            gc.collect()  # Force garbage collection
            
        logging.info(f"💾 Memory usage: {memory_mb:.1f}MB")
        
    except Exception as e:
        logging.error(f"Error monitoring memory: {e}")

def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        # Clean matplotlib temp files
        temp_patterns = [
            'temp_*.png',
            'match_*.png', 
            '*.tmp',
            '__pycache__/*'
        ]
        
        for pattern in temp_patterns:
            for file_path in glob.glob(pattern):
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception:
                    pass  # Ignore errors
                    
        logging.info("🧹 Cleaned up temporary files")
        
    except Exception as e:
        logging.error(f"Error cleaning temp files: {e}")

def format_team_name(team_name: str) -> str:
    """Standardize team names"""
    if not team_name:
        return "Unknown Team"
        
    # Common replacements
    replacements = {
        'Manchester United': 'Man United',
        'Manchester City': 'Man City',
        'Tottenham Hotspur': 'Tottenham',
        'Brighton & Hove Albion': 'Brighton',
        'Wolverhampton Wanderers': 'Wolves',
        'Sheffield United': 'Sheffield Utd',
        'West Ham United': 'West Ham'
    }
    
    return replacements.get(team_name, team_name)

def format_probability(prob: float, decimals: int = 1) -> str:
    """Format probability as percentage"""
    return f"{prob * 100:.{decimals}f}%"

def get_match_importance(league: str, teams: tuple) -> float:
    """Calculate match importance score"""
    importance = 0.5  # Base importance
    
    league_lower = league.lower()
    
    # Competition importance
    if 'champions league' in league_lower:
        importance = 1.0
    elif 'europa league' in league_lower:
        importance = 0.9
    elif any(comp in league_lower for comp in ['premier league', 'la liga', 'bundesliga']):
        importance = 0.8
    elif 'cup' in league_lower or 'final' in league_lower:
        importance = 0.9
    
    # Derby/rivalry bonus
    home_team, away_team = teams
    if are_rivals(home_team, away_team):
        importance = min(1.0, importance + 0.2)
    
    return importance

def are_rivals(team1: str, team2: str) -> bool:
    """Check if teams are rivals"""
    rivalry_pairs = [
        ('manchester united', 'manchester city'),
        ('manchester united', 'liverpool'),
        ('arsenal', 'tottenham'),
        ('chelsea', 'arsenal'),
        ('liverpool', 'everton'),
        ('barcelona', 'real madrid'),
        ('ac milan', 'inter milan'),
        ('borussia dortmund', 'bayern munich'),
        ('atletico madrid', 'real madrid')
    ]
    
    team1_lower = team1.lower()
    team2_lower = team2.lower()
    
    for pair in rivalry_pairs:
        if ((pair[0] in team1_lower and pair[1] in team2_lower) or
            (pair[1] in team1_lower and pair[0] in team2_lower)):
            return True
    
    return False

def parse_match_date(date_str: str) -> datetime:
    """Parse various date formats"""
    if not date_str:
        return datetime.now()
    
    # Common formats
    formats = [
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%d/%m/%Y %H:%M',
        '%d/%m/%Y'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str[:len(fmt)], fmt)
        except ValueError:
            continue
    
    # If all else fails, return current time
    logging.warning(f"Could not parse date: {date_str}")
    return datetime.now()

def calculate_confidence_score(factors: Dict[str, Any]) -> float:
    """Calculate prediction confidence based on available factors"""
    base_confidence = 0.5
    
    # Data quality factors
    if factors.get('recent_form_available'):
        base_confidence += 0.1
    if factors.get('head_to_head_available'):
        base_confidence += 0.1
    if factors.get('injury_news_available'):
        base_confidence += 0.1
    if factors.get('weather_data_available'):
        base_confidence += 0.05
    if factors.get('market_odds_available'):
        base_confidence += 0.15
    
    # Model factors
    if factors.get('ml_model_used'):
        base_confidence += 0.2
    
    # Match factors
    prob_spread = factors.get('probability_spread', 0)
    if prob_spread > 0.3:  # Clear favorite
        base_confidence += 0.1
    
    return min(0.95, base_confidence)

def validate_prediction_data(prediction: Dict) -> bool:
    """Validate prediction data structure"""
    required_fields = ['home_win_prob', 'draw_prob', 'away_win_prob']
    
    # Check required fields
    for field in required_fields:
        if field not in prediction:
            return False
        if not isinstance(prediction[field], (int, float)):
            return False
        if prediction[field] < 0 or prediction[field] > 1:
            return False
    
    # Check probabilities sum to ~1
    total_prob = sum(prediction[field] for field in required_fields)
    if abs(total_prob - 1.0) > 0.05:
        return False
    
    return True

def normalize_probabilities(probs: Dict[str, float]) -> Dict[str, float]:
    """Normalize probabilities to sum to 1"""
    total = sum(probs.values())
    
    if total == 0:
        # Equal probabilities
        num_outcomes = len(probs)
        return {key: 1.0 / num_outcomes for key in probs.keys()}
    
    return {key: value / total for key, value in probs.items()}

def create_match_id(home_team: str, away_team: str, date: str) -> str:
    """Create unique match identifier"""
    date_part = date[:10] if date else datetime.now().strftime('%Y-%m-%d')
    
    # Clean team names
    home_clean = ''.join(c for c in home_team if c.isalnum()).lower()
    away_clean = ''.join(c for c in away_team if c.isalnum()).lower()
    
    return f"{date_part}_{home_clean}_vs_{away_clean}"

def load_json_safely(file_path: str, default: Any = None) -> Any:
    """Safely load JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return default

def save_json_safely(data: Any, file_path: str) -> bool:
    """Safely save JSON file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        
        return True
    except Exception as e:
        logging.error(f"Error saving {file_path}: {e}")
        return False

def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging"""
    try:
        return {
            'platform': os.name,
            'python_version': os.sys.version,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.cpu_percent(),
            'disk_usage_mb': shutil.disk_usage('.').used / 1024 / 1024,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {'error': str(e)}

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Retry decorator for async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logging.error(f"All {max_retries} attempts failed")
            
            raise last_exception
        
        return wrapper
    return decorator

class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        logging.info(f"⏱️  {self.name} completed in {format_duration(duration)}")

# Color output for terminal (if supported)
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def colored_text(text: str, color: str) -> str:
    """Add color to text if terminal supports it"""
    if os.getenv('TERM') and 'color' in os.getenv('TERM', '').lower():
        return f"{color}{text}{Colors.ENDC}"
    return text

# Export main functions
__all__ = [
    'setup_logging', 'rate_limit', 'handle_errors', 'memory_monitor',
    'cleanup_temp_files', 'format_team_name', 'format_probability',
    'get_match_importance', 'are_rivals', 'parse_match_date',
    'calculate_confidence_score', 'validate_prediction_data',
    'normalize_probabilities', 'create_match_id', 'load_json_safely',
    'save_json_safely', 'get_system_info', 'format_duration',
    'chunk_list', 'retry_on_failure', 'PerformanceTimer', 'Colors',
    'colored_text'
]
