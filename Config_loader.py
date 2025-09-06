"""
Configuration loader for Football Predictor
Handles YAML config and environment variables
"""

import os
import yaml
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import logging

class ConfigLoader:
    """Thread-safe configuration loader"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from files"""
        # Load environment variables
        load_dotenv()
        
        # Load YAML config
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"Config file {config_path} not found, using defaults")
            self._config = self._get_default_config()
        except yaml.YAMLError as e:
            logging.error(f"Error parsing config file: {e}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'apis': {
                'api_football': {
                    'base_url': 'https://v3.football.api-sports.io',
                    'rate_limit': 100,
                    'timeout': 30
                },
                'sofascore': {
                    'base_url': 'https://api.sofascore.com/api/v1',
                    'timeout': 20
                },
                'openweather': {
                    'base_url': 'https://api.openweathermap.org/data/2.5',
                    'timeout': 15
                },
                'news': {
                    'google_news_rss': 'https://news.google.com/rss/search',
                    'timeout': 20
                }
            },
            'prediction': {
                'confidence_threshold': 60.0,
                'max_sources': 10,
                'min_data_sources': 3,
                'form_games': 10,
                'weights': {
                    'recent_form': 0.25,
                    'head_to_head': 0.15,
                    'home_advantage': 0.12,
                    'player_injuries': 0.18,
                    'xg_stats': 0.15,
                    'market_odds': 0.10,
                    'tactical_factors': 0.05
                }
            },
            'leagues': {
                'premier_league': 39,
                'la_liga': 140,
                'bundesliga': 78,
                'serie_a': 135,
                'ligue_1': 61,
                'champions_league': 2,
                'europa_league': 3
            },
            'fetching': {
                'days_ahead': 7,
                'default_season': 2024,
                'max_matches_per_run': 50,
                'retry_attempts': 3,
                'delay_between_requests': 1.0
            },
            'output': {
                'formats': ['telegram', 'csv', 'json'],
                'save_history': True,
                'cleanup_temp_files': True,
                'telegram': {
                    'send_predictions': True,
                    'send_results': True,
                    'daily_summary': True
                },
                'reports': {
                    'include_plots': True,
                    'include_confidence': True,
                    'include_risks': True
                }
            },
            'logging': {
                'level': 'INFO',
                'max_file_size': '10MB',
                'backup_count': 5,
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'performance': {
                'max_concurrent_requests': 5,
                'memory_limit_mb': 512,
                'cache_expiry_hours': 24,
                'use_compression': True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        if not self._config:
            return default
            
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key from environment variables"""
        key_mapping = {
            'api_football': 'API_FOOTBALL_KEY',
            'openweather': 'OPENWEATHER_KEY',
            'news': 'NEWS_API_KEY',
            'telegram': 'TELEGRAM_BOT_TOKEN'
        }
        
        env_key = key_mapping.get(service)
        if env_key:
            return os.getenv(env_key)
        return None
    
    def get_telegram_config(self) -> Dict[str, str]:
        """Get Telegram configuration"""
        return {
            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID')
        }
    
    def get_database_url(self) -> str:
        """Get database URL"""
        return os.getenv('DATABASE_URL', 'sqlite:///football_predictor.db')
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    
    def get_proxy_settings(self) -> Dict[str, str]:
        """Get proxy settings for network requests"""
        return {
            'http': os.getenv('HTTP_PROXY'),
            'https': os.getenv('HTTPS_PROXY')
        }
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate configuration and return status"""
        validation = {
            'config_loaded': self._config is not None,
            'api_football_key': bool(self.get_api_key('api_football')),
            'telegram_configured': bool(self.get_telegram_config()['bot_token']),
            'weather_key': bool(self.get_api_key('openweather')),
            'news_key': bool(self.get_api_key('news'))
        }
        
        return validation
    
    def reload_config(self):
        """Reload configuration from files"""
        self._load_config()
        logging.info("Configuration reloaded")
    
    def __str__(self) -> str:
        """String representation of config status"""
        validation = self.validate_config()
        status_lines = []
        
        for key, value in validation.items():
            status = "✅" if value else "❌"
            status_lines.append(f"{status} {key.replace('_', ' ').title()}")
        
        return "\n".join(status_lines)
