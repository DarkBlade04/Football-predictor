"""
Advanced Football Match Predictor
Multi-factor analysis with ML models optimized for Termux
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import json
import pickle
import os
from dataclasses import dataclass
import math

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb

from config_loader import ConfigLoader
from utils import setup_logging
from pmf import PoissonModel, EloRating

@dataclass
class TeamStats:
    """Team statistics container"""
    recent_form: float = 0.0
    goals_for: float = 0.0
    goals_against: float = 0.0
    xg_for: float = 0.0
    xg_against: float = 0.0
    home_advantage: float = 0.0
    injury_impact: float = 0.0
    tactical_score: float = 0.0

@dataclass
class PredictionResult:
    """Prediction result container"""
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    over_2_5_prob: float
    confidence: float
    risks: List[str]
    factors_used: List[str]
    
class AdvancedPredictor:
    def __init__(self):
        self.config = ConfigLoader()
        self.logger = setup_logging('predictor')
        self.weights = self.config.get('prediction.weights', {})
        
        # Initialize models
        self.poisson_model = PoissonModel()
        self.elo_model = EloRating()
        self.ml_models = {}
        self.scaler = StandardScaler()
        self.model_trained = False
        
        # Load or initialize historical data
        self.team_stats = {}
        self.historical_matches = []
        self.load_historical_data()
        
    def load_historical_data(self):
        """Load historical data for training"""
        try:
            if os.path.exists('data/historical_matches.pkl'):
                with open('data/historical_matches.pkl', 'rb') as f:
                    self.historical_matches = pickle.load(f)
                    
            if os.path.exists('data/team_stats.pkl'):
                with open('data/team_stats.pkl', 'rb') as f:
                    self.team_stats = pickle.load(f)
                    
            self.logger.info(f"Loaded {len(self.historical_matches)} historical matches")
            
        except Exception as e:
            self.logger.warning(f"Could not load historical data: {e}")
            
    def save_historical_data(self):
        """Save historical data for future use"""
        try:
            os.makedirs('data', exist_ok=True)
            
            with open('data/historical_matches.pkl', 'wb') as f:
                pickle.dump(self.historical_matches[-1000:], f)  # Keep last 1000 matches
                
            with open('data/team_stats.pkl', 'wb') as f:
                pickle.dump(self.team_stats, f)
                
        except Exception as e:
            self.logger.error(f"Error saving historical data: {e}")
            
    def calculate_recent_form(self, team: str, matches: List[Dict], is_home: bool = True) -> float:
        """Calculate team's recent form (last 10 games)"""
        team_matches = []
        
        # Find recent matches for this team
        for match in matches:
            if match.get('home_team') == team or match.get('away_team') == team:
                if match.get('status') in ['FT', 'AET', 'PEN']:  # Finished matches only
                    team_matches.append(match)
                    
        # Sort by date and take last 10
        team_matches.sort(key=lambda x: x.get('match_date', ''), reverse=True)
        recent_matches = team_matches[:10]
        
        if not recent_matches:
            return 0.5  # Neutral form
            
        points = 0
        games = 0
        
        for match in recent_matches:
            home_score = match.get('home_score', 0)
            away_score = match.get('away_score', 0)
            
            if match.get('home_team') == team:
                if home_score > away_score:
                    points += 3
                elif home_score == away_score:
                    points += 1
            else:  # Away team
                if away_score > home_score:
                    points += 3
                elif home_score == away_score:
                    points += 1
                    
            games += 1
            
        if games == 0:
            return 0.5
            
        form_score = points / (games * 3)  # Normalize to 0-1
        
        # Adjust for home/away if specified
        if is_home:
            form_score *= 1.1  # Small home advantage boost
        else:
            form_score *= 0.95  # Small away disadvantage
            
        return min(1.0, form_score)
    
    def calculate_head_to_head(self, home_team: str, away_team: str, matches: List[Dict]) -> Tuple[float, int]:
        """Calculate head-to-head record"""
        h2h_matches = []
        
        for match in matches:
            if ((match.get('home_team') == home_team and match.get('away_team') == away_team) or
                (match.get('home_team') == away_team and match.get('away_team') == home_team)):
                if match.get('status') in ['FT', 'AET', 'PEN']:
                    h2h_matches.append(match)
        
        if len(h2h_matches) < 3:  # Not enough data
            return 0.5, len(h2h_matches)
            
        home_wins = 0
        total_games = len(h2h_matches)
        
        for match in h2h_matches:
            home_score = match.get('home_score', 0)
            away_score = match.get('away_score', 0)
            
            if match.get('home_team') == home_team:
                if home_score > away_score:
                    home_wins += 1
            else:  # Teams reversed
                if away_score > home_score:
                    home_wins += 1
                    
        return home_wins / total_games, total_games
    
    def assess_injury_impact(self, team: str, news_data: List[Dict]) -> float:
        """Assess impact of injuries based on news data"""
        if not news_data:
            return 0.0
            
        injury_keywords = ['injury', 'injured', 'doubt', 'ruled out', 'suspended', 'unavailable']
        key_player_keywords = ['star', 'captain', 'striker', 'midfielder', 'defender', 'goalkeeper']
        
        impact_score = 0.0
        
        for news_item in news_data:
            if news_item.get('team') == team:
                content = f"{news_item.get('title', '')} {news_item.get('description', '')}".lower()
                
                has_injury = any(keyword in content for keyword in injury_keywords)
                has_key_player = any(keyword in content for keyword in key_player_keywords)
                
                if has_injury:
                    if has_key_player:
                        impact_score += 0.15  # High impact
                    else:
                        impact_score += 0.05  # Medium impact
                        
        return min(0.3, impact_score)  # Cap at 30% impact
    
    def analyze_market_odds(self, match: Dict) -> Dict[str, float]:
        """Analyze market odds for insights"""
        odds = {}
        
        home_odds = match.get('home_odds', 0)
        draw_odds = match.get('draw_odds', 0)
        away_odds = match.get('away_odds', 0)
        
        if all([home_odds, draw_odds, away_odds]):
            # Convert odds to implied probabilities
            total_prob = (1/home_odds) + (1/draw_odds) + (1/away_odds)
            
            odds['home_prob'] = (1/home_odds) / total_prob
            odds['draw_prob'] = (1/draw_odds) / total_prob
            odds['away_prob'] = (1/away_odds) / total_prob
            odds['market_confidence'] = 1 - (1/total_prob)  # Bookmaker margin
        else:
            # Default probabilities if no odds available
            odds = {
                'home_prob': 0.45,
                'draw_prob': 0.25,
                'away_prob': 0.30,
                'market_confidence': 0.0
            }
            
        return odds
    
    def calculate_xg_stats(self, team: str, matches: List[Dict]) -> Dict[str, float]:
        """Calculate expected goals statistics"""
        team_matches = []
        
        # Find matches for this team
        for match in matches:
            if match.get('home_team') == team or match.get('away_team') == team:
                if match.get('status') in ['FT', 'AET', 'PEN']:
                    team_matches.append(match)
        
        if len(team_matches) < 5:
            return {'xg_for': 1.5, 'xg_against': 1.5, 'xg_diff': 0.0}
        
        xg_for = []
        xg_against = []
        
        for match in team_matches[-10:]:  # Last 10 matches
            if match.get('home_team') == team:
                xg_for.append(match.get('home_xg', 1.5))
                xg_against.append(match.get('away_xg', 1.5))
            else:
                xg_for.append(match.get('away_xg', 1.5))
                xg_against.append(match.get('home_xg', 1.5))
        
        avg_xg_for = np.mean(xg_for)
        avg_xg_against = np.mean(xg_against)
        
        return {
            'xg_for': avg_xg_for,
            'xg_against': avg_xg_against,
            'xg_diff': avg_xg_for - avg_xg_against
        }
    
    def assess_tactical_factors(self, home_team: str, away_team: str, match: Dict) -> Dict[str, float]:
        """Assess tactical factors"""
        factors = {
            'motivation': 0.5,  # 0-1 scale
            'pressure': 0.5,
            'rivalry': 0.0,
            'importance': 0.5
        }
        
        # Check competition importance
        league = match.get('league', '').lower()
        if 'champions' in league or 'europa' in league:
            factors['importance'] = 0.8
        elif 'cup' in league or 'final' in league:
            factors['importance'] = 0.9
        
        # Check for derbies/rivalries
        rivalry_pairs = [
            ('manchester united', 'manchester city'),
            ('liverpool', 'everton'),
            ('arsenal', 'tottenham'),
            ('barcelona', 'real madrid'),
            ('ac milan', 'inter milan'),
            ('borussia dortmund', 'bayern munich')
        ]
        
        home_lower = home_team.lower()
        away_lower = away_team.lower()
        
        for team1, team2 in rivalry_pairs:
            if ((team1 in home_lower and team2 in away_lower) or 
                (team2 in home_lower and team1 in away_lower)):
                factors['rivalry'] = 0.3
                factors['motivation'] = 0.8
                break
        
        return factors
    
    def calculate_weather_impact(self, weather: Dict) -> float:
        """Calculate weather impact on match"""
        if not weather:
            return 0.0
        
        impact = 0.0
        
        # Temperature impact
        temp = weather.get('temperature', 20)
        if temp < 5 or temp > 35:
            impact += 0.1
        
        # Wind impact
        wind = weather.get('wind_speed', 0)
        if wind > 15:  # Strong wind
            impact += 0.15
        
        # Rain impact
        precipitation = weather.get('precipitation', 0)
        if precipitation > 5:
            impact += 0.2
        
        # Conditions impact
        conditions = weather.get('conditions', '').lower()
        severe_conditions = ['thunderstorm', 'heavy rain', 'snow', 'fog']
        if any(condition in conditions for condition in severe_conditions):
            impact += 0.25
        
        return min(0.5, impact)  # Cap at 50% impact
    
    def build_feature_vector(self, match: Dict, historical_matches: List[Dict]) -> np.ndarray:
        """Build feature vector for ML model"""
        home_team = match['home_team']
        away_team = match['away_team']
        
        features = []
        
        # Basic team stats
        home_form = self.calculate_recent_form(home_team, historical_matches, True)
        away_form = self.calculate_recent_form(away_team, historical_matches, False)
        
        features.extend([home_form, away_form])
        
        # Head-to-head
        h2h_score, h2h_games = self.calculate_head_to_head(home_team, away_team, historical_matches)
        features.extend([h2h_score, h2h_games / 20.0])  # Normalize games count
        
        # Expected goals
        home_xg = self.calculate_xg_stats(home_team, historical_matches)
        away_xg = self.calculate_xg_stats(away_team, historical_matches)
        
        features.extend([
            home_xg['xg_for'], home_xg['xg_against'],
            away_xg['xg_for'], away_xg['xg_against']
        ])
        
        # Injury impact
        injury_home = self.assess_injury_impact(home_team, match.get('news', []))
        injury_away = self.assess_injury_impact(away_team, match.get('news', []))
        features.extend([injury_home, injury_away])
        
        # Market odds
        market = self.analyze_market_odds(match)
        features.extend([
            market['home_prob'], market['draw_prob'], market['away_prob']
        ])
        
        # Tactical factors
        tactical = self.assess_tactical_factors(home_team, away_team, match)
        features.extend([
            tactical['motivation'], tactical['pressure'], 
            tactical['rivalry'], tactical['importance']
        ])
        
        # Weather impact
        weather_impact = self.calculate_weather_impact(match.get('weather', {}))
        features.append(weather_impact)
        
        # League strength (approximate)
        league_strength = self.get_league_strength(match.get('league', ''))
        features.append(league_strength)
        
        return np.array(features)
    
    def get_league_strength(self, league: str) -> float:
        """Get relative league strength"""
        league_lower = league.lower()
        
        if any(comp in league_lower for comp in ['champions league', 'europa league']):
            return 1.0
        elif any(comp in league_lower for comp in ['premier league', 'la liga', 'bundesliga']):
            return 0.9
        elif any(comp in league_lower for comp in ['serie a', 'ligue 1']):
            return 0.85
        elif 'championship' in league_lower or 'division' in league_lower:
            return 0.7
        else:
            return 0.6  # Lower leagues
    
    def train_ml_models(self, historical_data: List[Dict]):
        """Train machine learning models"""
        if len(historical_data) < 100:
            self.logger.warning("Not enough historical data for ML training")
            return
        
        try:
            X = []
            y_result = []  # 0: Away win, 1: Draw, 2: Home win
            y_goals = []   # Total goals
            
            for match in historical_data:
                if not all(key in match for key in ['home_score', 'away_score']):
                    continue
                
                features = self.build_feature_vector(match, historical_data)
                X.append(features)
                
                home_score = int(match['home_score'])
                away_score = int(match['away_score'])
                
                # Result classification
                if home_score > away_score:
                    y_result.append(2)  # Home win
                elif home_score < away_score:
                    y_result.append(0)  # Away win
                else:
                    y_result.append(1)  # Draw
                
                y_goals.append(home_score + away_score)
            
            if len(X) < 50:
                return
            
            X = np.array(X)
            y_result = np.array(y_result)
            y_goals = np.array(y_goals)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_result_train, y_result_test = train_test_split(
                X_scaled, y_result, test_size=0.2, random_state=42
            )
            
            # Train result prediction model
            self.ml_models['result'] = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=1  # Single thread for Termux
            )
            self.ml_models['result'].fit(X_train, y_result_train)
            
            # Train goals prediction model
            self.ml_models['goals'] = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=6,
                random_state=42
            )
            
            # Convert goals to classes (0-1, 2, 3, 4+)
            y_goals_class = np.clip(y_goals, 0, 4)
            _, _, y_goals_train, y_goals_test = train_test_split(
                X_scaled, y_goals_class, test_size=0.2, random_state=42
            )
            
            self.ml_models['goals'].fit(X_train, y_goals_train)
            
            # Evaluate models
            result_pred = self.ml_models['result'].predict(X_test)
            result_accuracy = accuracy_score(y_result_test, result_pred)
            
            self.logger.info(f"ML Model trained - Result accuracy: {result_accuracy:.3f}")
            self.model_trained = True
            
        except Exception as e:
            self.logger.error(f"Error training ML models: {e}")
    
    def predict_match_ml(self, match: Dict, historical_matches: List[Dict]) -> Dict[str, float]:
        """Predict match using ML models"""
        if not self.model_trained or 'result' not in self.ml_models:
            return self.predict_match_statistical(match, historical_matches)
        
        try:
            features = self.build_feature_vector(match, historical_matches)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get result probabilities
            result_probs = self.ml_models['result'].predict_proba(features_scaled)[0]
            
            # Get goals prediction
            if 'goals' in self.ml_models:
                goals_probs = self.ml_models['goals'].predict_proba(features_scaled)[0]
                over_2_5 = np.sum(goals_probs[3:])  # 3+ goals
            else:
                over_2_5 = 0.5
            
            return {
                'home_win_prob': result_probs[2] if len(result_probs) > 2 else 0.33,
                'draw_prob': result_probs[1] if len(result_probs) > 1 else 0.33,
                'away_win_prob': result_probs[0] if len(result_probs) > 0 else 0.33,
                'over_2_5_prob': over_2_5,
                'confidence': 0.8,
                'method': 'ml'
            }
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {e}")
            return self.predict_match_statistical(match, historical_matches)
    
    def predict_match_statistical(self, match: Dict, historical_matches: List[Dict]) -> Dict[str, float]:
        """Fallback statistical prediction method"""
        home_team = match['home_team']
        away_team = match['away_team']
        
        # Calculate factors
        home_form = self.calculate_recent_form(home_team, historical_matches, True)
        away_form = self.calculate_recent_form(away_team, historical_matches, False)
        
        h2h_score, _ = self.calculate_head_to_head(home_team, away_team, historical_matches)
        
        home_xg = self.calculate_xg_stats(home_team, historical_matches)
        away_xg = self.calculate_xg_stats(away_team, historical_matches)
        
        injury_impact_home = self.assess_injury_impact(home_team, match.get('news', []))
        injury_impact_away = self.assess_injury_impact(away_team, match.get('news', []))
        
        weather_impact = self.calculate_weather_impact(match.get('weather', {}))
        
        # Calculate base probabilities
        home_strength = (
            home_form * 0.3 + 
            h2h_score * 0.2 + 
            home_xg['xg_diff'] * 0.2 + 
            (1 - injury_impact_home) * 0.15 +
            0.55  # Home advantage base
        )
        
        away_strength = (
            away_form * 0.3 + 
            (1 - h2h_score) * 0.2 + 
            away_xg['xg_diff'] * 0.2 + 
            (1 - injury_impact_away) * 0.15 +
            0.45  # Away base
        )
        
        # Adjust for weather
        if weather_impact > 0.2:
            home_strength += 0.05  # Home team slight advantage in bad weather
            away_strength -= 0.05
        
        # Normalize probabilities
        total_strength = home_strength + away_strength
        
        if total_strength > 0:
            home_prob = home_strength / total_strength
            away_prob = away_strength / total_strength
        else:
            home_prob = 0.45
            away_prob = 0.35
        
        # Draw probability (inverse relationship with strength difference)
        strength_diff = abs(home_strength - away_strength)
        draw_prob = max(0.15, 0.35 - strength_diff * 0.5)
        
        # Normalize all probabilities
        total_prob = home_prob + draw_prob + away_prob
        home_prob /= total_prob
        draw_prob /= total_prob
        away_prob /= total_prob
        
        # Over 2.5 goals prediction
        avg_goals = (home_xg['xg_for'] + away_xg['xg_for']) / 2
        over_2_5_prob = 1 / (1 + math.exp(-(avg_goals - 2.5)))  # Sigmoid function
        
        confidence = min(0.9, 0.5 + abs(home_prob - away_prob))
        
        return {
            'home_win_prob': home_prob,
            'draw_prob': draw_prob,
            'away_win_prob': away_prob,
            'over_2_5_prob': over_2_5_prob,
            'confidence': confidence,
            'method': 'statistical'
        }
    
    def assess_prediction_risks(self, match: Dict, prediction: Dict) -> List[str]:
        """Assess risks in prediction"""
        risks = []
        
        # Low confidence
        if prediction['confidence'] < 0.6:
            risks.append("Low confidence prediction")
        
        # Missing key data
        if not match.get('news'):
            risks.append("No injury/team news available")
        
        if not match.get('weather'):
            risks.append("No weather data available")
        
        # Close probabilities
        probs = [prediction['home_win_prob'], prediction['draw_prob'], prediction['away_win_prob']]
        if max(probs) - min(probs) < 0.2:
            risks.append("Very close match - unpredictable")
        
        # Weather conditions
        weather = match.get('weather', {})
        if weather.get('precipitation', 0) > 10:
            risks.append("Heavy rain expected - affects play style")
        
        if weather.get('wind_speed', 0) > 20:
            risks.append("Strong winds expected - affects passing game")
        
        # Derby/rivalry
        if any('rivalry' in str(match.get('news', [])).lower() for _ in [1]):
            risks.append("Derby match - form often irrelevant")
        
        return risks
    
    def predict_match(self, match: Dict, historical_matches: List[Dict] = None) -> PredictionResult:
        """Main prediction method"""
        if historical_matches is None:
            historical_matches = self.historical_matches
        
        try:
            # Train models if we have enough data
            if not self.model_trained and len(historical_matches) > 100:
                self.train_ml_models(historical_matches)
            
            # Get prediction
            if self.model_trained:
                prediction = self.predict_match_ml(match, historical_matches)
            else:
                prediction = self.predict_match_statistical(match, historical_matches)
            
            # Assess risks
            risks = self.assess_prediction_risks(match, prediction)
            
            # Determine factors used
            factors_used = ['recent_form', 'head_to_head', 'xg_stats']
            if match.get('news'):
                factors_used.append('injury_news')
            if match.get('weather'):
                factors_used.append('weather')
            if prediction.get('method') == 'ml':
                factors_used.append('machine_learning')
            
            return PredictionResult(
                home_win_prob=round(prediction['home_win_prob'], 3),
                draw_prob=round(prediction['draw_prob'], 3),
                away_win_prob=round(prediction['away_win_prob'], 3),
                over_2_5_prob=round(prediction['over_2_5_prob'], 3),
                confidence=round(prediction['confidence'], 2),
                risks=risks,
                factors_used=factors_used
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting match {match.get('match_id')}: {e}")
            
            # Return default prediction
            return PredictionResult(
                home_win_prob=0.45,
                draw_prob=0.25,
                away_win_prob=0.30,
                over_2_5_prob=0.50,
                confidence=0.30,
                risks=["Prediction failed - using defaults"],
                factors_used=["default"]
            )
    
    def update_historical_data(self, finished_matches: List[Dict]):
        """Update historical data with finished matches"""
        for match in finished_matches:
            if match.get('status') in ['FT', 'AET', 'PEN'] and 'home_score' in match:
                self.historical_matches.append(match)
        
        # Keep only recent matches (last 2 years)
        cutoff_date = datetime.now() - timedelta(days=730)
        self.historical_matches = [
            m for m in self.historical_matches 
            if datetime.fromisoformat(m.get('match_date', '1900-01-01')[:19]) > cutoff_date
        ]
        
        # Retrain models if we have new data
        if len(finished_matches) > 10:
            self.model_trained = False
        
        self.save_historical_data()
        self.logger.info(f"Updated historical data: {len(self.historical_matches)} matches")

# Convenience function
def predict_matches(matches: List[Dict], historical_data: List[Dict] = None) -> List[Dict]:
    """Predict multiple matches"""
    predictor = AdvancedPredictor()
    
    predicted_matches = []
    for match in matches:
        try:
            prediction = predictor.predict_match(match, historical_data)
            
            # Add predictions to match data
            match['predictions'] = {
                'home_win': prediction.home_win_prob,
                'draw': prediction.draw_prob,
                'away_win': prediction.away_win_prob,
                'over_2_5': prediction.over_2_5_prob
            }
            match['confidence'] = prediction.confidence
            match['risks'] = prediction.risks
            match['factors_used'] = prediction.factors_used
            
            predicted_matches.append(match)
            
        except Exception as e:
            logging.error(f"Error predicting match {match.get('match_id')}: {e}")
            predicted_matches.append(match)
    
    return predicted_matches
