#!/usr/bin/env python3
"""
Football Predictor - Main Runner
Optimized for Termux environment
"""

import asyncio
import argparse
import sys
import os
import gc
import signal
from datetime import datetime
import json
import logging
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_loader import ConfigLoader
from fetcher import DataFetcher
from predictor import AdvancedPredictor
from output import OutputGenerator
from telegram_bot import TelegramNotifier
from utils import setup_logging, memory_monitor, cleanup_temp_files
from scoring import ResultsAnalyzer

class FootballPredictor:
    def __init__(self):
        self.config = ConfigLoader()
        self.logger = setup_logging('main')
        self.predictor = AdvancedPredictor()
        self.output_generator = OutputGenerator()
        self.telegram_notifier = None
        self.results_analyzer = ResultsAnalyzer()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Initialize Telegram if configured
        if os.getenv('TELEGRAM_BOT_TOKEN'):
            self.telegram_notifier = TelegramNotifier()
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        cleanup_temp_files()
        sys.exit(0)
    
    async def fetch_data(self, csv_path: str = None) -> List[Dict]:
        """Fetch match data from various sources"""
        self.logger.info("🔄 Starting data fetch...")
        
        try:
            async with DataFetcher() as fetcher:
                matches = await fetcher.fetch_match_data(csv_path)
                
            if not matches:
                self.logger.warning("❌ No matches found")
                return []
                
            self.logger.info(f"✅ Found {len(matches)} matches")
            return matches
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching data: {e}")
            return []
    
    def generate_predictions(self, matches: List[Dict]) -> List[Dict]:
        """Generate predictions for all matches"""
        self.logger.info("🧠 Generating predictions...")
        
        predicted_matches = []
        
        for i, match in enumerate(matches, 1):
            try:
                self.logger.info(f"Predicting {i}/{len(matches)}: {match['home_team']} vs {match['away_team']}")
                
                # Generate prediction
                prediction = self.predictor.predict_match(match)
                
                # Update match with predictions
                match.update({
                    'predictions': {
                        'home_win': prediction.home_win_prob,
                        'draw': prediction.draw_prob,
                        'away_win': prediction.away_win_prob,
                        'over_2_5': prediction.over_2_5_prob
                    },
                    'confidence': prediction.confidence,
                    'risks': prediction.risks,
                    'factors_used': prediction.factors_used,
                    'prediction_time': datetime.now().isoformat()
                })
                
                predicted_matches.append(match)
                
                # Memory management for Termux
                if i % 5 == 0:
                    gc.collect()
                    
            except Exception as e:
                self.logger.error(f"❌ Error predicting match {match.get('match_id')}: {e}")
                # Add match without predictions
                match.update({
                    'predictions': {'home_win': 0.33, 'draw': 0.33, 'away_win': 0.33, 'over_2_5': 0.5},
                    'confidence': 0.0,
                    'risks': ['Prediction failed'],
                    'factors_used': ['error']
                })
                predicted_matches.append(match)
        
        self.logger.info(f"✅ Generated predictions for {len(predicted_matches)} matches")
        return predicted_matches
    
    async def send_notifications(self, matches: List[Dict]):
        """Send predictions via Telegram"""
        if not self.telegram_notifier:
            self.logger.info("📱 No Telegram bot configured")
            return
            
        self.logger.info("📱 Sending Telegram notifications...")
        
        try:
            await self.telegram_notifier.send_daily_predictions(matches)
            self.logger.info("✅ Telegram notifications sent")
            
        except Exception as e:
            self.logger.error(f"❌ Error sending Telegram notifications: {e}")
    
    def generate_outputs(self, matches: List[Dict], output_formats: List[str] = None):
        """Generate output files"""
        if output_formats is None:
            output_formats = self.config.get('output.formats', ['csv', 'json'])
            
        self.logger.info(f"📄 Generating outputs: {', '.join(output_formats)}")
        
        try:
            # Generate requested formats
            for format_type in output_formats:
                if format_type == 'csv':
                    self.output_generator.save_to_csv(matches)
                elif format_type == 'json':
                    self.output_generator.save_to_json(matches)
                elif format_type == 'docx':
                    self.output_generator.generate_word_report(matches)
                elif format_type == 'pdf':
                    self.output_generator.generate_pdf_report(matches)
                    
            self.logger.info("✅ Output files generated")
            
        except Exception as e:
            self.logger.error(f"❌ Error generating outputs: {e}")
    
    def save_results_history(self, matches: List[Dict]):
        """Save predictions for future analysis"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"storage/predictions_{timestamp}.json"
            
            os.makedirs('storage', exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(matches, f, indent=2, default=str)
                
            self.logger.info(f"💾 Saved predictions to {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving results: {e}")
    
    async def analyze_previous_results(self):
        """Analyze previous predictions against actual results"""
        self.logger.info("📊 Analyzing previous predictions...")
        
        try:
            accuracy = await self.results_analyzer.analyze_accuracy()
            if accuracy:
                self.logger.info(f"📈 Model accuracy: {accuracy['overall']:.1%}")
                
                if self.telegram_notifier:
                    await self.telegram_notifier.send_accuracy_report(accuracy)
                    
        except Exception as e:
            self.logger.error(f"❌ Error analyzing results: {e}")
    
    def display_summary(self, matches: List[Dict]):
        """Display execution summary"""
        if not matches:
            return
            
        print("\n" + "="*60)
        print("🏆 FOOTBALL PREDICTOR SUMMARY")
        print("="*60)
        
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Matches processed: {len(matches)}")
        
        # Count by league
        leagues = {}
        high_confidence = 0
        
        for match in matches:
            league = match.get('league', 'Unknown')
            leagues[league] = leagues.get(league, 0) + 1
            
            if match.get('confidence', 0) > 0.7:
                high_confidence += 1
        
        print(f"🏟️  Leagues covered: {len(leagues)}")
        print(f"💪 High confidence predictions: {high_confidence}")
        
        # Top predictions
        print(f"\n🔥 TOP PREDICTIONS:")
        sorted_matches = sorted(matches, key=lambda x: x.get('confidence', 0), reverse=True)
        
        for i, match in enumerate(sorted_matches[:3], 1):
            pred = match.get('predictions', {})
            confidence = match.get('confidence', 0)
            
            # Determine most likely outcome
            outcomes = [
                ('Home Win', pred.get('home_win', 0)),
                ('Draw', pred.get('draw', 0)),
                ('Away Win', pred.get('away_win', 0))
            ]
            best_outcome = max(outcomes, key=lambda x: x[1])
            
            print(f"  {i}. {match['home_team']} vs {match['away_team']}")
            print(f"     📈 {best_outcome[0]} ({best_outcome[1]:.1%}) | Confidence: {confidence:.1%}")
            
        print(f"\n📁 Files saved in current directory")
        if os.getenv('TELEGRAM_BOT_TOKEN'):
            print(f"📱 Notifications sent to Telegram")
            
        print("="*60)
    
    async def run_full_cycle(self, csv_path: str = None, output_formats: List[str] = None):
        """Run complete prediction cycle"""
        start_time = datetime.now()
        self.logger.info("🚀 Starting Football Predictor")
        
        try:
            # Step 1: Fetch match data
            matches = await self.fetch_data(csv_path)
            if not matches:
                self.logger.error("❌ No matches to process")
                return
            
            # Step
