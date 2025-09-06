"""
Telegram Bot for Football Predictor notifications
Optimized for Termux environment
"""

import asyncio
import logging
import os
from typing import List, Dict, Optional
from datetime import datetime
import json

try:
    from telegram import Bot, InputFile
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    logging.warning("python-telegram-bot not installed, using HTTP fallback")
    TELEGRAM_AVAILABLE = False
    import aiohttp

from config_loader import ConfigLoader
from utils import setup_logging, format_probability, format_team_name

class TelegramNotifier:
    """Telegram notification handler"""
    
    def __init__(self):
        self.config = ConfigLoader()
        self.logger = setup_logging('telegram')
        
        # Get configuration
        telegram_config = self.config.get_telegram_config()
        self.bot_token = telegram_config['bot_token']
        self.chat_id = telegram_config['chat_id']
        
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN not configured")
        
        if not self.chat_id:
            raise ValueError("TELEGRAM_CHAT_ID not configured")
        
        # Initialize bot
        if TELEGRAM_AVAILABLE:
            self.bot = Bot(token=self.bot_token)
        else:
            self.bot = None
            self.session = None
    
    async def __aenter__(self):
        if not TELEGRAM_AVAILABLE:
            self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not TELEGRAM_AVAILABLE and self.session:
            await self.session.close()
    
    async def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """Send message to Telegram"""
        try:
            if TELEGRAM_AVAILABLE and self.bot:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=parse_mode
                )
            else:
                # HTTP fallback
                await self._send_message_http(message, parse_mode)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}")
            return False
    
    async def _send_message_http(self, message: str, parse_mode: str = 'HTML'):
        """Send message using HTTP API"""
        if not self.session:
            async with aiohttp.ClientSession() as session:
                await self._make_telegram_request(session, message, parse_mode)
        else:
            await self._make_telegram_request(self.session, message, parse_mode)
    
    async def _make_telegram_request(self, session: aiohttp.ClientSession, message: str, parse_mode: str):
        """Make HTTP request to Telegram API"""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        data = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': parse_mode
        }
        
        async with session.post(url, data=data) as response:
            if response.status != 200:
                response_text = await response.text()
                raise Exception(f"Telegram API error: {response.status} - {response_text}")
    
    def format_match_prediction(self, match: Dict) -> str:
        """Format single match prediction for Telegram"""
        home_team = format_team_name(match.get('home_team', 'Unknown'))
        away_team = format_team_name(match.get('away_team', 'Unknown'))
        league = match.get('league', 'Unknown League')
        match_date = match.get('match_date', '')
        
        # Parse date
        try:
            if match_date:
                date_obj = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                formatted_date = date_obj.strftime('%d/%m %H:%M')
            else:
                formatted_date = 'TBD'
        except:
            formatted_date = match_date[:16] if match_date else 'TBD'
        
        predictions = match.get('predictions', {})
        confidence = match.get('confidence', 0.0)
        risks = match.get('risks', [])
        
        # Determine most likely outcome
        outcomes = [
            ('🏠 Home Win', predictions.get('home_win', 0)),
            ('⚖️ Draw', predictions.get('draw', 0)),
            ('✈️ Away Win', predictions.get('away_win', 0))
        ]
        
        best_outcome = max(outcomes, key=lambda x: x[1])
        
        # Build message
        message = f"<b>{home_team} vs {away_team}</b>\n"
        message += f"📅 {formatted_date} | 🏆 {league}\n\n"
        
        # Predictions
        message += f"📊 <b>Predictions:</b>\n"
        for outcome_name, prob in outcomes:
            percentage = format_probability(prob)
            if (outcome_name, prob) == best_outcome:
                message += f"  <b>{outcome_name}: {percentage}</b> ⭐\n"
            else:
                message += f"  {outcome_name}: {percentage}\n"
        
        # Over/Under
        over_25 = predictions.get('over_2_5', 0.5)
        message += f"  ⚽ Over 2.5: {format_probability(over_25)}\n"
        
        # Confidence
        confidence_emoji = "🔥" if confidence > 0.8 else "💪" if confidence > 0.6 else "🤔"
        message += f"\n{confidence_emoji} <b>Confidence: {format_probability(confidence)}</b>\n"
        
        # Risks
        if risks:
            message += f"\n⚠️ <b>Risks:</b>\n"
            for risk in risks[:2]:  # Limit to 2 risks to avoid long messages
                message += f"  • {risk}\n"
        
        return message
    
    def format_daily_summary(self, matches: List[Dict]) -> str:
        """Format daily summary message"""
        if not matches:
            return "📋 No matches found for today"
        
        message = f"⚽ <b>Football Predictions - {datetime.now().strftime('%d/%m/%Y')}</b>\n\n"
        
        # Count by league
        leagues = {}
        high_confidence_matches = 0
        
        for match in matches:
            league = match.get('league', 'Unknown')
            leagues[league] = leagues.get(league, 0) + 1
            
            if match.get('confidence', 0) > 0.7:
                high_confidence_matches += 1
        
        # Summary stats
        message += f"📊 <b>Today's Overview:</b>\n"
        message += f"  🎯 Total Matches: {len(matches)}\n"
        message += f"  🏆 Leagues: {len(leagues)}\n"
        message += f"  🔥 High Confidence: {high_confidence_matches}\n\n"
        
        # Top leagues
        if leagues:
            message += f"🏟️ <b>Leagues:</b>\n"
            sorted_leagues = sorted(leagues.items(), key=lambda x: x[1], reverse=True)
            for league, count in sorted_leagues[:3]:  # Top 3 leagues
                message += f"  • {league}: {count} matches\n"
            
            if len(leagues) > 3:
                message += f"  • +{len(leagues) - 3} more leagues\n"
        
        message += "\n💬 Use /predictions to see detailed predictions"
        
        return message
    
    async def send_daily_predictions(self, matches: List[Dict]):
        """Send daily predictions summary and top matches"""
        if not matches:
            await self.send_message("📋 No matches scheduled for today")
            return
        
        try:
            # Send summary first
            summary = self.format_daily_summary(matches)
            await self.send_message(summary)
            
            await asyncio.sleep(1)  # Rate limiting
            
            # Sort matches by confidence and send top predictions
            sorted_matches = sorted(matches, key=lambda x: x.get('confidence', 0), reverse=True)
            top_matches = sorted_matches[:5]  # Top 5 matches
            
            for i, match in enumerate(top_matches, 1):
                message = f"<b>🎯 TOP PICK #{i}</b>\n\n"
                message += self.format_match_prediction(match)
                
                await self.send_message(message)
                await asyncio.sleep(1)  # Rate limiting
                
            # Send remaining matches in batches
            remaining_matches = sorted_matches[5:]
            if remaining_matches:
                await self.send_message(f"📋 <b>{len(remaining_matches)} More Matches Available</b>")
                
                # Send in groups of 3
                for i in range(0, len(remaining_matches), 3):
                    batch = remaining_matches[i:i+3]
                    
                    batch_message = ""
                    for match in batch:
                        batch_message += self.format_match_prediction(match) + "\n" + "─"*30 + "\n"
                    
                    if batch_message:
                        await self.send_message(batch_message.rstrip("─\n "))
                        await asyncio.sleep(2)
            
        except Exception as e:
            self.logger.error(f"Error sending daily predictions: {e}")
            await self.send_message("❌ Error sending predictions. Check logs for details.")
    
    async def send_accuracy_report(self, accuracy_data: Dict):
        """Send prediction accuracy report"""
        try:
            message = f"📈 <b>Prediction Accuracy Report</b>\n\n"
            
            overall_accuracy = accuracy_data.get('overall', 0)
            message += f"🎯 <b>Overall Accuracy: {format_probability(overall_accuracy)}</b>\n\n"
            
            # Breakdown by outcome
            if 'by_outcome' in accuracy_data:
                message += f"📊 <b>By Outcome:</b>\n"
                for outcome, acc in accuracy_data['by_outcome'].items():
                    message += f"  • {outcome.title()}: {format_probability(acc)}\n"
                message += "\n"
            
            # Recent performance
            if 'recent_games' in accuracy_data:
                recent = accuracy_data['recent_games']
                message += f"🕐 <b>Last {recent.get('count', 0)} predictions:</b>\n"
                message += f"  ✅ Correct: {recent.get('correct', 0)}\n"
                message += f"  ❌ Wrong: {recent.get('wrong', 0)}\n"
                message += f"  📊 Rate: {format_probability(recent.get('accuracy', 0))}\n\n"
            
            # Confidence analysis
            if 'by_confidence' in accuracy_data:
                message += f"💪 <b>By Confidence Level:</b>\n"
                for conf_range, acc in accuracy_data['by_confidence'].items():
                    message += f"  • {conf_range}: {format_probability(acc)}\n"
            
            message += f"\n📅 Report generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
            
            await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending accuracy report: {e}")
    
    async def send_match_result(self, match: Dict, actual_result: Dict):
        """Send match result with prediction comparison"""
        try:
            home_team = format_team_name(match.get('home_team', 'Unknown'))
            away_team = format_team_name(match.get('away_team', 'Unknown'))
            
            home_score = actual_result.get('home_score', 0)
            away_score = actual_result.get('away_score', 0)
            
            # Determine actual outcome
            if home_score > away_score:
                actual_outcome = 'home_win'
                result_emoji = '🏠'
                outcome_text = 'Home Win'
            elif away_score > home_score:
                actual_outcome = 'away_win'
                result_emoji = '✈️'
                outcome_text = 'Away Win'
            else:
                actual_outcome = 'draw'
                result_emoji = '⚖️'
                outcome_text = 'Draw'
            
            # Check prediction accuracy
            predictions = match.get('predictions', {})
            predicted_outcome = max(predictions.items(), key=lambda x: x[1])[0] if predictions else None
            
            prediction_correct = predicted_outcome == actual_outcome
            accuracy_emoji = '✅' if prediction_correct else '❌'
            
            message = f"{accuracy_emoji} <b>Match Result</b>\n\n"
            message += f"<b>{home_team} {home_score} - {away_score} {away_team}</b>\n"
            message += f"{result_emoji} <b>Result: {outcome_text}</b>\n\n"
            
            if predictions:
                message += f"🔮 <b>Our Prediction:</b>\n"
                for outcome, prob in predictions.items():
                    outcome_name = {'home_win': '🏠 Home', 'draw': '⚖️ Draw', 'away_win': '✈️ Away'}.get(outcome, outcome)
                    mark = ' ⭐' if outcome == predicted_outcome else ''
                    message += f"  {outcome_name}: {format_probability(prob)}{mark}\n"
                
                confidence = match.get('confidence', 0)
                message += f"\n💪 Confidence: {format_probability(confidence)}"
                
                if prediction_correct:
                    message += f"\n🎉 <b>Prediction CORRECT!</b>"
                else:
                    message += f"\n😅 Prediction missed this time"
            
            await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending match result: {e}")
    
    async def send_error_notification(self, error_message: str):
        """Send error notification"""
        try:
            message = f"🚨 <b>Error Alert</b>\n\n"
            message += f"❌ {error_message}\n\n"
            message += f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
            
            await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending error notification: {e}")
    
    async def send_system_status(self):
        """Send system status message"""
        try:
            from utils import get_system_info
            
            system_info = get_system_info()
            
            message = f"🔧 <b>System Status</b>\n\n"
            message += f"💾 Memory: {system_info.get('memory_usage_mb', 0):.1f} MB\n"
            message += f"🖥️ CPU: {system_info.get('cpu_percent', 0):.1f}%\n"
            message += f"💿 Disk: {system_info.get('disk_usage_mb', 0):.0f} MB used\n"
            message += f"🐍 Python: {system_info.get('python_version', 'Unknown')[:10]}\n"
            message += f"📱 Platform: {system_info.get('platform', 'Unknown')}\n\n"
            message += f"✅ System running normally\n"
            message += f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
            
            await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending system status: {e}")
    
    async def send_file(self, file_path: str, caption: str = ""):
        """Send file to Telegram"""
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return False
            
            if TELEGRAM_AVAILABLE and self.bot:
                with open(file_path, 'rb') as f:
                    await self.bot.send_document(
                        chat_id=self.chat_id,
                        document=InputFile(f),
                        caption=caption
                    )
            else:
                self.logger.warning("File sending not supported in HTTP fallback mode")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending file: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test Telegram connection"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a task
                task = asyncio.create_task(self.send_message("🧪 Test message - Football Predictor is working!"))
                return True
            else:
                # Run in new event loop
                return asyncio.run(self.send_message("🧪 Test message - Football Predictor is working!"))
        except Exception as e:
            self.logger.error(f"Telegram connection test failed: {e}")
            return False

# Convenience functions
async def send_quick_message(message: str) -> bool:
    """Quick message sender"""
    try:
        async with TelegramNotifier() as notifier:
            return await notifier.send_message(message)
    except Exception as e:
        logging.error(f"Error in quick message: {e}")
        return False

async def notify_predictions(matches: List[Dict]) -> bool:
    """Quick prediction notification"""
    try:
        async with TelegramNotifier() as notifier:
            await notifier.send_daily_predictions(matches)
            return True
    except Exception as e:
        logging.error(f"Error notifying predictions: {e}")
        return False
