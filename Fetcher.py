"""
Advanced Football Data Fetcher with Multi-Source Support
Optimized for Termux environment
"""

import asyncio
import aiohttp
import json
import csv
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional, Any
import os
import time
import random
from bs4 import BeautifulSoup
import re
from dataclasses import dataclass
import gc

from config_loader import ConfigLoader
from utils import setup_logging, rate_limit, handle_errors

@dataclass
class MatchData:
    match_id: str
    home_team: str
    away_team: str
    match_date: str
    league: str
    venue: str
    status: str = "scheduled"
    home_odds: float = 0.0
    draw_odds: float = 0.0
    away_odds: float = 0.0
    
class DataFetcher:
    def __init__(self):
        self.config = ConfigLoader()
        self.logger = setup_logging('fetcher')
        self.session: Optional[aiohttp.ClientSession] = None
        self.sources_attempted = 0
        self.max_sources = self.config.get('prediction.max_sources', 10)
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=5, limit_per_host=2)
        self.session = aiohttp.ClientSession(
            timeout=timeout, 
            connector=connector,
            headers={'User-Agent': 'Football-Predictor/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    @rate_limit(calls=1, period=1.0)
    async def fetch_url(self, url: str, headers: Dict = None) -> Optional[Dict]:
        """Rate-limited URL fetcher with error handling"""
        if not self.session:
            raise RuntimeError("Session not initialized")
            
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.warning(f"HTTP {response.status} for {url}")
                    return None
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout fetching {url}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return None
    
    async def fetch_from_csv(self, csv_path: str = "matches.csv") -> List[Dict]:
        """Load matches from CSV file"""
        matches = []
        
        if not os.path.exists(csv_path):
            self.logger.info(f"CSV file {csv_path} not found")
            return matches
            
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                match = {
                    'match_id': str(row.get('match_id', f"csv_{len(matches)}")),
                    'home_team': str(row.get('home_team', '')),
                    'away_team': str(row.get('away_team', '')),
                    'match_date': str(row.get('match_date', '')),
                    'league': str(row.get('league', 'Unknown')),
                    'venue': str(row.get('venue', 'Unknown')),
                    'status': str(row.get('status', 'scheduled')),
                    'source': 'csv'
                }
                matches.append(match)
                
            self.logger.info(f"Loaded {len(matches)} matches from {csv_path}")
            
        except Exception as e:
            self.logger.error(f"Error reading CSV {csv_path}: {e}")
            
        return matches
    
    async def fetch_api_football(self, leagues: List[int], season: int = None) -> List[Dict]:
        """Fetch from API-Football.com"""
        matches = []
        api_key = os.getenv('API_FOOTBALL_KEY')
        
        if not api_key:
            self.logger.warning("API-Football key not found")
            return matches
            
        if season is None:
            season = datetime.now().year
            
        base_url = self.config.get('apis.api_football.base_url')
        headers = {'X-RapidAPI-Key': api_key}
        
        # Get fixtures for next 7 days
        date_from = datetime.now().strftime('%Y-%m-%d')
        date_to = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        
        for league_id in leagues:
            try:
                url = f"{base_url}/fixtures?league={league_id}&season={season}&from={date_from}&to={date_to}"
                data = await self.fetch_url(url, headers)
                
                if data and 'response' in data:
                    for fixture in data['response'][:20]:  # Limit for free tier
                        match = {
                            'match_id': str(fixture['fixture']['id']),
                            'home_team': fixture['teams']['home']['name'],
                            'away_team': fixture['teams']['away']['name'],
                            'match_date': fixture['fixture']['date'],
                            'league': fixture['league']['name'],
                            'venue': fixture['fixture']['venue']['name'] if fixture['fixture']['venue'] else 'TBD',
                            'status': fixture['fixture']['status']['short'],
                            'source': 'api_football'
                        }
                        matches.append(match)
                        
                await asyncio.sleep(1.1)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Error fetching league {league_id}: {e}")
                continue
                
        self.logger.info(f"Fetched {len(matches)} matches from API-Football")
        return matches
    
    async def fetch_sofascore(self, date: str = None) -> List[Dict]:
        """Fetch from Sofascore API"""
        matches = []
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
        try:
            url = f"https://api.sofascore.com/api/v1/sport/football/scheduled-events/{date}"
            data = await self.fetch_url(url)
            
            if data and 'events' in data:
                for event in data['events'][:15]:  # Limit results
                    match = {
                        'match_id': f"sofa_{event['id']}",
                        'home_team': event['homeTeam']['name'],
                        'away_team': event['awayTeam']['name'],
                        'match_date': datetime.fromtimestamp(event['startTimestamp']).isoformat(),
                        'league': event['tournament']['name'],
                        'venue': event.get('venue', {}).get('stadium', {}).get('name', 'Unknown'),
                        'status': event['status']['description'],
                        'source': 'sofascore'
                    }
                    matches.append(match)
                    
        except Exception as e:
            self.logger.error(f"Error fetching from Sofascore: {e}")
            
        self.logger.info(f"Fetched {len(matches)} matches from Sofascore")
        return matches
    
    async def fetch_weather_data(self, venue: str, date: str) -> Dict:
        """Fetch weather data for match venue and date"""
        weather_key = os.getenv('OPENWEATHER_KEY')
        
        if not weather_key:
            return {}
            
        try:
            # Geocoding API to get coordinates
            geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={venue}&limit=1&appid={weather_key}"
            geo_data = await self.fetch_url(geo_url)
            
            if not geo_data:
                return {}
                
            lat, lon = geo_data[0]['lat'], geo_data[0]['lon']
            
            # Get weather forecast
            weather_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={weather_key}&units=metric"
            weather_data = await self.fetch_url(weather_url)
            
            if weather_data and 'list' in weather_data:
                # Find forecast closest to match date
                match_time = datetime.fromisoformat(date.replace('Z', '+00:00'))
                
                closest_forecast = min(
                    weather_data['list'],
                    key=lambda x: abs(datetime.fromtimestamp(x['dt']) - match_time)
                )
                
                return {
                    'temperature': closest_forecast['main']['temp'],
                    'humidity': closest_forecast['main']['humidity'],
                    'wind_speed': closest_forecast['wind']['speed'],
                    'conditions': closest_forecast['weather'][0]['description'],
                    'precipitation': closest_forecast.get('rain', {}).get('3h', 0)
                }
                
        except Exception as e:
            self.logger.error(f"Error fetching weather for {venue}: {e}")
            
        return {}
    
    async def fetch_news_data(self, team1: str, team2: str) -> List[Dict]:
        """Fetch relevant news for teams"""
        news_key = os.getenv('NEWS_API_KEY')
        news_items = []
        
        if not news_key:
            return news_items
            
        try:
            # Search for team news
            for team in [team1, team2]:
                query = f"{team} football news injuries transfers"
                url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=3&apiKey={news_key}"
                
                data = await self.fetch_url(url)
                
                if data and 'articles' in data:
                    for article in data['articles']:
                        news_items.append({
                            'team': team,
                            'title': article['title'],
                            'description': article.get('description', ''),
                            'url': article['url'],
                            'published_at': article['publishedAt']
                        })
                        
                await asyncio.sleep(0.5)  # Rate limiting
                
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            
        return news_items
    
    async def enrich_match_data(self, matches: List[Dict]) -> List[Dict]:
        """Enrich matches with additional data (weather, news, etc.)"""
        enriched_matches = []
        
        for match in matches:
            try:
                # Add weather data
                if match.get('venue') and match.get('match_date'):
                    weather = await self.fetch_weather_data(match['venue'], match['match_date'])
                    match['weather'] = weather
                
                # Add news data
                news = await self.fetch_news_data(match['home_team'], match['away_team'])
                match['news'] = news
                
                # Initialize prediction fields
                match.update({
                    'predictions': {},
                    'confidence': 0.0,
                    'risks': [],
                    'data_sources': [match.get('source', 'unknown')],
                    'last_updated': datetime.now().isoformat()
                })
                
                enriched_matches.append(match)
                
                # Memory management for Termux
                if len(enriched_matches) % 10 == 0:
                    gc.collect()
                    
            except Exception as e:
                self.logger.error(f"Error enriching match {match.get('match_id')}: {e}")
                enriched_matches.append(match)  # Add without enrichment
                
        return enriched_matches
    
    async def save_to_csv(self, matches: List[Dict], filename: str = "matches.csv"):
        """Save matches to CSV file"""
        if not matches:
            self.logger.warning("No matches to save")
            return
            
        try:
            # Flatten nested data for CSV
            flattened_matches = []
            for match in matches:
                flat_match = match.copy()
                
                # Convert complex fields to JSON strings
                for field in ['weather', 'news', 'predictions', 'risks', 'data_sources']:
                    if field in flat_match:
                        flat_match[field] = json.dumps(flat_match[field])
                        
                flattened_matches.append(flat_match)
            
            df = pd.DataFrame(flattened_matches)
            df.to_csv(filename, index=False)
            
            self.logger.info(f"Saved {len(matches)} matches to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving to CSV: {e}")
    
    async def fetch_match_data(self, csv_path: str = None) -> List[Dict]:
        """Main method to fetch match data from multiple sources"""
        all_matches = []
        
        try:
            # Try CSV first if provided
            if csv_path and os.path.exists(csv_path):
                csv_matches = await self.fetch_from_csv(csv_path)
                if csv_matches:
                    all_matches.extend(csv_matches)
                    self.logger.info(f"Using {len(csv_matches)} matches from CSV")
                    return await self.enrich_match_data(all_matches)
            
            # Fallback to API sources
            self.logger.info("Fetching from API sources...")
            
            # Get configured leagues
            leagues = list(self.config.get('leagues', {}).values())
            
            # Fetch from multiple sources
            api_matches = await self.fetch_api_football(leagues)
            if api_matches:
                all_matches.extend(api_matches)
            
            sofa_matches = await self.fetch_sofascore()
            if sofa_matches:
                # Avoid duplicates
                existing_ids = {m.get('match_id') for m in all_matches}
                new_matches = [m for m in sofa_matches if m.get('match_id') not in existing_ids]
                all_matches.extend(new_matches)
            
            # Remove duplicates based on teams and date
            unique_matches = []
            seen = set()
            
            for match in all_matches:
                key = (match['home_team'], match['away_team'], match['match_date'][:10])
                if key not in seen:
                    seen.add(key)
                    unique_matches.append(match)
            
            self.logger.info(f"Found {len(unique_matches)} unique matches")
            
            # Enrich with additional data
            enriched_matches = await self.enrich_match_data(unique_matches)
            
            # Save to CSV for next time
            await self.save_to_csv(enriched_matches)
            
            return enriched_matches
            
        except Exception as e:
            self.logger.error(f"Error in fetch_match_data: {e}")
            return []

# Standalone function for easy import
async def fetch_matches(csv_path: str = None) -> List[Dict]:
    """Convenience function to fetch match data"""
    async with DataFetcher() as fetcher:
        return await fetcher.fetch_match_data(csv_path)
