import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pytz
import warnings
import os
import textwrap
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr
from PIL import Image # type: ignore
from io import BytesIO # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import matplotlib.offsetbox as offsetbox # type: ignore
import matplotlib.font_manager as fm # type: ignore
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math
from collections import Counter, defaultdict
from plottable import Table # type: ignore
from plottable.plots import image, circled_image # type: ignore
from plottable import ColumnDefinition # type: ignore
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
from typing import Dict, Optional
session = requests.Session()
from matplotlib import font_manager

def scrape_massey_scores(url):
    """
    Scrape baseball scores from Massey Ratings

    Args:
        url: URL to scrape (e.g., https://masseyratings.com/scores.php?s=614639&sub=11606&dt=20250131)

    Returns:
        DataFrame with columns: Date, home_team, away_team, home_score, away_score, location, scheduled
    """

    # Use Selenium to fetch the page (bypasses bot detection)
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(url)
        time.sleep(2)  # Wait for page to load
        page_source = driver.page_source
    finally:
        driver.quit()

    soup = BeautifulSoup(page_source, 'html.parser')
    
    # Find the pre tag containing the scores
    pre_tag = soup.find('pre')
    if not pre_tag:
        return pd.DataFrame()  # Return empty DataFrame if no data
    
    text_content = pre_tag.get_text()
    lines = text_content.split('\n')
    
    games = []
    
    for line in lines:
        # Skip empty lines and the "Games: XX" line
        if not line.strip() or line.strip().startswith('Games:'):
            continue
        
        parts = line.split()
        
        if len(parts) < 5:
            continue
        
        try:
            date = parts[0]
            
            # Find where the @ symbol is to identify teams
            team1_parts = []
            team2_parts = []
            score1 = None
            score2 = None
            location_parts = []
            team1_is_home = False
            team2_is_home = False
            
            i = 1
            # Process first team
            if parts[i].startswith('@'):
                team1_is_home = True
                parts[i] = parts[i][1:]  # Remove @
            
            while i < len(parts):
                if parts[i].isdigit():
                    score1 = int(parts[i])
                    i += 1
                    break
                team1_parts.append(parts[i])
                i += 1
            
            # Process second team
            if i < len(parts):
                if parts[i].startswith('@'):
                    team2_is_home = True
                    parts[i] = parts[i][1:]  # Remove @
                
                while i < len(parts):
                    if parts[i].isdigit():
                        score2 = int(parts[i])
                        i += 1
                        break
                    team2_parts.append(parts[i])
                    i += 1
            
            # Collect location/notes
            while i < len(parts):
                location_parts.append(parts[i])
                i += 1

            team1_str = ' '.join(team1_parts)
            team2_str = ' '.join(team2_parts)

            # Determine location first
            location_str = ' '.join(location_parts).strip()
            # Check if "Sch" appears in the text after the second team name
            scheduled = 'Sch' in location_parts
            # Remove overtime indicators like "O1", "O2", etc.
            location_str = re.sub(r'\s*O\d+\s*$', '', location_str).strip()
            
            # Determine if this is truly a neutral site (has location text that's not just "Sch")
            # Filter out "Sch" and other non-location indicators
            neutral_location_text = ' '.join([part for part in location_parts if part not in ['Sch', 'Scheduled']]).strip()
            is_neutral_site = bool(neutral_location_text) and not team1_is_home and not team2_is_home

            # Assign home/away based on @ symbol or location
            if team1_is_home:
                home_team = team1_str
                away_team = team2_str
                home_score = score1
                away_score = score2
                location = home_team
            elif team2_is_home:
                home_team = team2_str
                away_team = team1_str
                home_score = score2
                away_score = score1
                location = home_team
            elif is_neutral_site:
                # No @ symbol but has a location - this is a neutral site game
                # Treat first team as home for consistency
                home_team = team1_str
                away_team = team2_str
                home_score = score1
                away_score = score2
                location = 'Neutral'
            else:
                # No @ symbol and no location - treat first team as home
                # This is standard convention in score reporting
                home_team = team1_str
                away_team = team2_str
                home_score = score1
                away_score = score2
                location = home_team
            
            games.append({
                'Date': date,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'location': location,
                'scheduled': scheduled
            })
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing line: {line}")
            print(f"Error: {e}")
            continue
    
    return pd.DataFrame(games)

def expand_games_to_team_rows(df):
    """
    Convert game-level data to team-level data where each game becomes two rows

    Args:
        df: DataFrame with columns: Date, home_team, away_team, home_score, away_score, location, scheduled

    Returns:
        DataFrame with columns: Team, Date, Opponent, Location, Result,
                               home_team, away_team, home_score, away_score, scheduled
    """
    
    team_rows = []
    
    for _, row in df.iterrows():
        date = row['Date']
        home_team = row['home_team']
        away_team = row['away_team']
        home_score = row['home_score']
        away_score = row['away_score']
        original_location = row['location']
        scheduled = row.get('scheduled', False)

        # Determine if it's a neutral site game
        # location is either the home_team name or "Neutral"
        is_neutral = original_location == 'Neutral'

        # Create row for home team
        if scheduled:
            home_result = "SCH"
        elif home_score > away_score:
            home_result = f"W{home_score}-{away_score}"
        else:
            home_result = f"L{home_score}-{away_score}"

        home_row = {
            'Team': home_team,
            'Date': date,
            'Opponent': away_team,
            'Location': 'Neutral' if is_neutral else 'Home',
            'Result': home_result,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'scheduled': scheduled
        }
        team_rows.append(home_row)

        # Create row for away team
        if scheduled:
            away_result = "SCH"
        elif away_score > home_score:
            away_result = f"W{away_score}-{home_score}"
        else:
            away_result = f"L{away_score}-{home_score}"

        away_row = {
            'Team': away_team,
            'Date': date,
            'Opponent': home_team,
            'Location': 'Neutral' if is_neutral else 'Away',
            'Result': away_result,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'scheduled': scheduled
        }
        team_rows.append(away_row)
    
    return pd.DataFrame(team_rows)

def create_massey_to_standard_mapping(division):
    """
    Create mapping from Massey Ratings team names to standardized names
    """
    mapping = {
        'Abilene Chr': 'Abilene Christian',
        'Air Force': 'Air Force',
        'Akron': 'Akron',
        'Alabama': 'Alabama',
        'Alabama A&M': 'Alabama A&M',
        'Alabama St': 'Alabama St.',
        'Southern Indiana':'Southern Ind.',
        'La Salle':'La Salle',
        'St Thomas MN':'St. Thomas (MN)',
        'Stonehill':'Stonehill',
        'Le Moyne': 'Le Moyne',
        'Lindenwood':'Lindenwood',
        'West Georgia': 'West Ga.',
        'Mercyhurst':'Mercyhurst',
        'Queens NC':'Queens (NC)',
        # 'Albright': None,  # Not in standard set (D3 school)
        'Alcorn St': 'Alcorn',
        'Appalachian St': 'App State',
        'Arizona': 'Arizona',
        'Arizona St': 'Arizona St.',
        # 'Ark Baptist': None,  # Not in standard set
        'Ark Little Rock': 'Little Rock',
        'Ark Pine Bluff': 'Ark.-Pine Bluff',
        'Arkansas': 'Arkansas',
        'Arkansas St': 'Arkansas St.',
        'Army': 'Army West Point',
        'Auburn': 'Auburn',
        # 'Augustana IL': None,  # Not in standard set (D3 school)
        # 'Aurora': None,  # Not in standard set (D3 school)
        'Austin Peay': 'Austin Peay',
        'BYU': 'BYU',
        # 'Baldwin-Wallace': None,  # Not in standard set (D3 school)
        'Ball St': 'Ball St.',
        'Baylor': 'Baylor',
        'Bellarmine': 'Bellarmine',
        'Belmont': 'Belmont',
        'Bethune-Cookman': 'Bethune-Cookman',
        'Binghamton': 'Binghamton',
        'Boston College': 'Boston College',
        'Bowling Green': 'Bowling Green',
        'Bradley': 'Bradley',
        # 'Brit Columbia': None,  # Not in standard set
        'Brown': 'Brown',
        'Bryant': 'Bryant',
        'Bucknell': 'Bucknell',
        'Butler': 'Butler',
        'C Michigan': 'Central Mich.',
        'CS Bakersfield': 'CSU Bakersfield',
        'CS Fullerton': 'Cal St. Fullerton',
        'CS Northridge': 'CSUN',
        'CS Sacramento': 'Sacramento St.',
        'Cal Baptist': 'California Baptist',
        'Cal Poly': 'Cal Poly',
        'California': 'California',
        'Campbell': 'Campbell',
        'Canisius': 'Canisius',
        # 'Case Western': None,  # Not in standard set (D3 school)
        # 'Cedarville': None,  # Not in standard set (D2 school)
        'Cent Arkansas': 'Central Ark.',
        'Centenary': 'Centenary-LA',  # Not in standard set
        'Le Moyne':'Le Moyne',
        'Central Conn': 'Central Conn. St.',
        # 'Chaminade': None,  # Not in standard set (D2 school)
        'Charleston So': 'Charleston So.',
        'Charlotte': 'Charlotte',
        'Cincinnati': 'Cincinnati',
        'Citadel': 'The Citadel',
        # 'Cleary': None,  # Not in standard set
        'Clemson': 'Clemson',
        'Coastal Car': 'Coastal Carolina',
        'Col Charleston': 'Col. of Charleston',
        'Columbia': 'Columbia',
        # 'Columbia MO': None,  # Not in standard set
        'Connecticut': 'UConn',
        'Coppin St': 'Coppin St.',
        'Cornell': 'Cornell',
        'Creighton': 'Creighton',
        # 'Dakota Wesleyan': None,  # Not in standard set
        'Dallas Bap': 'DBU',
        'Dartmouth': 'Dartmouth',
        # 'Davenport': None,  # Not in standard set (D2 school)
        'Davidson': 'Davidson',
        'Dayton': 'Dayton',
        'Delaware': 'Delaware',
        'Delaware St': 'Delaware St.',
        # 'Dillard': None,  # Not in standard set
        'Duke': 'Duke',
        'E Illinois': 'Eastern Ill.',
        'E Kentucky': 'Eastern Ky.',
        'E Michigan': 'Eastern Mich.',
        'ETSU': 'ETSU',
        'East Carolina': 'East Carolina',
        'Elon': 'Elon',
        # 'Embry-Riddle AZ': None,  # Not in standard set
        'Evansville': 'Evansville',
        'F Dickinson': 'FDU',
        'FGCU': 'FGCU',
        'FL Atlantic': 'Fla. Atlantic',
        'Fairfield': 'Fairfield',
        # 'Findlay': None,  # Not in standard set (D2 school)
        'Florida': 'Florida',
        'Florida A&M': 'Florida A&M',
        'Florida Intl': 'FIU',
        'Florida St': 'Florida St.',
        'Fordham': 'Fordham',
        'Fresno St': 'Fresno St.',
        'G Washington': 'George Washington',
        'Ga Southern': 'Ga. Southern',
        'Gardner Webb': 'Gardner-Webb',
        'George Mason': 'George Mason',
        'Georgetown': 'Georgetown',
        'Georgia': 'Georgia',
        'Georgia St': 'Georgia St.',
        'Georgia Tech': 'Georgia Tech',
        'Gonzaga': 'Gonzaga',
        # 'Grace Chr': None,  # Not in standard set
        'Grambling': 'Grambling',
        'Grand Canyon': 'Grand Canyon',
        'Harvard': 'Harvard',
        'Hawaii': 'Hawaii',
        # 'Hawaii Hilo': None,  # Not in standard set (D2 school)
        # 'Hawaii Pacific': None,  # Not in standard set (D2 school)
        'High Point': 'High Point',
        'Hofstra': 'Hofstra',
        'Holy Cross': 'Holy Cross',
        'Houston': 'Houston',
        'Houston Chr': 'Houston Christian',
        # 'Houston-Victoria': None,  # Not in standard set
        # 'Husson': None,  # Not in standard set (D3 school)
        # 'Huston-Tillot': None,  # Not in standard set
        'IL Chicago': 'UIC',
        # 'IL Wesleyan': None,  # Not in standard set (D3 school)
        'Illinois': 'Illinois',
        'Illinois St': 'Illinois St.',
        'Incarnate Word': 'UIW',
        'Indiana': 'Indiana',
        'Indiana St': 'Indiana St.',
        'Iona': 'Iona',
        'Iowa': 'Iowa',
        'Jackson St': 'Jackson St.',
        'Jacksonville': 'Jacksonville',
        'Jacksonville St': 'Jacksonville St.',
        'James Madison': 'James Madison',
        'Kansas': 'Kansas',
        'Kansas St': 'Kansas St.',
        'Kennesaw': 'Kennesaw St.',
        'Kent': 'Kent St.',
        'Kentucky': 'Kentucky',
        # 'Kentucky St.': None,  # Not in standard set (D2 school)
        'LIU Brooklyn': 'LIU',
        'LSU': 'LSU',
        # 'LSU-Alexandria': None,  # Not in standard set
        'Lafayette': 'Lafayette',
        'Lamar': 'Lamar University',
        # 'Lane': None,  # Not in standard set (D2 school)
        # 'Le Moyne': None,  # Not in standard set
        # 'Le Tourneau': None,  # Not in standard set
        'Lehigh': 'Lehigh',
        'Liberty': 'Liberty',
        # 'Lincoln (PA)': None,  # Not in standard set (D2 school)
        # 'Lindenwood': None,  # Not in standard set
        'Lipscomb': 'Lipscomb',
        'Long Beach St': 'Long Beach St.',
        'Longwood': 'Longwood',
        # 'Loras': None,  # Not in standard set (D3 school)
        'Louisiana': 'Louisiana',
        # 'Louisiana Chr': None,  # Not in standard set
        'Louisiana Tech': 'Louisiana Tech',
        'Louisville': 'Louisville',
        'Loy Marymount': 'LMU (CA)',
        'MA Lowell': 'UMass Lowell',
        'MD E Shore': 'UMES',
        # 'ME Farmington': None,  # Not in standard set (D3 school)
        'MS Valley St': 'Mississippi Val.',
        'MTSU': 'Middle Tenn.',
        'Maine': 'Maine',
        'Manhattan': 'Manhattan',
        'Marist': 'Marist',
        'Marshall': 'Marshall',
        'Maryland': 'Maryland',
        'Massachusetts': 'Massachusetts',
        'McNeese St': 'McNeese',
        'Memphis': 'Memphis',
        'Mercer': 'Mercer',
        # 'Mercyhurst': None,  # Not in standard set
        'Merrimack': 'Merrimack',
        'Miami FL': 'Miami (FL)',
        'Miami OH': 'Miami (OH)',
        'Michigan': 'Michigan',
        'Wm Jessup':'Jessup',
        'Michigan St': 'Michigan St.',
        # 'Miles': None,  # Not in standard set (D2 school)
        # 'Milwaukee Eng': None,  # Not in standard set (D3 school)
        'Minnesota': 'Minnesota',
        'Mississippi': 'Ole Miss',
        'Mississippi St': 'Mississippi St.',
        'Missouri': 'Missouri',
        'Missouri St': 'Missouri St.',
        # 'Mo.-St. Louis': None,  # Not in standard set (D2 school)
        'Monmouth NJ': 'Monmouth',
        'Morehead St': 'Morehead St.',
        "Mt St Mary's": "Mount St. Mary's",
        'Murray St': 'Murray St.',
        'N Colorado': 'Northern Colo.',
        'N Dakota St': 'North Dakota St.',
        'N Illinois': 'NIU',
        'N Kentucky': 'Northern Ky.',
        'NC A&T': 'N.C. A&T',
        'NC State': 'NC State',
        'NE Omaha': 'Omaha',
        'NJIT': 'NJIT',
        'Navy': 'Navy',
        'Nebraska': 'Nebraska',
        'Nevada': 'Nevada',
        # 'New Haven': None,  # Not in standard set (D2 school)
        'New Mexico': 'New Mexico',
        'New Mexico St': 'New Mexico St.',
        'New Orleans': 'New Orleans',
        # 'Newman': None,  # Not in standard set (D2 school)
        'Niagara': 'Niagara',
        'Nicholls St': 'Nicholls',
        'Norfolk St': 'Norfolk St.',
        'North Alabama': 'North Ala.',
        'North Carolina': 'North Carolina',
        'North Florida': 'North Florida',
        # 'North Park': None,  # Not in standard set (D3 school)
        'Northeastern': 'Northeastern',
        'Northwestern': 'Northwestern',
        # 'Northwestern IA': None,  # Not in standard set
        'Northwestern LA': 'Northwestern St.',
        'Notre Dame': 'Notre Dame',
        'Oakland': 'Oakland',
        # 'Oakwood': None,  # Not in standard set
        'Ohio': 'Ohio',
        'Ohio St': 'Ohio St.',
        'Oklahoma': 'Oklahoma',
        'Oklahoma St': 'Oklahoma St.',
        'Old Dominion': 'Old Dominion',
        'Oral Roberts': 'Oral Roberts',
        'Oregon': 'Oregon',
        'Oregon St': 'Oregon St.',
        'PFW': 'Purdue Fort Wayne',
        # 'Pac Lutheran': None,  # Not in standard set (D3 school)
        'Pacific': 'Pacific',
        'Penn': 'Penn',
        'Penn St': 'Penn St.',
        'Pepperdine': 'Pepperdine',
        # 'Pitt.-Johnstown': None,  # Not in standard set (D2 school)
        'Pittsburgh': 'Pittsburgh',
        'Portland': 'Portland',
        'Prairie View': 'Prairie View',
        'Presbyterian': 'Presbyterian',
        'Princeton': 'Princeton',
        'Purdue': 'Purdue',
        # 'Queens NC': None,  # Not in standard set
        'Quinnipiac': 'Quinnipiac',
        'Radford': 'Radford',
        'Rhode Island': 'Rhode Island',
        'Rice': 'Rice',
        'Richmond': 'Richmond',
        'Rider': 'Rider',
        # 'Rockhurst': None,  # Not in standard set (D2 school)
        # 'Rust': None,  # Not in standard set
        'Rutgers': 'Rutgers',
        'S Dakota St': 'South Dakota St.',
        'S Illinois': 'Southern Ill.',
        'SC Upstate': 'USC Upstate',
        'SE Louisiana': 'Southeastern La.',
        'SE Missouri St': 'Southeast Mo. St.',
        'SF Austin': 'SFA',
        'SIUE': 'SIUE',
        'SUNY Albany': 'UAlbany',
        'Sacred Heart': 'Sacred Heart',
        # "Saint Martin's": None,  # Not in standard set (D2 school)
        'Sam Houston St': 'Sam Houston',
        'Samford': 'Samford',
        'San Diego': 'San Diego',
        'San Diego St': 'San Diego St.',
        'San Francisco': 'San Francisco',
        'San Jose St': 'San Jose St.',
        'Santa Clara': 'Santa Clara',
        'Seattle': 'Seattle U',
        'Seton Hall': 'Seton Hall',
        'Siena': 'Siena',
        # 'Simpson CA': None,  # Not in standard set
        'South Alabama': 'South Alabama',
        'South Carolina': 'South Carolina',
        'South Florida': 'South Fla.',
        # 'Southeastern Bap': None,  # Not in standard set
        # 'Southern Indiana': None,  # Not in standard set
        'Southern Miss': 'Southern Miss.',
        # 'Southern N.O.': None,  # Not in standard set
        'Southern Univ': 'Southern U.',
        # 'Southwest Minn. St.': None,  # Not in standard set (D2 school)
        # 'St Ambrose': None,  # Not in standard set
        'St Bonaventure': 'St. Bonaventure',
        "St John's": "St. John's (NY)",
        "St Joseph's PA": "Saint Joseph's",
        'St Louis': 'Saint Louis',
        "St Mary's CA": "Saint Mary's (CA)",
        "St Peter's": "Saint Peter's",
        # 'St Thomas MN': None,  # Not in standard set
        # 'St Xavier IL': None,  # Not in standard set
        'Stanford': 'Stanford',
        'Stetson': 'Stetson',
        # 'Stillman': None,  # Not in standard set
        # 'Stonehill': None,  # Not in standard set
        'Stony Brook': 'Stony Brook',
        'TAM C. Christi': 'A&M-Corpus Christi',
        'TCU': 'TCU',
        'TN Martin': 'UT Martin',
        'TX Southern': 'Texas Southern',
        'Tarleton St': 'Tarleton St.',
        'Tennessee': 'Tennessee',
        'Tennessee Tech': 'Tennessee Tech',
        'Texas': 'Texas',
        'Texas A&M': 'Texas A&M',
        # 'Texas Col': None,  # Not in standard set
        'Texas St': 'Texas St.',
        'Texas Tech': 'Texas Tech',
        # 'Tiffin': None,  # Not in standard set (D2 school)
        'Toledo': 'Toledo',
        # 'Tougaloo': None,  # Not in standard set
        'Towson': 'Towson',
        # 'Trevecca Nazarene': None,  # Not in standard set (D2 school)
        'Troy': 'Troy',
        'Tulane': 'Tulane',
        # 'Tuskegee': None,  # Not in standard set (D2 school)
        'UAB': 'UAB',
        'UC Davis': 'UC Davis',
        'UC Irvine': 'UC Irvine',
        'UC Riverside': 'UC Riverside',
        'UC San Diego': 'UC San Diego',
        'UC Santa Barbara': 'UC Santa Barbara',
        'UCF': 'UCF',
        'UCLA': 'UCLA',
        'ULM': 'ULM',
        'UMBC': 'UMBC',
        'UNC Asheville': 'UNC Asheville',
        'UNC Greensboro': 'UNC Greensboro',
        'UNC Wilmington': 'UNCW',
        'Union Commonwealth': 'Union (TN)',
        'Ferrum':'Ferrum',
        'UNLV': 'UNLV',
        'USC': 'Southern California',
        'UT Arlington': 'UT Arlington',
        'UT San Antonio': 'UTSA',
        'UTRGV': 'UTRGV',
        # 'Union (TN)': None,  # Not in standard set (D2 school)
        'Utah': 'Utah',
        'Utah Tech': 'Utah Tech',
        'Utah Valley': 'Utah Valley',
        'VCU': 'VCU',
        'VMI': 'VMI',
        'Valparaiso': 'Valparaiso',
        'Vanderbilt': 'Vanderbilt',
        'Villanova': 'Villanova',
        'Virginia': 'Virginia',
        'Virginia Tech': 'Virginia Tech',
        'W Carolina': 'Western Caro.',
        'W Illinois': 'Western Ill.',
        'W Michigan': 'Western Mich.',
        'WI Milwaukee': 'Milwaukee',
        # 'WI River Falls': None,  # Not in standard set (D3 school)
        'WKU': 'Western Ky.',
        'Wagner': 'Wagner',
        'Wake Forest': 'Wake Forest',
        # 'Wash & Jeff': None,  # Not in standard set (D3 school)
        'Washington': 'Washington',
        'Washington St': 'Washington St.',
        # 'West Georgia': None,  # Not in standard set
        'West Virginia': 'West Virginia',
        # 'Westcliff': None,  # Not in standard set
        # 'Western Ore.': None,  # Not in standard set (D2 school)
        'Wichita St': 'Wichita St.',
        # 'Wiley': None,  # Not in standard set
        'William & Mary': 'William & Mary',
        'Winthrop': 'Winthrop',
        'Wofford': 'Wofford',
        'Wright St': 'Wright St.',
        'Xavier': 'Xavier',
        # 'Xavier LA': None,  # Not in standard set
        'Yale': 'Yale',
        'Youngstown St': 'Youngstown St.',
        'Chicago St': 'Chicago St.',
        'Cleveland St': 'Cleveland St.',
        'NYIT': 'New York Tech',
        'Savannah St': 'Savannah St.',
        'Northern Iowa': 'UNI',
        'St Thomas MN': 'St. Thomas (MN)',
        'Queens NC': 'Queens (NC)',
        'Southern Indiana': 'Southern Ind.',
        'Ald-Broaddus':'Alderson Broaddus',
        'Concordia CA':'CUI',
        'Queens NC':'Queens (NC)',
        'Southern Indiana':'Southern Ind.',
        'Univ Sciences':'USciences',
        'AL Huntsville': 'UAH',
        'Adams St': 'Adams St.',
        'Adelphi': 'Adelphi',
        'Albany GA': 'Albany St. (GA)',
        'American Intl': "American Int'l",
        'Anderson SC': 'Anderson (SC)',
        'Angelo St': 'Angelo St.',
        'Ark Baptist': 'Ark Baptist',  # Not in standard set
        'Ark Monticello': 'Ark.-Monticello',
        'Arkansas Tech': 'Arkansas Tech',
        'Arkansas-FS': 'Ark.-Fort Smith',
        'Ashland': 'Ashland',
        "Auburn M'gomery": 'AUM',
        'Augusta': 'Augusta',
        'Augustana SD': 'Augustana (SD)',
        'Azusa Pacific': 'Azusa Pacific',
        'Barry': 'Barry',
        'Barton': 'Barton',
        'Belmont Abbey': 'Belmont Abbey',
        'Bemidji St': 'Bemidji St.',
        'Benedict': 'Benedict',
        'Bentley': 'Bentley',
        'Biola': 'Biola',
        'Bloomfield': 'Bloomfield',
        'Bloomsburg': 'Bloomsburg',
        'Bluefield St': 'Bluefield St.',
        'Bridgeport': 'Bridgeport',
        'C Missouri': 'Central Mo.',
        'C Oklahoma': 'Central Okla.',
        'C Washington': 'Central Wash.',
        'CS Chico': 'Chico St.',
        'CS Dom. Hills': 'Cal St. Dom. Hills',
        'CS Monterey Bay': 'Cal St. Monterey Bay',
        'CS Poly Pomona': 'Cal Poly Pomona',
        'CS San Bern.': 'CSUSB',
        'CS San Marcos': 'Cal St. San Marcos',
        'CS Stanislaus': 'Stanislaus St.',
        'CSU East Bay': 'Cal St. East Bay',
        'CSU-Pueblo': 'CSU Pueblo',
        'Cal St-LA': 'Cal State LA',
        'Caldwell': 'Caldwell',
        'California PA': 'California (PA)',
        'Cameron': 'Cameron',
        'Carson-Newman': 'Carson-Newman',
        'Catawba': 'Catawba',
        'Cedarville': 'Cedarville',
        'Chaminade': 'Chaminade',
        'Charleston WV': 'Charleston (WV)',
        'Chestnut Hill': 'Chestnut Hill',
        'Chowan': 'Chowan',
        'Chr Brothers': 'Christian Brothers',
        'Claflin': 'Claflin',
        'Clarion': 'Clarion',
        'Clark Atlanta': 'Clark Atlanta',
        'Coker': 'Coker',
        'Col Springs': 'UCCS',
        'Colorado Chr': 'Colo. Christian',
        'Colorado Mines': 'Colo. Sch. of Mines',
        'CO Mesa': 'Colorado Mesa',
        'Columbus St': 'Columbus St.',
        'Concord': 'Concord',
        'Concordia SP': 'Concordia-St. Paul',
        "D'Youville": "D'Youville",
        'Davenport': 'Davenport',
        'Davis & Elkins': 'Davis & Elkins',
        'Delta St': 'Delta St.',
        'Dominican NY': 'Dominican (NY)',
        'Drury': 'Drury',
        'E New Mexico': 'Eastern N.M.',
        'E Stroudsburg': 'East Stroudsburg',
        'East Central OK': 'East Central',
        'Eckerd': 'Eckerd',
        'Edward Waters': 'Edward Waters',
        'Embry-Riddle FL': 'Embry-Riddle (FL)',
        'Emmanuel GA': 'Emmanuel (GA)',
        'Emory & Henry': 'Emory & Henry',
        'Emporia St': 'Emporia St.',
        'Erskine': 'Erskine',
        'FL Southern': 'Fla. Southern',
        'Fairmont St': 'Fairmont St.',
        'Felician': 'Felician',
        'Findlay': 'Findlay',
        'Flagler': 'Flagler',
        'Florida Tech': 'Florida Tech',
        'Fort Hays St': 'Fort Hays St.',
        'Francis Marion': 'Francis Marion',
        'Franklin Pierce': 'Franklin Pierce',
        'Fresno Pacific': 'Fresno Pacific',
        'Frostburg St': 'Frostburg St.',
        'Gannon': 'Gannon',
        'Georgia C&S': 'Georgia College',
        'Georgia SW': 'Ga. Southwestern',
        'Georgian Court': 'Georgian Court',
        'Glenville St': 'Glenville St.',
        'Goldey Beacom': 'Goldey-Beacom',
        'Grand Valley St': 'Grand Valley St.',
        'Harding': 'Harding',
        'Hawaii Hilo': 'Hawaii Hilo',
        'Hawaii Pacific': 'Hawaii Pacific',
        'Henderson St': 'Henderson St.',
        'Hillsdale': 'Hillsdale',
        'Holy Family': 'Holy Family',
        'IL Springfield': 'Ill. Springfield',
        'Indiana PA': 'Indiana (PA)',
        'Indianapolis': 'UIndy',
        'Kentucky St': 'Kentucky St.',
        'King': 'King (TN)',
        'Kutztown': 'Kutztown',
        'KY Wesleyan': 'Ky. Wesleyan',
        'Lake Erie': 'Lake Erie',
        'Lander': 'Lander',
        'Lane': 'Lane',
        'LeMoyne-Owen': 'LeMoyne-Owen',
        'Lee TN': 'Lee',
        'Lenoir-Rhyne': 'Lenoir-Rhyne',
        'Lewis': 'Lewis',
        'Limestone': 'Limestone',
        'Lincoln MO': 'Lincoln (MO)',
        'Lincoln Mem': 'Lincoln Memorial',
        'Lincoln PA': 'Lincoln (PA)',
        'Lock Haven': 'Lock Haven',
        'Lubbock Chr': 'Lubbock Christian',
        'Lynn': 'Lynn',
        'MN Crookston': 'Minn.-Crookston',
        'MN Duluth': 'Minn. Duluth',
        'MN Mankato': 'Minnesota St.',
        'MO Southern': 'Mo. Southern St.',
        'MO St Louis': 'Mo.-St. Louis',
        'MO Western': 'Missouri Western',
        'Molloy': 'Molloy',
        'MT St-Billings': 'Mont. St. Billings',
        'Malone': 'Malone',
        'Mansfield': 'Mansfield',
        'Mars Hill': 'Mars Hill',
        'Mary ND': 'Mary',
        'Maryville MO': 'Maryville (MO)',
        'McKendree': 'McKendree',
        'Menlo': 'Menlo',
        'Mercy': 'Mercy',
        'Metro St': 'MSU Denver',
        'Miles': 'Miles',
        'Millersville': 'Millersville',
        'Minot St': 'Minot St.',
        'Mississippi Col': 'Mississippi Col.',
        'Missouri S&T': 'Missouri S&T',
        'Montevallo': 'Montevallo',
        'Morehouse': 'Morehouse',
        'Mt Olive': 'Mount Olive',
        'NC Pembroke': 'UNC Pembroke',
        'NE Oklahoma': 'Northeastern St.',
        'NM Highlands': 'N.M. Highlands',
        'NW Missouri': 'Northwest Mo. St.',
        'NW Nazarene': 'Northwest Nazarene',
        'NW Oklahoma St': 'Northwestern Okla.',
        'New Haven': 'New Haven',
        'Newberry': 'Newberry',
        'Newman': 'Newman',
        'North Georgia': 'North Georgia',
        'North Greenville': 'North Greenville',
        'Northern St SD': 'Northern St.',
        'Northwood MI': 'Northwood',
        'Nova SE': 'Nova Southeastern',
        'OH Dominican': 'Ohio Dominican',
        'OK Baptist': 'Okla. Baptist',
        'Oklahoma Chr': 'Okla. Christian',
        'Ouachita': 'Ouachita Baptist',
        'Pace': 'Pace',
        'Palm Beach Atl': 'Palm Beach Atl.',
        'Pitt-Johnstown': 'Pitt.-Johnstown',
        'Pittsburg St': 'Pittsburg St.',
        'Point Loma': 'Point Loma',
        'Post': 'Post',
        'Purdue Northwest': 'Purdue Northwest',
        'Queens NY': 'Queens (NY)',
        'Quincy': 'Quincy',
        'Regis CO': 'Regis (CO)',
        'Rockhurst': 'Rockhurst',
        'Rogers St': 'Rogers St.',
        'Rollins': 'Rollins',
        'S Arkansas': 'Southern Ark.',
        'S Connecticut': 'Southern Conn. St.',
        'S Francisco St': 'San Fran. St.',
        'S New Hampshire': 'Southern N.H.',
        'S Wesleyan': 'Southern Wesleyan',
        'SC Aiken': 'USC Aiken',
        'SC Beaufort': 'SC Beaufort',  # Not in standard set
        'SE Oklahoma': 'Southeastern Okla.',
        'SW Baptist': 'Southwest Baptist',
        'SW Minnesota': 'Southwest Minn. St.',
        'SW Oklahoma': 'Southwestern Okla.',
        'Saginaw Val': 'Saginaw Valley',
        'Salem WV': 'Salem (WV)',
        'Savannah St': 'Savannah St.',
        'Seton Hill': 'Seton Hill',
        'Shepherd': 'Shepherd',
        'Shippensburg': 'Shippensburg',
        'Shorter': 'Shorter',
        'Sioux Falls': 'Sioux Falls',
        'Slippery Rock': 'Slippery Rock',
        'Sonoma St': 'Sonoma St.',
        'Southern N.O.': 'Southern N.O.',  # Not in standard set
        'Southern Nazarene': 'Southern Nazarene',
        'Spring Hill': 'Spring Hill',
        'St Anselm': 'Saint Anselm',
        'St Cloud': 'St. Cloud St.',
        "St Edward's": "St. Edward's",
        'St Leo': 'Saint Leo',
        "St Martin's": "Saint Martin's",
        "St Mary's TX": "St. Mary's (TX)",
        "St Michael's": "Saint Michael's",
        'Staten Island': 'Staten Island',
        'T Jefferson': 'Jefferson',
        "TX A&M K'ville": 'Tex. A&M-Kingsville',
        'Tampa': 'Tampa',
        'Texas A&M Intl': "Tex. A&M Int'l",
        'Thomas Aquinas': 'St. Thomas Aquinas',
        'Tiffin': 'Tiffin',
        'Trevecca Naz': 'Trevecca Nazarene',
        'Truman St': 'Truman St.',
        'Tusculum': 'Tusculum',
        'Tuskegee': 'Tuskegee',
        'UT Tyler': 'UT Tyler',
        'UVA-Wise': 'UVA Wise',
        'Union TN': 'Union (TN)',
        'Upper Iowa': 'Upper Iowa',
        'Valdosta St': 'Valdosta St.',
        'Vanguard': 'CUI',
        'Virginia St': 'Virginia St.',
        'W Oregon': 'Western Ore.',
        'W Texas A&M': 'West Tex. A&M',
        'WI Parkside': 'Wis.-Parkside',
        'WV Wesleyan': 'West Va. Wesleyan',
        'WV State': 'West Virginia St.',
        'Walsh': 'Walsh',
        'Washburn': 'Washburn',
        'Wayne St MI': 'Wayne St. (MI)',
        'Wayne St NE': 'Wayne St. (NE)',
        'West Alabama': 'West Ala.',
        'West Chester': 'West Chester',
        'West Florida': 'West Florida',
        'West Liberty': 'West Liberty',
        'Westmont': 'Westmont',
        'Wheeling': 'Wheeling',
        'Wm Jewell': 'William Jewell',
        'Wilmington DE': 'Wilmington (DE)',
        'Wingate': 'Wingate',
        'Winona St': 'Winona St.',
        'Young Harris': 'Young Harris',
        'Permian Basin': 'UT Permian Basin',
        'Notre Dame OH': 'Notre Dame (OH)',
        'St Rose':'Saint Rose',
        'West Georgia':'West Ga.',
        'Adrian': 'Adrian',
        'Sul Ross':'Sul Ross St.',
        'Albertus Magnus': 'Albertus Magnus',
        'Albion': 'Albion',
        'Albright': 'Albright',
        'Alfred': 'Alfred',
        'Alfred St': 'Alfred St.',
        'Allegheny': 'Allegheny',
        'Alma': 'Alma',
        'Alvernia': 'Alvernia',
        'Amherst': 'Amherst',
        'Anderson IN': 'Anderson (IN)',
        'SC Beaufort': 'USC Beaufort',
        'Anna Maria': 'Anna Maria',
        # 'Apprentice': None,  # Not in standard set
        # 'Aquinas': None,  # Not in standard set
        'Arcadia': 'Arcadia',
        # 'Arlington Bap': None,  # Not in standard set
        'Asbury': 'Asbury',
        'Augsburg': 'Augsburg',
        'Alice Lloyd':'Alice Lloyd (KY)',
        'Houston-Victoria':'A&M-Victoria',
        'Augustana IL': 'Augustana (IL)',
        'Aurora': 'Aurora',
        'Austin Col': 'Austin',
        'Averett': 'Averett',
        'Babson': 'Babson',
        'Baldwin-Wallace': 'Baldwin Wallace',
        'Bard': 'Bard',
        'Baruch': 'Baruch',
        'Bates': 'Bates',
        'Belhaven MS': 'Belhaven',
        'Beloit': 'Beloit',
        'Benedictine IL': 'Benedictine (IL)',
        'Berea': 'Berea',
        'Berry': 'Berry',
        'Bethany Luth': 'Bethany Lutheran',
        'Bethany WV': 'Bethany (WV)',
        'Bethel MN': 'Bethel (MN)',
        # 'Bethesda': None,  # Not in standard set
        'Blackburn': 'Blackburn',
        'Bluffton': 'Bluffton',
        # 'Bob Jones': None,  # Not in standard set
        'Bowdoin': 'Bowdoin',
        'Brandeis': 'Brandeis',
        'Brevard': 'Brevard',
        'Bridgewater MA': 'Bridgewater St.',
        'Bridgewater VA': 'Bridgewater (VA)',
        'Brockport St': 'SUNY Brockport',
        'Buena Vista': 'Buena Vista',
        # 'Bushnell': None,  # Not in standard set
        # 'C Washington': None,  # Not in standard set (might be Central Washington but that's not in D3)
        'Cairn': 'Cairn',
        'Cal Lutheran': 'Cal Lutheran',
        'Caltech': 'Caltech',
        'Calvin': 'Calvin',
        'Capital': 'Capital',
        'Carleton MN': 'Carleton',
        # 'Carolina Univ': None,  # Not in standard set
        'Carroll WI': 'Carroll (WI)',
        'Carthage': 'Carthage',
        'Case Western': 'CWRU',
        'Castleton': 'VTSU Castleton',
        'Catholic': 'Catholic',
        'Centenary': 'Centenary (LA)',
        'Centenary NJ': 'Centenary (NJ)',
        'Central IA': 'Central (IA)',
        # 'Central Penn': None,  # Not in standard set
        'Centre': 'Centre',
        # 'Champion Bap': None,  # Not in standard set
        'Chapman': 'Chapman',
        'Chatham': 'Chatham',
        'Chicago': 'UChicago',
        'Chris Newport': 'Chris. Newport',
        'City Col NY': 'CCNY',
        'Claremont M.S.': 'Claremont-M-S',
        'Clark MA': 'Clark (MA)',
        'Clarkson': 'Clarkson',
        # 'Cleary': None,  # Not in standard set
        'Coast Guard': 'Coast Guard',
        'Coe': 'Coe',
        'Colby': 'Colby',
        'Colby-Sawyer': 'Colby-Sawyer',
        'College of NJ': 'TCNJ',
        'Concordia IL': 'Concordia Chicago',
        'Concordia Mhd': "Concordia-M'head",
        # 'Concordia SP': None,  # Not in standard set (might be St. Paul but unclear)
        'Concordia TX': 'Concordia (TX)',
        'Concordia WI': 'Concordia Wisconsin',
        # 'Corban': None,  # Not in standard set
        'Cornell IA': 'Cornell College',
        'Cortland St': 'Cortland',
        'Covenant': 'Covenant',
        # "Crowley's Ridge": None,  # Not in standard set
        'Crown MN': 'Crown (MN)',
        'Curry': 'Curry',
        # 'Dakota Wesleyan': None,  # Not in standard set
        # 'Dallas Chr': None,  # Not in standard set
        'Dallas Univ': 'Dallas',
        'DePauw': 'DePauw',
        'DeSales': 'DeSales',
        'Dean': 'Dean',
        # 'Defiance': None,  # Not in standard set
        'Delaware Val': 'Delaware Valley',
        'Denison': 'Denison',
        'Dickinson': 'Dickinson',
        'Dominican IL': 'Dominican (IL)',
        'Drew': 'Drew',
        'Dubuque': 'Dubuque',
        'E Connecticut': 'Eastern Conn. St.',
        # 'E Illinois': None,  # Not in standard set
        'E Mennonite': 'East. Mennonite',
        'E Texas Bap': 'East Tex. Baptist',
        'Earlham': 'Earlham',
        # 'Eastern Oregon': None,  # Not in standard set
        'Eastern Univ': 'Eastern',
        'Edgewood': 'Edgewood',
        'Elizabethtown': 'Elizabethtown',
        'Elmhurst': 'Elmhurst',
        'Elmira': 'Elmira',
        'Elms': 'Elms',
        'Emerson': 'Emerson',
        'Emory': 'Emory',
        'Endicott': 'Endicott',
        'Eureka': 'Eureka',
        'FDU Madison': 'FDU-Florham',
        'Farmingdale': 'Farmingdale St.',
        'Ferrum': 'Ferrum',
        'Fisher': 'St. John Fisher',
        'Fitchburg St': 'Fitchburg St.',
        'Fontbonne': 'Fontbonne',
        'Framingham St': 'Framingham St.',
        'Franciscan OH': 'Franciscan',
        'Frank & Marsh': 'Franklin & Marshall',
        'Franklin': 'Franklin',
        'Geneva': 'Geneva',
        'George Fox': 'George Fox',
        'Gettysburg': 'Gettysburg',
        'Gordon': 'Gordon',
        # 'Grace Chr': None,  # Not in standard set
        'Greensboro': 'Greensboro',
        'Greenville': 'Greenville',
        'Grinnell': 'Grinnell',
        'Grove City': 'Grove City',
        'Guilford': 'Guilford',
        'Gust Adolphus': 'Gustavus Adolphus',
        'Gwynedd-Mercy': 'Gwynedd Mercy',
        'Hamilton': 'Hamilton',
        'Hamline': 'Hamline',
        'Hampden-Sydney': 'Hampden-Sydney',
        'Hanover': 'Hanover',
        'Hardin-Simmons': 'Hardin-Simmons',
        # 'Hartford': None,  # Not in standard set
        # 'Hastings': None,  # Not in standard set
        'Haverford': 'Haverford',
        'Heidelberg': 'Heidelberg',
        'Hendrix': 'Hendrix',
        'Hilbert': 'Hilbert',
        # 'Hinds CC': None,  # Not in standard set
        'Hiram': 'Hiram',
        'Hobart & Smith': 'Hobart',
        'Hood': 'Hood',
        'Hope': 'Hope',
        # 'Hope Intl': None,  # Not in standard set
        'Houghton': 'Houghton',
        # 'Houston-Victoria': None,  # Not in standard set
        'Howard Payne': 'Howard Payne',
        'Huntingdon': 'Huntingdon',
        # 'Huntington': None,  # Not in standard set
        'Husson': 'Husson',
        # 'IL Chicago': None,  # Not in standard set
        'IL Tech': 'Illinois Tech',
        'IL Wesleyan': 'Ill. Wesleyan',
        # 'Illinois': None,  # Not in standard set
        'Illinois Col': 'Illinois Col.',
        'Immaculata': 'Immaculata',
        # 'Iowa': None,  # Not in standard set
        'Ithaca': 'Ithaca',
        'J&W RI': 'JWU (Providence)',
        'John Carroll': 'John Carroll',
        'John Jay': 'John Jay',
        'Johns Hopkins': 'Johns Hopkins',
        'Juniata': 'Juniata',
        'Kalamazoo': 'Kalamazoo',
        'Kean': 'Kean',
        'Keene St': 'Keene St.',
        # 'Kent': None,  # Not in standard set
        # 'Kent-Tusc': None,  # Not in standard set
        'Kenyon': 'Kenyon',
        'Keuka': 'Keuka',
        'Keystone': 'Keystone',
        "King's PA": "King's (PA)",
        'Knox': 'Knox',
        'La Roche': 'La Roche',
        # 'La Sierra': None,  # Not in standard set
        'La Verne': 'La Verne',
        'LaGrange': 'LaGrange',
        'Lakeland': 'Lakeland',
        'Lancaster Bib': 'Lancaster Bible',
        'Lasell': 'Lasell',
        'Lawrence': 'Lawrence',
        'Le Tourneau': 'LeTourneau',
        'Lebanon Val': 'Lebanon Valley',
        # 'Lehigh': None,  # Not in standard set
        'Lehman': 'Lehman',
        'Lesley': 'Lesley',
        'Lewis & Clark': 'Lewis & Clark',
        # 'Lewis-Clark ID': None,  # Not in standard set
        'Linfield': 'Linfield',
        'Loras': 'Loras',
        'Luther IA': 'Luther',
        'Lycoming': 'Lycoming',
        'Lynchburg': 'Lynchburg',
        # 'Lyon': None,  # Not in standard set
        'M Hardin-Baylor': 'Mary Hardin-Baylor',
        'MA Boston': 'UMass Boston',
        'MA Col Lib Arts': 'MCLA',
        'MA Dartmouth': 'UMass Dartmouth',
        'MA Maritime': 'Mass. Maritime',
        # 'ME Augusta': None,  # Not in standard set
        'ME Farmington': 'Me.-Farmington',
        'ME Presque Isle': 'Me.-Presque Isle',
        'MIT': 'MIT',
        'MN Morris': 'Minn.-Morris',
        'MS Women': 'MUW',
        'Macalester': 'Macalester',
        # 'Maine': None,  # Not in standard set
        'Manchester': 'Manchester',
        'Manhattanville': 'Manhattanville',
        'Maranatha Bap': 'Maranatha Baptist',
        'Marian WI': 'Marian (WI)',
        'Marietta': 'Marietta',
        'Martin Luther': 'Martin Luther',
        'Mary Baldwin': 'Mary Baldwin',
        'Mary Washington': 'Mary Washington',
        'Marymount VA': 'Marymount (VA)',
        'Maryville TN': 'Maryville (TN)',
        'Marywood': 'Marywood',
        'McDaniel': 'McDaniel',
        'McMurry': 'McMurry',
        'Merchant Marine': 'Merchant Marine',
        'Messiah': 'Messiah',
        'Methodist': 'Methodist',
        # 'Mid-Atlantic Chr': None,  # Not in standard set
        'Middlebury': 'Middlebury',
        'Millikin': 'Millikin',
        'Millsaps': 'Millsaps',
        'Milwaukee Eng': 'MSOE',
        # 'Minot St': None,  # Not in standard set
        'Misericordia': 'Misericordia',
        'Mitchell': 'Mitchell',
        'Monmouth IL': 'Monmouth (IL)',
        'Montclair St': 'Montclair St.',
        # 'Montreat': None,  # Not in standard set
        'Moravian': 'Moravian',
        'Mt Aloysius': 'Mount Aloysius',
        'Mt St Joseph': 'Mt. St. Joseph',
        'Mt St Mary NY': 'Mt. St. Mary (NY)',
        'Mt St Vincent': 'UMSV',  # Not in standard set (UMSV exists but different)
        'Mt Union': 'Mount Union',
        'Muhlenberg': 'Muhlenberg',
        'Muskingum': 'Muskingum',
        'N Central IL': 'North Central (IL)',
        'N Central MN': 'North Central (MN)',
        'N England Col': 'New England Col.',
        'NC Wesleyan': 'N.C. Wesleyan',
        'NE Wesleyan': 'Neb. Wesleyan',
        'NJ City': 'New Jersey City',
        'NVU-Lyndon': 'VTSU Lyndon',
        'NYU': 'NYU',
        'Neumann': 'Neumann',
        'Nichols': 'Nichols',
        'North Park': 'North Park',
        # 'Northern St SD': None,  # Not in standard set
        'Northland': 'Northland',
        # 'Northwestern LA': None,  # Not in standard set
        'Northwestern MN': 'Northwestern-St. Paul',
        'Norwich': 'Norwich',
        'Notre Dame MD': 'Notre Dame (MD)',
        'Oberlin': 'Oberlin',
        'Occidental': 'Occidental',
        'Oglethorpe': 'Oglethorpe',
        # 'Ohio Chr': None,  # Not in standard set
        'Ohio Northern': 'Ohio Northern',
        'Ohio Wesleyan': 'Ohio Wesleyan',
        'Old Westbury': 'Old Westbury',
        'Olivet MI': 'Olivet',
        'Oswego St': 'Oswego St.',
        'Otterbein': 'Otterbein',
        # 'PSU Wilkes-Barre': None,  # Not in standard set
        # 'PSU York': None,  # Not in standard set
        'PSU-Abington': 'Penn St.-Abington',
        'PSU-Altoona': 'Penn St.-Altoona',
        # 'PSU-Beaver': None,  # Not in standard set
        'PSU-Behrend': 'Penn St.-Behrend',
        'PSU-Berks': 'Penn St.-Berks',
        # 'PSU-Brandywine': None,  # Not in standard set
        # 'PSU-DuBois': None,  # Not in standard set
        'PSU-Harrisburg': 'Penn St. Harrisburg',
        # 'PSU-Hazleton': None,  # Not in standard set
        # 'PSU-New Kens': None,  # Not in standard set
        # 'PSU-Schuylkill': None,  # Not in standard set
        # 'PSU-Scranton': None,  # Not in standard set
        # 'PSU-Shenango': None,  # Not in standard set
        'Pac Lutheran': 'Pacific Lutheran',
        'Pacific OR': 'Pacific (OR)',
        # 'Paine': None,  # Not in standard set
        'Peace': 'William Peace',
        'Penn Col Tech': 'Penn College',
        'Pfeiffer': 'Pfeiffer',
        'Piedmont': 'Piedmont',
        'Pitt-Bradford': 'Pitt.-Bradford',
        'Pitt-Greensburg': 'Pitt.-Greensburg',
        'Plattsburgh St': 'Plattsburgh St.',
        'Plymouth St': 'Plymouth St.',
        # 'Point Park': None,  # Not in standard set
        'Pomona-Pitzer': 'Pomona-Pitzer',
        'Principia': 'Principia',
        # 'Providence Chr': None,  # Not in standard set
        'Puget Sound': 'Puget Sound',
        'Purchase': 'Purchase',
        'R Stockton': 'Stockton',
        'RI College': 'Rhode Island Col.',
        'Ramapo': 'Ramapo',
        'Randolph-Macon': 'Randolph-Macon',
        'Redlands': 'Redlands',
        'Rensselaer': 'Rensselaer',
        'Rhodes': 'Rhodes',
        'Ripon': 'Ripon',
        'Rivier': 'Rivier',
        'Roanoke': 'Roanoke',
        'Rochester NY': 'Rochester (NY)',
        'Rochester Tech': 'RIT',
        'Rockford': 'Rockford',
        'Roger Williams': 'Roger Williams',
        'Rose-Hulman': 'Rose-Hulman',
        'Rosemont': 'Rosemont',
        'Rowan': 'Rowan',
        'Russell Sage': 'Russell Sage',
        'Rutgers-Camden': 'Rutgers-Camden',
        'Rutgers-Newark': 'Rutgers-Newark',
        'S Maine': 'Southern Me.',
        'S Virginia': 'Southern Va.',
        # 'SE Ark': None,  # Not in standard set
        # 'SE Oklahoma': None,  # Not in standard set
        'SUNY Canton': 'SUNY Canton',
        'SUNY Cobleskill': 'SUNY Cobleskill',
        'SUNY Fredonia': 'Fredonia',  # Not in standard set (Fredonia exists but not SUNY Fredonia)
        'SUNY Maritime': 'SUNY Maritime',
        'SUNY New Paltz': 'SUNY New Paltz',
        'SUNY Oneonta': 'SUNY Oneonta',
        'SUNY Poly': 'SUNY Poly',
        # 'SW Minnesota': None,  # Not in standard set
        'SW Univ TX': 'Southwestern (TX)',
        'Salem MA': 'Salem St.',
        'Salisbury': 'Salisbury',
        'Salve Regina': 'Salve Regina',
        'Schreiner': 'Schreiner',
        'Scranton': 'Scranton',
        'Sewanee': 'Sewanee',
        'Shenandoah': 'Shenandoah',
        # 'Siena Hts': None,  # Not in standard set
        'Simpson IA': 'Simpson',
        'Skidmore': 'Skidmore',
        # 'Southeastern Bap': None,  # Not in standard set
        'Spalding': 'Spalding',
        'Springfield': 'Springfield',
        # "St Andrew's": None,  # Not in standard set
        'St Elizabeth': 'Saint Elizabeth',
        'St John Fisher': 'St. John Fisher',
        "St John's MN": "Saint John's (MN)",
        'St Joseph CT': 'Saint Joseph (CT)',
        "St Joseph's LI": "St. Joseph's (L.I.)",
        "St Joseph's ME": "St. Joseph's (ME)",
        "St Joseph's NY": "St. Joseph's (Brkln)",
        'St Lawrence': 'St. Lawrence',
        # "St Martin's": None,  # Not in standard set (exists in D2)
        "St Mary's MD": "St. Mary's (MD)",
        "St Mary's MN": "Saint Mary's (MN)",
        'St Norbert': 'St. Norbert',
        'St Olaf': 'St. Olaf',
        'St Scholastica': 'St. Scholastica',
        'St Thomas TX': 'St. Thomas (TX)',
        'St Vincent': 'Saint Vincent',
        'Stevens': 'Stevens',
        'Stevenson': 'Stevenson',
        'Suffolk': 'Suffolk',
        'Susquehanna': 'Susquehanna',
        'Swarthmore': 'Swarthmore',
        # 'TX A&M Texarkana': None,  # Not in standard set
        'TX Lutheran': 'Texas Lutheran',
        # "The Master's": None,  # Not in standard set
        'Thiel': 'Thiel',
        'Thomas ME': 'Thomas (ME)',
        # 'Tiffin': None,  # Not in standard set
        # 'Toccoa Falls': None,  # Not in standard set
        'Transylvania': 'Transylvania',
        'Trine': 'Trine',
        'Trinity CT': 'Trinity (CT)',
        # 'Trinity Jax': None,  # Not in standard set
        'Trinity TX': 'Trinity (TX)',
        # 'Truett-McConnell': None,  # Not in standard set
        'Tufts': 'Tufts',
        # 'UHSP': None,  # Not in standard set
        # 'UT Dallas': None,  # Not in standard set
        'Union NY': 'Union (NY)',
        'Univ Ozarks': 'Ozarks (AR)',
        'Ursinus': 'Ursinus',
        'Utica': 'Utica',
        'VA Wesleyan': 'Va. Wesleyan',
        'Valley Forge': 'Valley Forge',
        'Vassar': 'Vassar',
        'Viterbo': 'Viterbo',
        'W Connecticut': 'WestConn',
        'W New England': 'Western New Eng.',
        # 'W Oregon': None,  # Not in standard set
        # 'W Woods': None,  # Not in standard set
        'WI Eau Claire': 'Wis.-Eau Claire',
        'WI LaCrosse': 'Wis.-La Crosse',
        'WI Lutheran': 'Wis. Lutheran',
        # 'WI Milwaukee': None,  # Not in standard set
        'WI Oshkosh': 'Wis.-Oshkosh',
        # 'WI Parkside': None,  # Not in standard set
        'WI Platteville': 'Wis.-Platteville',
        'WI River Falls': 'UW-River Falls',
        'WI Stevens Pt': 'Wis.-Stevens Point',
        'WI Stout': 'Wis.-Stout',
        'WI Superior': 'Wis.-Superior',
        'WI Whitewater': 'Wis.-Whitewater',
        'Wabash': 'Wabash',
        # 'Warner Pacific': None,  # Not in standard set
        'Wartburg': 'Wartburg',
        'Wash & Jeff': 'Wash. & Jeff.',
        # 'Washington': None,  # Not in standard set (ambiguous)
        'Washington & Lee': 'Wash. & Lee',
        'Washington MD': 'Washington Col.',
        'Washington StL': 'WashU',
        'Waynesburg': 'Waynesburg',
        'Webster': 'Webster',
        'Wentworth Tech': 'Wentworth',
        'Wesleyan CT': 'Wesleyan (CT)',
        # 'Westcliff': None,  # Not in standard set
        'Westfield St': 'Westfield St.',
        'Westminster MO': 'Westminster (MO)',
        'Westminster PA': 'Westminster (PA)',
        'Wheaton IL': 'Wheaton (IL)',
        'Wheaton MA': 'Wheaton (MA)',
        'Wheeling': 'Wheeling',
        'Whitman': 'Whitman',
        'Whittier': 'Whittier',
        'Whitworth': 'Whitworth',
        'Widener': 'Widener',
        'Wilkes': 'Wilkes',
        'Willamette': 'Willamette',
        'Williams': 'Williams',
        # 'Williams Bap': None,  # Not in standard set
        'Wilmington OH': 'Wilmington (OH)',
        'Wilson': 'Wilson',
        # 'Winona St': None,  # Not in standard set
        'Wittenberg': 'Wittenberg',
        'Wm Paterson': 'Wm. Paterson',
        'Wooster': 'Wooster',
        'Worcester St': 'Worcester St.',
        'Worcester Tech': 'WPI',  # Not in standard set (WPI exists)
        'Yeshiva': 'Yeshiva',
        'York PA': 'York (PA)',
        'Birmingham So':'Birmingham-So.',
        'Summit':'Clarks Summit',
        'E Nazarene':'Eastern Nazarene',
        'Texas-Dallas':'Texas-Dallas',
        'Abe Baldwin': 'Abraham Baldwin (GA)',
        'Aquinas': 'Aquinas (MI)',
        'Arizona Chr': 'Arizona Christian',
        'Ark Baptist': 'Arkansas Baptist',
        'Ave Maria': 'Ave Maria',
        'Avila': 'Avila (MO)',
        'Baker KS': 'Baker',
        'Bellevue': 'Bellevue (NE)',
        'Benedictine KS': 'Benedictine (KS)',
        'Benedictine AZ': 'Benedictine Mesa',
        'Bethany KS': 'Bethany (KS)',
        'Bethel IN': 'Bethel (IN)',
        'Bethel TN': 'Bethel (TN)',
        'Blue Mtn': 'Blue Mountain (MS)',
        'Bluefield Univ': 'Bluefield (VA)',
        'Brescia': 'Brescia (KY)',
        'Brewton-Parker': 'Brewton-Parker (GA)',
        'Briar Cliff': 'Briar Cliff (IA)',
        'Brit Columbia': 'British Columbia',
        'Bryan': 'Bryan (TN)',
        'Bushnell': 'Bushnell (OR)',
        'Columbia Intl': 'CIU (SC)',
        'Calumet-St Jos': 'Calumet (IN)',
        'Campbellsville': 'Campbellsville (KY)',
        'Carolina Univ': 'Carolina (NC)',
        'Central Bap': 'Central Baptist (AR)',
        'Central Chr': 'Central Christian (KS)',
        'Cent Methodist': 'Central Methodist',
        'Clarke': 'Clarke (IA)',
        'Cleary': 'Cleary (MI)',
        'Col of Idaho': 'College of Idaho',
        'Col Ozarks': 'College of the Ozarks',
        'Columbia MO': 'Columbia (MO)',
        'Concordia MI': 'Concordia (MI)',
        'Concordia NE': 'Concordia (NE)',
        'Corban': 'Corban',
        'Cornerstone': 'Cornerstone',
        "Crowley's Ridge": "Crowley's Ridge (AR)",
        'Culver-Stockton': 'Culver-Stockton (MO)',
        'Cumberland TN': 'Cumberland (TN)',
        'Cumberlands KY': 'Cumberlands (KY)',
        'Dakota St': 'Dakota State (SD)',
        'Dakota Wesleyan': 'Dakota Wesleyan (SD)',
        'Defiance': 'Defiance College (OH)',
        'Dickinson St ND': 'Dickinson State (ND)',
        'Dillard': 'Dillard (LA)',
        'Doane': 'Doane (NE)',
        'Dordt': 'Dordt (IA)',
        'Eastern Oregon': 'Eastern Oregon',
        'Embry-Riddle AZ': 'Embry-Riddle (AZ)',
        'Evangel': 'Evangel (MO)',
        'Faulkner': 'Faulkner (AL)',
        'Fisher': 'Fisher (MA)',
        'FL Memorial': 'Florida Memorial',
        'Florida National': 'Florida National',
        'Freed-Hardeman': 'Freed-Hardeman (TN)',
        'Friends': 'Friends (KS)',
        'Georgetown KY': 'Georgetown (KY)',
        'GA Gwinnett': 'Georgia Gwinnett',
        'Goshen': 'Goshen (IN)',
        'Grace IN': 'Grace (IN)',
        'Graceland': 'Graceland',
        'Grand View': 'Grand View',
        'Hannibal-Lagr': 'Hannibal-LaGrange (MO)',
        'Harris-Stowe': 'Harris-Stowe (MO)',
        'Hastings': 'Hastings',
        'Hope Intl': 'Hope International',
        'Huntington': 'Huntington (IN)',
        'Huston-Tillot': 'Huston-Tillotson',
        'IU Columbus': 'IU Columbus (IN)',
        'IU Kokomo': 'IU Kokomo',
        'IN S Bend': 'Indiana South Bend',
        'IN Southeast': 'Indiana Southeast',
        'Indiana Tech': 'Indiana Tech',
        'IN Wesleyan': 'Indiana Wesleyan',
        'Hesston':'Hesston College',
        'Jamestown': 'Jamestown (ND)',
        'Jarvis Chr': 'Jarvis Christian',
        'Johnson TN': 'Johnson (TN)',
        'Judson IL': 'Judson (IL)',
        'KS Wesleyan': 'Kansas Wesleyan',
        'Keiser': 'Keiser (FL)',
        'Kentucky Chr': 'Kentucky Christian',
        'LSU-Alexandria': 'LSU Alexandria (LA)',
        'LSU Shreveport': 'LSU Shreveport (LA)',
        'La Sierra': 'La Sierra',
        'Lawrence Tech': 'Lawrence Tech',
        'Lewis-Clark ID': 'Lewis-Clark (ID)',
        'Lindsey Wilson': 'Lindsey Wilson (KY)',
        'Louisiana Chr': 'Louisiana Christian',
        'Lourdes': 'Lourdes',
        'Loyola NO': 'Loyola (LA)',
        'Madonna': 'Madonna (MI)',
        'Marian IN': 'Marian',
        'Mayville St': 'Mayville State',
        'McPherson': 'McPherson (KS)',
        'Menlo': 'Menlo',
        'MI Dearborn': 'Michigan-Dearborn',
        'Mid-Am Chr': 'Mid-America Christian',
        'Mid Am Nazarene': 'MidAmerica Nazarene',
        'Mid Georgia': 'Middle Georgia State',
        'Midland NE': 'Midland',
        'Midway': 'Midway',
        'Milligan': 'Milligan (TN)',
        'Mission': 'Mission (MO)',
        'MO Baptist': 'Missouri Baptist',
        'Missouri Val': 'Missouri Valley',
        'Mobile': 'Mobile (AL)',
        'Montreat': 'Montreat (NC)',
        'Morningside': 'Morningside (IA)',
        'Morris SC': 'Morris',
        'Mt Marty': 'Mount Marty',
        'Mt Mercy': 'Mount Mercy',
        'Mt Vernon Naz': 'Mount Vernon (OH)',
        'Nelson TX': 'Nelson (TX)',
        'New College FL': 'New College (FL)',
        'Northwestern IA': 'Northwestern (IA)',
        'Northwestern OH': 'Northwestern (OH)',
        'Ottawa AZ': 'OUAZ',
        'Oakland City': 'Oakland City (IN)',
        'Oakwood': 'Oakwood (AL)',
        'Ohio Chr': 'Ohio Christian',
        'OK City': 'Oklahoma City',
        'Panhandle St': 'Panhandle State',
        'OK Wesleyan': 'Oklahoma Wesleyan',
        'Olivet Naz': 'Olivet Nazarene (IL)',
        'Oregon Tech': 'Oregon Tech',
        'Ottawa KS': 'Ottawa (KS)',
        'Our Lady Lake': 'Our Lady Lake',
        'Park': 'Park',
        'Park-Gilbert': 'Park-Gilbert (AZ)',
        'Peru St': 'Peru State',
        'Philander Smith': 'Philander Smith (AR)',
        'Pikeville': 'Pikeville (KY)',
        'Point': 'Point',
        'Providence Chr': 'Providence (CA)',
        'Reinhardt': 'Reinhardt (GA)',
        'Rio Grande': 'Rio Grande',
        'Rochester MI': 'Rochester (MI)',
        'Rust': 'Rust (MS)',
        'St Francis IN': 'Saint Francis (IN)',
        'St Mary KS': 'Saint Mary',
        'St Xavier IL': 'Saint Xavier (IL)',
        'Oklahoma S&A': 'Science and Arts',
        'Shawnee St': 'Shawnee State',
        'Siena Hts': 'Siena Heights',
        'Simpson CA': 'Simpson (CA)',
        'Southeastern FL': 'Southeastern (FL)',
        'Southeastern Bap': 'Southeastern Baptist',
        'Southern N.O.': 'SUNO (LA)',
        'Univ of Southwest': 'Southwest',
        'Southwestern KS': 'Southwestern',
        'SW Christian': 'SW Christian',
        'Spartanburg Meth': 'Spartanburg Methodist',
        'Spring Arbor': 'Spring Arbor (MI)',
        'St Ambrose': 'St. Ambrose (IA)',
        "St Andrew's": 'St. Andrews (NC)',
        'St Francis IL': 'St. Francis (IL)',
        'St Thomas FL': 'St. Thomas (FL)',
        'Sterling': 'Sterling',
        'Stillman': 'Stillman',
        'Tabor': 'Tabor',
        'Talladega': 'Talladega (AL)',
        'Taylor IN': 'Taylor (IN)',
        'TN Southern': 'Tennessee Southern',
        'TN Wesleyan': 'Tennessee Wesleyan',
        'TX A&M Texarkana': 'Texas A&M  Texarkana',
        'Texas Col': 'Texas College',
        'TX Wesleyan': 'Texas Wesleyan',
        "The Master's": "The Master's (CA)",
        'Thomas GA': 'Thomas',
        'Tougaloo': 'Tougaloo (MS)',
        'Trinity Chr': 'Trinity Christian (IL)',
        'Truett-McConnell': 'Truett McConnell',
        'UHSP': 'UHSP',
        # 'Union TN': 'Union Commonwealth',
        'Valley City St': 'Valley City State',
        'Viterbo': 'Viterbo',
        'Voorhees': 'Voorhees University',
        'WV Tech': 'WVU Tech (WV)',
        'Waldorf': 'Waldorf',
        'Warner': 'Warner (FL)',
        'Wayland': 'Wayland Baptist (TX)',
        'Webber': 'Webber (FL)',
        'Westcliff': 'Westcliff (CA)',
        'Wilberforce': 'Wilberforce (OH)',
        'Wiley': 'Wiley (TX)',
        'Wm Carey': 'William Carey (MS)',
        'William Penn': 'William Penn (IA)',
        'W Woods': 'William Woods',
        'Williams Bap': 'Williams Baptist (AR)',
        'Xavier LA': 'Xavier (LA)',
        'York NE': 'York (NE)',
        'Union KY':'Union Commonwealth',
    }


    return mapping

def standardize_team_names(df, division, mapping=None):
    """
    Standardize team names in the DataFrame
    
    Args:
        df: DataFrame with Team, Opponent, home_team, away_team columns
        mapping: Optional custom mapping dictionary. If None, uses default mapping.
    
    Returns:
        DataFrame with standardized team names
    """
    if mapping is None:
        mapping = create_massey_to_standard_mapping(division)
    
    df = df.copy()
    
    # Track teams that couldn't be mapped
    unmapped_teams = set()
    
    # Standardize all team name columns
    for col in ['Team', 'Opponent', 'home_team', 'away_team']:
        if col in df.columns:
            # Apply mapping
            df[col] = df[col].map(lambda x: mapping.get(x, x))
            
            # Track unmapped teams (those that returned None)
            unmapped = df[df[col].isna()][col].unique()
            if len(unmapped) > 0:
                unmapped_teams.update(unmapped)
    
    # Remove rows where any team name is None (couldn't be mapped)
    original_len = len(df)
    df = df.dropna(subset=['Team', 'Opponent', 'home_team', 'away_team'])
    removed_len = original_len - len(df)
    
    if removed_len > 0:
        print(f"\nRemoved {removed_len} rows with unmapped teams")
        if unmapped_teams:
            print(f"Unmapped teams: {unmapped_teams}")
    
    return df

# --- Helper Functions ---
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
_session = None

def get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(
            max_retries=retry, 
            pool_connections=50, 
            pool_maxsize=50,
            pool_block=False
        )
        _session.mount('http://', adapter)
        _session.mount('https://', adapter)
        _session.headers.update({"User-Agent": "Mozilla/5.0"})
    return _session

def get_soup(url):
    session = get_session()
    response = session.get(url, timeout=10)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")

def scrape_massey_ratings(url, headless: bool = True) -> pd.DataFrame:
    """
    Scrape NCAA D3 baseball ratings from Massey Ratings using Selenium.
    
    Args:
        url: The URL to scrape (default: Massey D3 ratings page)
        headless: Run browser in headless mode (default: True)
    
    Returns:
        DataFrame with team ratings data (Massey team names, not yet mapped)
    """
    # Set up Chrome options
    chrome_options = Options()
    if headless:
        chrome_options.add_argument('--headless=new')  # Use new headless mode
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')  # Disable GPU for faster headless
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-images')  # Don't load images
    chrome_options.add_argument('--blink-settings=imagesEnabled=false')
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    chrome_options.page_load_strategy = 'eager'  # Don't wait for all resources
    
    # Initialize driver
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_page_load_timeout(15)  # Timeout if page takes too long
    
    try:
        # Load page
        print(f"Loading {url}...")
        driver.get(url)
        
        # Wait for table to load - reduced from 20 to 10 seconds
        wait = WebDriverWait(driver, 5)
        table = wait.until(EC.presence_of_element_located((By.ID, "SHCtable")))
        
        # Remove the sleep - unnecessary with WebDriverWait
        # time.sleep(2)
        
        # Get page source and parse with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Close driver immediately after getting page source
        driver.quit()
        
        # Find the table
        table = soup.find('table', {'id': 'SHCtable'})

        if not table:
            raise ValueError("Could not find ratings table on page")

        # Check if Delta column exists by looking at headers
        thead = table.find('thead')
        headers = [th.text.strip() for th in thead.find_all('th', class_='frank')] if thead else []
        has_delta = 'Δ' in headers
        offset = 1 if has_delta else 0  # Offset indices if Delta column exists

        # Extract data with list comprehension for better performance
        data = []
        tbody = table.find('tbody')

        for row in tbody.find_all('tr', class_='bodyrow'):
            row_data = {}

            # Team name and conference
            team_cell = row.find('td', class_='fteam')
            team_link = team_cell.find('a')
            row_data['Team'] = team_link.text.strip()
            row_data['Team_ID'] = team_link.get('href', '').split('/')[-1]

            conf_div = team_cell.find('div', class_='detail')
            if conf_div:
                conf_link = conf_div.find('a')
                row_data['Conference'] = conf_link.text.strip() if conf_link else ''
                row_data['Conference_ID'] = conf_link.get('href', '').split('/')[-1] if conf_link else ''
            else:
                row_data['Conference'] = ''
                row_data['Conference_ID'] = ''

            # Record
            record_cell = row.find('td', class_='fwlt')
            record_text = record_cell.contents[0].strip()
            row_data['Record'] = record_text

            win_pct_div = record_cell.find('div', class_='detail')
            row_data['Win_Pct'] = float(win_pct_div.text.strip()) if win_pct_div else None

            # Rating columns - may or may not have Delta (Δ) column first
            # With Delta: Delta, Rat, Pwr, Off, Def, HFA, SoS, ...
            # Without Delta: Rat, Pwr, Off, Def, HFA, SoS, ...
            rating_cells = row.find_all('td', class_='frank')

            # Use helper function to reduce repetition
            def extract_rating(cell, index):
                if len(rating_cells) > index:
                    try:
                        rank_text = cell.contents[0].strip()
                        detail = cell.find('div', class_='detail')
                        return int(rank_text), float(detail.text.strip()) if detail else None
                    except (ValueError, IndexError, AttributeError):
                        return None, None
                return None, None

            # Rat (Rating)
            rat_idx = 0 + offset
            if len(rating_cells) > rat_idx:
                row_data['Rat_Rank'], row_data['Rat'] = extract_rating(rating_cells[rat_idx], rat_idx)

            # Pwr (Power)
            pwr_idx = 1 + offset
            if len(rating_cells) > pwr_idx:
                row_data['Pwr_Rank'], row_data['Pwr'] = extract_rating(rating_cells[pwr_idx], pwr_idx)

            # Off (Offense)
            off_idx = 2 + offset
            if len(rating_cells) > off_idx:
                row_data['Off_Rank'], row_data['Off'] = extract_rating(rating_cells[off_idx], off_idx)

            # Def (Defense)
            def_idx = 3 + offset
            if len(rating_cells) > def_idx:
                row_data['Def_Rank'], row_data['Def'] = extract_rating(rating_cells[def_idx], def_idx)

            # HFA (Home Field Advantage)
            hfa_idx = 4 + offset
            if len(rating_cells) > hfa_idx:
                try:
                    row_data['HFA'] = float(rating_cells[hfa_idx].text.strip())
                except (ValueError, AttributeError):
                    row_data['HFA'] = None

            # SoS (Strength of Schedule) - may not be valid for teams with no games
            sos_idx = 5 + offset
            if len(rating_cells) > sos_idx:
                try:
                    sos_text = rating_cells[sos_idx].contents[0].strip()
                    row_data['SoS_Rank'] = int(sos_text)
                    sos_detail = rating_cells[sos_idx].find('div', class_='detail')
                    row_data['SoS'] = float(sos_detail.text.strip()) if sos_detail else None
                except (ValueError, IndexError):
                    row_data['SoS_Rank'] = None
                    row_data['SoS'] = None

            data.append(row_data)
        
        print(f"Successfully scraped {len(data)} teams")
        
    except Exception as e:
        driver.quit()
        raise e
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add overall rank based on Rat_Rank
    df.insert(0, 'Rank', df['Rat_Rank'])
    
    return df

def get_conference_mapping(division) -> dict:
    """
    Get the mapping dictionary from Massey conference names to standardized names.
    
    Returns:
        Dictionary mapping Massey conference names to standard names
    """
    if division == "D1":
        mapping = {
            'Southwestern AC':'SWAC',
            'Southeastern':'SEC',
            'Atlantic Coast':'ACC',
            'Coastal':'Coastal Athletic',
            'Atlantic Sun':'ASUN',
            'OH Valley':'Ohio Valley',
            'Big 10':'Big Ten',
            'Pac 10':'Pac-12',
            'Mid-Eastern AC':'MEAC',
            'Missouri Val':'Missouri Valley',
            'Horizon':'Horizon League',
            'D1 Independent':'Independent',
            'Metro Atlantic':'MAAC',
            'Mid-Continent':'The Summit League',
        }
    elif division == "D2":
        mapping = {
            'California CAA': 'CCAA',
            'Central Atlantic': 'CACC',
            'Chicagoland': 'CCAC',
            'Conf Carolinas': 'Conference Carolinas',
            'East Coast': 'ECC',
            'Great American': 'GAC',
            'Great Lakes IAC': 'GLIAC',
            'Great Lakes Val': 'GLVC',
            'Great Midwest': 'G-MAC',
            'Great Northwest': 'Great Northwest',
            'Gulf South': 'Gulf South',
            'Lone Star': 'Lone Star',
            'Mid America IAA': 'Mid-America Intercollegiate',
            'Mountain East': 'MEC',
            'NAIA Independent': 'NAIA Ind.',
            'NCAA II Ind': 'DII Independent',
            'New South AC': 'NSAC',
            'Northeast-10': 'NE10',
            'Northern Sun': 'NSIC',
            'Pacific West': 'PacWest',
            'Peach Belt': 'Peach Belt',
            'Penn St AC': 'PSAC',
            'Rocky Mtn AC': 'RMAC',
            'South Atlantic': 'SAC',
            'Southern IAC': 'SIAC',
            'Sunshine State': 'Sunshine State',
        }
    elif division == "D3":
        mapping = {
            # Division III conferences
            'Allegheny Mtn': 'AMCC',
            'American Rivers': 'American Rivers',
            'American SW': 'ASC',
            'Atlantic East': 'Atlantic East',
            'C of New England': 'CNE',
            'CC South': 'CCS',
            'Centennial': 'Centennial',
            'City Univ NY': 'CUNYAC',
            'Coast-to-Coast': 'C2C',
            'Empire 8': 'Empire 8',
            'Great Northeast': 'Great Northeast',
            'Heartland CAC': 'HCAC',
            'Ill & Wisc': 'CCIW',
            'Landmark': 'Landmark',
            'Liberty League': 'Liberty League',
            'Little East': 'Little East',
            'MASCAC': 'MASCAC',
            'Michigan IAA': 'Michigan Intercol. Ath. Assn.',
            'Mid Atlantic': 'MAC',  # Note: This splits into MAC Commonwealth and MAC Freedom
            'Midwest': 'MWC',
            'Minnesota IAC': 'MIAC',
            'NCAA III Ind': 'DIII Independent',
            'NE W&M': 'NEWMAC',
            'NESCAC': 'NESCAC',
            'New Jersey A.C.': 'NJAC',
            'North Atlantic': 'North Atlantic',
            'North Coast AC': 'NCAC',
            'Northern Athletic': 'NACC',
            'Northwest': 'NWC',
            'Ohio AC': 'OAC',
            'Old Dominion AC': 'ODAC',
            "Presidents' AC": 'PAC',
            'SAA': 'SAA',
            'SUNY AC': 'SUNYAC',
            'Skyline': 'Skyline',
            'Southern CAC': 'SCAC',
            'Southern Cal IAC': 'SCIAC',
            'St Louis IAC': 'SLIAC',
            'USA South': 'USA South',
            'United East': 'United East',
            'University AA': 'UAA',
            'Upper Midwest': 'UMAC',
            'Wisconsin IAC': 'WIAC',
            
            # Division II conferences
            'California CAA': 'CCAA',
            'Central Atlantic': 'CACC',
            'Chicagoland': 'CCAC',
            'Conf Carolinas': 'Conference Carolinas',
            'East Coast': 'ECC',
            'Great American': 'GAC',
            'Great Lakes IAC': 'GLIAC',
            'Great Lakes Val': 'GLVC',
            'Great Midwest': 'G-MAC',
            'Great Northwest': 'Great Northwest',
            'Gulf South': 'Gulf South',
            'Lone Star': 'Lone Star',
            'Mid America IAA': 'Mid-America Intercollegiate',
            'Mountain East': 'MEC',
            'NAIA Independent': 'NAIA Ind.',
            'NCAA II Ind': 'DII Independent',
            'New South AC': 'NSAC',
            'Northeast-10': 'NE10',
            'Northern Sun': 'NSIC',
            'Pacific West': 'PacWest',
            'Peach Belt': 'Peach Belt',
            'Penn St AC': 'PSAC',
            'Rocky Mtn AC': 'RMAC',
            'South Atlantic': 'SAC',
            'Southern IAC': 'SIAC',
            'Sunshine State': 'Sunshine State',
        }
    elif division == 'NAIA':
        mapping = {}
    return mapping

def scrape_and_map_massey_ratings(url, division,
                                   headless: bool = True,
                                   warn_unmapped: bool = True) -> pd.DataFrame:
    """
    Scrape Massey ratings and map team names and conferences to standardized format.
    
    Args:
        url: The URL to scrape
        headless: Run browser in headless mode
        warn_unmapped: Print warning for unmapped teams or conferences
    
    Returns:
        DataFrame with standardized team and conference names
    """
    # Scrape the data
    df = scrape_massey_ratings(url, headless=headless)
    
    # Get mappings
    team_mapping = create_massey_to_standard_mapping(division)
    conference_mapping = get_conference_mapping(division)
    
    # Store original Massey names
    df['Massey_Team_Name'] = df['Team']
    df['Massey_Conference'] = df['Conference']
    
    # Map team names, keeping original if no mapping exists
    df['Team'] = df['Massey_Team_Name'].map(team_mapping).fillna(df['Massey_Team_Name'])
    
    # Map conference names, keeping original if no mapping exists
    df['Conference'] = df['Massey_Conference'].map(conference_mapping).fillna(df['Massey_Conference'])
    
    # Check for unmapped teams (those not in the mapping dictionary)
    unmapped_teams = df[~df['Massey_Team_Name'].isin(team_mapping.keys())]
    
    if len(unmapped_teams) > 0 and warn_unmapped:
        print(f"\nWarning: {len(unmapped_teams)} teams could not be mapped:")
        for team in unmapped_teams['Massey_Team_Name'].values:
            print(f"  - {team}")
        print()
    
    # Check for unmapped conferences
    unmapped_confs = df[df['Conference'].isna() & df['Massey_Conference'].notna()]
    if len(unmapped_confs) > 0 and warn_unmapped:
        unique_confs = unmapped_confs['Massey_Conference'].unique()
        print(f"\nWarning: {len(unique_confs)} conferences could not be mapped:")
        for conf in unique_confs:
            print(f"  - {conf}")
        print()
    
    # Reorder columns to put mapped names first, then originals
    cols = ['Rank', 'Team', 'Conference', 'Massey_Team_Name', 'Massey_Conference'] + \
           [col for col in df.columns if col not in ['Rank', 'Team', 'Conference', 
                                                       'Massey_Team_Name', 'Massey_Conference']]
    df = df[cols]
    
    mapped_teams = len(df) - len(unmapped_teams)
    mapped_confs = len(df[df['Conference'].notna()])
    
    print(f"Mapped {mapped_teams} / {len(df)} teams successfully")
    print(f"Mapped {mapped_confs} / {len(df[df['Massey_Conference'].notna()])} conferences successfully")
    
    return df

def get_stat_page(stat_name, url, page_num):
    """Fetch a single page of stats."""
    if page_num > 1:
        url = f"{url}/p{page_num}"
    
    try:
        soup = get_soup(url)
        table = soup.find("table")
        if not table:
            return None
        
        headers = [th.text.strip() for th in table.find_all("th")]
        data = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            data.append([col.text.strip() for col in cols])
        
        return {'headers': headers, 'data': data, 'page': page_num}
    
    except requests.exceptions.HTTPError:
        return None
    except Exception as e:
        print(f"Error for {stat_name}, page {page_num}: {e}")
        return None


def get_stat_dataframe(stat_name, stat_links, max_workers=10):
    """Fetch all pages for a stat in parallel."""
    if stat_name not in stat_links:
        print(f"Stat '{stat_name}' not found. Available stats: {list(stat_links.keys())}")
        return None
    
    url = stat_links[stat_name]
    
    # Fetch pages in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_stat_page, stat_name, url, page_num): page_num
            for page_num in range(1, 10)
        }
        
        page_results = []
        for future in as_completed(futures):
            result = future.result()
            if result:
                page_results.append(result)
        
        # If no results, return None
        if not page_results:
            return None
        
        # Sort by page number to maintain order
        page_results.sort(key=lambda x: x['page'])
        
        # Combine all data
        headers = page_results[0]['headers']
        all_data = []
        for result in page_results:
            all_data.extend(result['data'])
    
    if all_data:
        df = pd.DataFrame(all_data, columns=headers)
        for col in df.columns:
            if col != "Team":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    else:
        return None


####################### Threading #######################

def threaded_stat_fetch(stat_names, stat_links, max_workers=10):
    """Fetch multiple stats in parallel, with each stat's pages also parallelized."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_stat = {
            executor.submit(get_stat_dataframe, stat, stat_links, max_workers=5): stat
            for stat in stat_names
        }
        results = {}
        for future in as_completed(future_to_stat):
            stat = future_to_stat[future]
            try:
                results[stat] = future.result()
            except Exception as e:
                print(f"Failed to fetch {stat}: {e}")
                results[stat] = None
    return results

def clean_duplicates(df, group_col, min_col):
    duplicates = df[df.duplicated(group_col, keep=False)]
    filtered = duplicates.loc[duplicates.groupby(group_col)[min_col].idxmin()]
    cleaned = df[~df[group_col].isin(duplicates[group_col])]
    return pd.concat([cleaned, filtered], ignore_index=True)

####################### Merging + Final Stats #######################

def clean_and_merge(stats_raw, transforms_dict):
    dfs = []
    for stat, df in stats_raw.items():
        if df is not None and stat in transforms_dict:
            df["Team"] = df["Team"].str.strip()
            df = df.dropna(subset=["Team"])
            df_clean = transforms_dict[stat](df)
            dfs.append(df_clean)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="Team", how="outer")  # outer to catch all teams

    merged = merged.loc[:, ~merged.columns.duplicated()].sort_values('Team').reset_index(drop=True)
    
    # Only compute derived stats for rows that have the required columns populated
    merged["OPS"] = merged["SLG"] + merged["OBP"]
    merged["PYTHAG"] = round(
        (merged["RS"] ** 1.83) / ((merged["RS"] ** 1.83) + (merged["RA"] ** 1.83)), 3
    )
    return merged




















###############################################
###############################################
###############################################
###############################################
### ACTUAL RUNNING OF THE CODE IS DOWN HERE ###
###############################################
###############################################
###############################################
###############################################

import os
cst = pytz.timezone('America/Chicago')
formatted_date = datetime.now(cst).strftime('%m_%d_%Y')
current_season = datetime.today().year
comparison_date = pd.to_datetime(formatted_date, format="%m_%d_%Y")

df = scrape_massey_scores("https://masseyratings.com/scores.php?s=658933&sub=11620&all=1&mode=3&sch=on&format=0")
schedule_df = expand_games_to_team_rows(df)
schedule_df = standardize_team_names(schedule_df, division="D3")
schedule_df['Date'] = pd.to_datetime(schedule_df['Date'])
offensive_whip = schedule_df[
    (schedule_df["Date"] <= comparison_date) & (schedule_df["home_score"] != schedule_df["away_score"])
].reset_index(drop=True)

base_url = "https://www.ncaa.com"
soup = get_soup(f"{base_url}/stats/baseball/d3")
dropdown = soup.find("select", {"id": "select-container-team"})
stat_links = {
    option.text.strip(): base_url + option["value"]
    for option in dropdown.find_all("option") if option.get("value")
}

massey = scrape_and_map_massey_ratings(url = "https://masseyratings.com/cbase/ncaa-d3/ratings", division="D3")[['Team', 'Conference', 'Win_Pct', 'Record', 'Rat_Rank', 'Rat', 'Pwr_Rank', 'Pwr']]

####################### Transform Config #######################

STAT_TRANSFORMS = {
    "Batting Average": lambda df: df.assign(
        HPG=df["H"] / df["G"],
        ABPG=df["AB"] / df["G"],
        HPAB=df["H"] / df["AB"]
    ).drop(columns=['Rank']),

    "Base on Balls": lambda df: df.assign(
        BBPG=df["BB"] / df["G"]
    ).drop(columns=['Rank', 'G']),

    "Earned Run Average": lambda df: df.rename(columns={"R": "RA"}).drop(columns=['Rank', 'G']),

    "Fielding Percentage": lambda df: df.assign(
        APG=df["A"] / df["G"],
        EPG=df["E"] / df["G"]
    ).drop(columns=['Rank', 'G']),

    "On Base Percentage": lambda df: df.rename(columns={"PCT": "OBP"}).assign(
        HBPPG=df["HBP"] / df["G"]
    ).drop(columns=['Rank', 'G', 'AB', 'H', 'BB', 'SF', 'SH']),

    "Runs": lambda df: df.assign(
        RPG=df["R"] / df["G"]
    ).rename(columns={"R": "RS"}).drop(columns=['Rank', 'G']),

    "Slugging Percentage": lambda df: df.rename(columns={"SLG PCT": "SLG"}).drop(columns=['Rank', 'G', 'AB']),

    "Strikeouts Per Nine Innings": lambda df: df.rename(columns={"K/9": "KP9"}).drop(columns=['Rank', 'G', 'IP', 'SO']),

    "Walks Allowed Per Nine Innings": lambda df: df.rename(columns={"PG": "WP9"}).drop(columns=['Rank', 'G', 'IP', 'BB']),

    "WHIP": lambda df: df.drop(columns=['Rank', 'HA', 'IP', 'BB']),
}

####################### Merging + Final Stats #######################

####################### Run It #######################

stat_list = list(STAT_TRANSFORMS.keys())
raw_stats = threaded_stat_fetch(stat_list, stat_links, max_workers=20)
baseball_stats = clean_and_merge(raw_stats, STAT_TRANSFORMS)
baseball_stats = pd.merge(baseball_stats, massey, on='Team', how='left')

####################### wOBA Stat Transforms #######################

STAT_TRANSFORMS_WOBA = {
    "Base on Balls": lambda df: df.assign(
        BBPG=df["BB"] / df["G"]
    )[["Team", "BB", "G", "BBPG"]],

    "Hit by Pitch": lambda df: df[["Team", "HBP"]],

    "Hits": lambda df: df[["Team", "AB", "H"]],

    "Doubles": lambda df: df[["Team", "2B"]],

    "Triples": lambda df: df[["Team", "3B"]],

    "Home Runs Per Game": lambda df: (
        lambda _df: pd.concat([
            _df[~_df["Team"].isin(
                _df[_df.duplicated("Team", keep=False)]["Team"]
            )],
            _df[_df.duplicated("Team", keep=False)].groupby("Team", as_index=False).apply(
                lambda g: g.loc[g["HR"].idxmin()]
            )
        ], ignore_index=True)
    )(df.rename(columns={"PG": "HRPG"}).drop(columns=["Rank", "G"])),

    "Sacrifice Flies": lambda df: df[["Team", "SF"]],

    "Runs": lambda df: df.assign(
        RPG=df["R"] / df["G"]
    ).rename(columns={"R": "RS"}).drop(columns=["Rank", "G"]),

    "Sacrifice Bunts": lambda df: df.rename(columns={"SH": "SB"}).assign(
        SBPG=lambda x: x["SB"] / x["G"]
    ).drop(columns=["Rank", "G"]),

    "Earned Run Average": lambda df: df.rename(columns={"R": "RA"}).drop(columns=["Rank", "G"]),

    "Strikeout-to-Walk Ratio": lambda df: df.rename(columns={"BB": "PBB"})[["Team", "K/BB", "PBB", "SO"]],

    "Hits Allowed Per Nine Innings": lambda df: df.rename(columns={"PG": "HAPG"})[["Team", "HA", "HAPG"]],

    "Hit Batters": lambda df: df[["Team", "HB"]],
}

####################### Fetch + Transform + Merge #######################

# Only pull stats we need
woba_stats = list(STAT_TRANSFORMS_WOBA.keys())

# Threaded fetch
raw_woba_stats = threaded_stat_fetch(woba_stats, stat_links)

# Apply transforms + merge
dfs = []
for stat, df in raw_woba_stats.items():
    if df is not None and stat in STAT_TRANSFORMS_WOBA:
        df["Team"] = df["Team"].str.strip()
        df = df.dropna(subset=["Team"])
        dfs.append(STAT_TRANSFORMS_WOBA[stat](df))

# Merge all together
wOBA = dfs[0]
for df in dfs[1:]:
    wOBA = pd.merge(wOBA, df, on="Team", how="left")

# Fill and compute final metrics
wOBA = wOBA.fillna(0)
wOBA["PA"] = wOBA["AB"] + wOBA["BB"] + wOBA["HBP"] + wOBA["SF"] + wOBA["SB"]
league_HR_per_game = wOBA["HR"].sum() / wOBA["G"].sum()
wOBA["HR_A"] = wOBA["G"] * league_HR_per_game

wOBA['1B'] = wOBA['H'] - wOBA['2B'] - wOBA['3B'] - wOBA['HR']
wOBA['wOBA'] = ((0.69 * wOBA['BB']) + (0.72 * wOBA['HBP']) + (0.88 * wOBA['1B']) + (1.24 * wOBA['2B']) + (1.56 * wOBA['3B']) + (1.95 * wOBA['HR'])) / (wOBA['PA'])
league_wOBA = (wOBA['wOBA'] * wOBA['PA']).sum() / wOBA['PA'].sum()
league_R_PA = wOBA['RS'].sum() / wOBA['PA'].sum()
wOBA_scale = league_R_PA / league_wOBA
wOBA['wRAA'] = ((wOBA['wOBA'] - league_wOBA) / wOBA_scale) * wOBA['PA']
league_RS = wOBA['RS'].sum()
league_G = wOBA['G'].sum()
RPW = 2 * (league_RS / league_G)
wOBA['oWAR'] = wOBA['wRAA'] / RPW
wOBA['ISO'] = (wOBA['2B'] + (2 * wOBA['3B']) + (3 * wOBA['HR'])) / wOBA['AB']
wOBA['wRC'] = (((wOBA['wOBA'] - league_wOBA) / wOBA_scale) + league_R_PA) * wOBA['PA']
wOBA['wRC+'] = (wOBA['wRC'] / wOBA['PA']) / league_R_PA * 100
wOBA['BB%'] = wOBA['BB'] / wOBA['PA']
wOBA['BABIP'] = (wOBA['H'] - wOBA['HR']) / (wOBA['AB'] + wOBA['SF'])

wOBA['RA9'] = (wOBA['RA'] / wOBA['IP']) * 9
wOBA['LOB%'] = (wOBA['HA'] + wOBA['PBB'] + wOBA['HB'] - wOBA['RA']) / (wOBA['HA'] + wOBA['PBB'] + wOBA['HB'] - (1.4*wOBA['HR_A']))
wOBA['FIP'] = ((13 * wOBA['HR_A'] + 3 * (wOBA['PBB'] + wOBA['HB']) - 2 * wOBA['SO']) / wOBA['IP'])

league_RA9 = wOBA['RA'].sum() / wOBA['G'].sum()
league_ERA = (wOBA['ER'].sum() * 9) / wOBA['IP'].sum()
replacement_level_ERA = wOBA['ERA'].quantile(0.80)
multiplier = replacement_level_ERA / league_ERA
replacement_RA9 = league_RA9 * multiplier
league_FIP = (wOBA['FIP'] * wOBA['IP']).sum() / wOBA['IP'].sum()
replacement_level_FIP = wOBA['FIP'].quantile(0.80)
multiplier = replacement_level_FIP / league_FIP
replacement_RA9 = league_RA9 * multiplier  # Adjust RA9 to match replacement level
wOBA['pWAR'] = ((replacement_RA9 - wOBA['FIP']) / RPW) * (wOBA['IP'] / 9)

mean_oWAR = wOBA['oWAR'].mean()
std_oWAR = wOBA['oWAR'].std()
mean_pWAR = wOBA['pWAR'].mean()
std_pWAR = wOBA['pWAR'].std()
wOBA['oWAR_z'] = (wOBA['oWAR'] - mean_oWAR) / std_oWAR
wOBA['pWAR_z'] = (wOBA['pWAR'] - mean_pWAR) / std_pWAR
wOBA['fWAR'] = wOBA['oWAR_z'] + wOBA['pWAR_z']
wOBA['Offensive_WHIP'] = (wOBA['H'] + wOBA['BB']) / ((wOBA['AB'] - wOBA['H']) / 3)
offensive_whip = pd.merge(offensive_whip,
    wOBA[['Team', 'Offensive_WHIP']],
    how='left',
    left_on='Opponent',
    right_on='Team'
).drop(columns=['Team_y']).rename(columns={'Team_x':'Team'})
avg_off_whip = offensive_whip.groupby('Team')['Offensive_WHIP'].mean().reset_index()
avg_off_whip.rename(columns={'Offensive_WHIP': 'Avg_Opp_Offensive_WHIP'}, inplace=True)
wOBA = wOBA.merge(avg_off_whip, how='left', on='Team')
baseball_stats = pd.merge(baseball_stats, wOBA[['Team', 'wOBA', 'wRAA', 'oWAR_z', 'pWAR_z', 'fWAR', 'ISO', 'wRC+', 'BB%', 'BABIP', 'RA9', 'FIP', 'LOB%', 'K/BB', 'Avg_Opp_Offensive_WHIP']], how='left', on='Team')
baseball_stats['WHIP+'] = 100 * (baseball_stats['WHIP'] / baseball_stats['Avg_Opp_Offensive_WHIP'])
baseball_stats['WHIP+'] = baseball_stats['WHIP+'].fillna(95)
baseball_stats = baseball_stats.drop_duplicates(subset='Team', keep='first')
baseball_stats = baseball_stats.dropna(subset=['G'])  # Drop teams with no games played
