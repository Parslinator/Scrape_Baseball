from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
from io import StringIO
import re

def setup_driver():
    """Setup Chrome driver with anti-detection options"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver


def scrape_ncaa_game_by_game(df, team_name, season, driver=None):
    """
    Scrape game-by-game stats for a team from NCAA.org
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'team_id', 'team_name', and 'season' columns
    team_name : str
        Name of the team to scrape
    season : int or str
        Season year
    driver : selenium.webdriver.Chrome, optional
        Selenium webdriver instance. If None, a new driver will be created and closed after use.
    
    Returns:
    --------
    pandas.DataFrame
        Game-by-game statistics table
    """
    
    # Filter dataframe for the specified team and season
    team_data = df[(df['team_name'] == team_name) & (df['season'] == season)]
    
    if team_data.empty:
        raise ValueError(f"No data found for team '{team_name}' in season {season}")
    
    team_id = team_data['year_id'].iloc[0]
    
    # Determine if we need to create and manage the driver
    driver_provided = driver is not None
    if not driver_provided:
        driver = setup_driver()
    
    try:
        # Step 1: Navigate to the team page
        team_url = f"https://stats.ncaa.org/teams/{team_id}"
        driver.get(team_url)
        
        # Wait for the navigation tabs to load
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "nav-tabs")))
        
        time.sleep(2)  # Additional wait for page to fully load
        
        # Step 2: Find and click the Game By Game link
        game_by_game_link = None
        nav_links = driver.find_elements(By.CSS_SELECTOR, "ul.nav.nav-tabs a.nav-link")
        
        for link in nav_links:
            if 'Game By Game' in link.text:
                game_by_game_url = link.get_attribute('href')
                player_id = game_by_game_url.split('/')[-1]
                break
        else:
            raise ValueError(f"Could not find 'Game By Game' link for team {team_name}")
        
        # Step 3: Navigate to the game-by-game page
        driver.get(game_by_game_url)
        
        # Wait for the table to load
        table_id = f"game_log_{player_id}_player"
        wait.until(EC.presence_of_element_located((By.ID, table_id)))
        
        time.sleep(2)  # Additional wait for table to fully render
        
        # Step 4: Get the table HTML and parse it
        table = driver.find_element(By.ID, table_id)
        table_html = table.get_attribute('outerHTML')
        
        # Parse the table into a DataFrame using StringIO to avoid FutureWarning
        game_log_df = pd.read_html(StringIO(table_html))[0]
        
        # Step 5: Clean up the data
        # Remove doubleheader indicators from Date column (e.g., "(1)" or "(2)")
        if 'Date' in game_log_df.columns:
            game_log_df['Date'] = game_log_df['Date'].astype(str).str.replace(r'\(\d+\)$', '', regex=True).str.strip()
        
        # Clean up Opponent column
        if 'Opponent' in game_log_df.columns:
            # Remove ranking numbers (e.g., "#15 " or "#6 ")
            game_log_df['Opponent'] = game_log_df['Opponent'].astype(str).str.replace(r'^#\d+\s+', '', regex=True)
            # Remove the @ symbol at the start (for away games)
            game_log_df['Opponent'] = game_log_df['Opponent'].str.replace(r'^@', '', regex=True)
            # Remove neutral site location (space followed by @ and location)
            game_log_df['Opponent'] = game_log_df['Opponent'].str.replace(r'\s+@.*$', '', regex=True)
            # Strip any leading or trailing whitespace
            game_log_df['Opponent'] = game_log_df['Opponent'].str.strip()
        
        # Convert numeric columns to numeric types
        numeric_columns = ['AB', 'BB', 'HBP', 'SF', 'SH', 'H', '2B', '3B', 'HR', 'R']
        for col in numeric_columns:
            if col in game_log_df.columns:
                # Remove trailing "/" markers (e.g., "3/" becomes "3")
                game_log_df[col] = game_log_df[col].astype(str).str.replace(r'/$', '', regex=True)
                # Convert to numeric
                game_log_df[col] = pd.to_numeric(game_log_df[col], errors='coerce')
        
        # Fill NaN values with 0 for numeric columns
        game_log_df[numeric_columns] = game_log_df[numeric_columns].fillna(0)
        
        # Calculate PA and 1B
        game_log_df['PA'] = game_log_df['AB'] + game_log_df['BB'] + game_log_df['HBP'] + game_log_df['SF'] + game_log_df['SH']
        game_log_df['1B'] = game_log_df['H'] - game_log_df['2B'] - game_log_df['3B'] - game_log_df['HR']
        
        return game_log_df
        
    except Exception as e:
        raise Exception(f"Error scraping game-by-game data: {str(e)}")
    
    finally:
        # Only quit the driver if we created it
        if not driver_provided:
            driver.quit()

# must have this years_df dataframe saved and loaded
years_df = pd.read_csv("./ncaa_team_ids.csv")
driver = setup_driver()
# check years and teams available in the years_df dataframe
arkansas_game_by_game = scrape_ncaa_game_by_game(years_df, "Arkansas", 2025, driver)


# IF YOU EVER PLAN ON SCRAPING MORE THAN ONE TEAM AT A TIME
# USE A RANDOM SLEEP DELAY BETWEEN EACH SCRAPE (0.1 - 5 SECONDS)
# ONLY DO 50 OR SO TEAMS AT A TIME, THEN WAIT FOR A COUPLE MINUTES
# YOU HAVE TO BE VERY ROBUST / CAUTIOUS