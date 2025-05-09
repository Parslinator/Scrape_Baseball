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

# --- Warren Nolan Helper Functions ---
def get_soup(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")

def scrape_warrennolan_table(url, expected_columns):
    soup = get_soup(url)
    table = soup.find('table', class_='normal-grid alternating-rows stats-table')
    data = []
    if table:
        for row in table.find('tbody').find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 2:
                name_div = cells[1].find('div', class_='name-subcontainer')
                full_text = name_div.text.strip() if name_div else cells[1].text.strip()
                parts = full_text.split("\n")
                team_name = parts[0].strip()
                conference = parts[1].split("(")[0].strip() if len(parts) > 1 else ""
                data.append([cells[0].text.strip(), team_name, conference])
    return pd.DataFrame(data, columns=expected_columns)

def clean_team_names(df, column='Team'):
    df[column] = df[column].str.replace('State', 'St.', regex=False)
    df[column] = df[column].replace(team_replacements)
    return df

# --- Projected RPI ---
projected_rpi = scrape_warrennolan_table(
    'https://www.warrennolan.com/baseball/2025/rpi-predict',
    expected_columns=["RPI", "Team", "Conference"]
)

# --- Live RPI ---
live_rpi = scrape_warrennolan_table(
    'https://www.warrennolan.com/baseball/2025/rpi-live',
    expected_columns=["Live_RPI", "Team", "Conference"]
)

# --- ELO Ratings ---
url = 'https://www.warrennolan.com/baseball/2025/elo'

# Fetch the webpage content
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table with the specified class
table = soup.find('table', class_='normal-grid alternating-rows stats-table')

if table:
    # Extract table headers
    headers = [th.text.strip() for th in table.find('thead').find_all('th')]
    headers.insert(1, "Team Link")  # Adding extra column for team link

    # Extract table rows
    data = []
    for row in table.find('tbody').find_all('tr'):
        cells = row.find_all('td')
        row_data = []
        for i, cell in enumerate(cells):
            # If it's the first cell, extract team name and link from 'name-subcontainer'
            if i == 0:
                name_container = cell.find('div', class_='name-subcontainer')
                if name_container:
                    team_name = name_container.text.strip()
                    team_link_tag = name_container.find('a')
                    team_link = team_link_tag['href'] if team_link_tag else ''
                else:
                    team_name = cell.text.strip()
                    team_link = ''
                row_data.append(team_name)
                row_data.append(team_link)  # Add team link separately
            else:
                row_data.append(cell.text.strip())
        data.append(row_data)


    elo_data = pd.DataFrame(data, columns=[headers])
    elo_data.columns = elo_data.columns.get_level_values(0)
    elo_data = elo_data.drop_duplicates(subset='Team', keep='first')
    elo_data = elo_data.astype({col: 'str' for col in elo_data.columns if col not in ['ELO', 'Rank']})
    elo_data['ELO'] = elo_data['ELO'].astype(float, errors='ignore')
    elo_data['Rank'] = elo_data['Rank'].astype(int, errors='ignore')
    elo_data.rename(columns={'Rank': 'ELO_Rank'}, inplace=True)

else:
    print("Table not found on the page.")

# --- Team Name Replacements (Maps to be the same as teams on NCAA site) ---
team_replacements = {
    'North Carolina St.': 'NC State',
    'Southern Miss': 'Southern Miss.',
    'USC': 'Southern California',
    'Dallas Baptist': 'DBU',
    'Charleston': 'Col. of Charleston',
    'Georgia Southern': 'Ga. Southern',
    'UNCG': 'UNC Greensboro',
    'East Tennessee St.': 'ETSU',
    'Lamar': 'Lamar University',
    "Saint Mary's College": "Saint Mary's (CA)",
    'Western Kentucky': 'Western Ky.',
    'FAU': 'Fla. Atlantic',
    'Connecticut': 'UConn',
    'Southeast Missouri': 'Southeast Mo. St.',
    'Alcorn St.': 'Alcorn',
    'Appalachian St.': 'App State',
    'Arkansas-Pine Bluff': 'Ark.-Pine Bluff',
    'Army': 'Army West Point',
    'Cal St. Bakersfield': 'CSU Bakersfield',
    'Cal St. Northridge': 'CSUN',
    'Central Arkansas': 'Central Ark.',
    'Central Michigan': 'Central Mich.',
    'Charleston Southern': 'Charleston So.',
    'Eastern Illinois': 'Eastern Ill.',
    'Eastern Kentucky': 'Eastern Ky.',
    'Eastern Michigan': 'Eastern Mich.',
    'Fairleigh Dickinson': 'FDU',
    'Grambling St.': 'Grambling',
    'Incarnate Word': 'UIW',
    'Long Island': 'LIU',
    'Maryland Eastern Shore': 'UMES',
    'Middle Tennessee': 'Middle Tenn.',
    'Mississippi Valley St.': 'Mississippi Val.',
    "Mount Saint Mary's": "Mount St. Mary's",
    'North Alabama': 'North Ala.',
    'North Carolina A&T': 'N.C. A&T',
    'Northern Colorado': 'Northern Colo.',
    'Northern Kentucky': 'Northern Ky.',
    'Prairie View A&M': 'Prairie View',
    'Presbyterian College': 'Presbyterian',
    'Saint Bonaventure': 'St. Bonaventure',
    "Saint John's": "St. John's (NY)",
    'Sam Houston St.': 'Sam Houston',
    'Seattle University': 'Seattle U',
    'South Carolina Upstate': 'USC Upstate',
    'South Florida': 'South Fla.',
    'Southeastern Louisiana': 'Southeastern La.',
    'Southern': 'Southern U.',
    'Southern Illinois': 'Southern Ill.',
    'Stephen F. Austin': 'SFA',
    'Tennessee-Martin': 'UT Martin',
    'Texas A&M-Corpus Christi': 'A&M-Corpus Christi',
    'UMass-Lowell': 'UMass Lowell',
    'UTA': 'UT Arlington',
    'Western Carolina': 'Western Caro.',
    'Western Illinois': 'Western Ill.',
    'Western Michigan': 'Western Mich.',
    'Albany': 'UAlbany',
    'Southern Indiana': 'Southern Ind.',
    'Queens': 'Queens (NC)',
    'Central Connecticut': 'Central Conn. St.',
    'Saint Thomas': 'St. Thomas (MN)',
    'Northern Illinois': 'NIU',
    'UMass': 'Massachusetts',
    'Loyola-Marymount': 'LMU (CA)'
}

# Apply team name cleanup
elo_data = clean_team_names(elo_data)
projected_rpi = clean_team_names(projected_rpi)
live_rpi = clean_team_names(live_rpi)




#### NCAA Site stuff ####
# --- NCAA Stats Dropdown ---
base_url = "https://www.ncaa.com"
soup = get_soup(f"{base_url}/stats/baseball/d1")
dropdown = soup.find("select", {"id": "select-container-team"})
stat_links = {
    option.text.strip(): base_url + option["value"]
    for option in dropdown.find_all("option") if option.get("value")
}
# use stat_links for urls to Team Stats

# --- NCAA RPI Table ---
rpi_url = "https://www.ncaa.com/rankings/baseball/d1/rpi"
rpi_soup = get_soup(rpi_url)
table = rpi_soup.find("table", class_="sticky")

if table:
    headers = [th.text.strip() for th in table.find_all("th")]
    data = [
        [td.text.strip() for td in row.find_all("td")]
        for row in table.find_all("tr")[1:]
    ]
    rpi = pd.DataFrame(data, columns=headers).drop(columns=["Previous"])
    rpi.rename(columns={"School": "Team"}, inplace=True)
else:
    print("NCAA RPI Table not found.")
    rpi = pd.DataFrame()


####################### CONFIG #######################

# Must be defined elsewhere in your script:
# - stat_links: dict of stat_name -> URL
# - get_soup(url): function that returns BeautifulSoup of given URL

####################### Core Stat Fetching #######################

# returns a dataframe for a specific stat name in stat_links
def get_stat_dataframe(stat_name):
    if stat_name not in stat_links:
        print(f"Stat '{stat_name}' not found. Available stats: {list(stat_links.keys())}")
        return None

    all_data = []
    page_num = 1

    while True:
        url = stat_links[stat_name]
        if page_num > 1:
            url = f"{url}/p{page_num}"

        try:
            soup = get_soup(url)
            table = soup.find("table")
            if not table:
                break

            headers = [th.text.strip() for th in table.find_all("th")]
            data = []
            for row in table.find_all("tr")[1:]:
                cols = row.find_all("td")
                data.append([col.text.strip() for col in cols])

            all_data.extend(data)

        except requests.exceptions.HTTPError:
            break
        except Exception as e:
            print(f"Error for {stat_name}, page {page_num}: {e}")
            break

        page_num += 1

    if all_data:
        df = pd.DataFrame(all_data, columns=headers)
        for col in df.columns:
            if col != "Team":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    else:
        return None

####################### Threading #######################

# threaded stat retrieval
def threaded_stat_fetch(stat_names, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_stat = {
            executor.submit(get_stat_dataframe, stat): stat
            for stat in stat_names
        }
        results = {}
        for future in as_completed(future_to_stat):
            stat = future_to_stat[future]
            try:
                results[stat] = future.result()
            except Exception as e:
                print(f"Failed to fetch {stat}: {e}")
    return results

####################### Utility #######################

# for the "Home Runs per Game" stat, not in use here
def clean_duplicates(df, group_col, min_col):
    duplicates = df[df.duplicated(group_col, keep=False)]
    filtered = duplicates.loc[duplicates.groupby(group_col)[min_col].idxmin()]
    cleaned = df[~df[group_col].isin(duplicates[group_col])]
    return pd.concat([cleaned, filtered], ignore_index=True)

####################### Transform Config #######################

# define the stats you want and any transformations you want to make to it
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

# cleaning, merging it all into one dataframe. Calculating OPS and PYTHAG
def clean_and_merge(stats_raw, transforms_dict):
    dfs = []
    for stat, df in stats_raw.items():
        if df is not None and stat in transforms_dict:
            df["Team"] = df["Team"].str.strip()
            df_clean = transforms_dict[stat](df)
            dfs.append(df_clean)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="Team", how="inner")

    merged = merged.loc[:, ~merged.columns.duplicated()].sort_values('Team').reset_index(drop=True)
    merged["OPS"] = merged["SLG"] + merged["OBP"]
    merged["PYTHAG"] = round(
        (merged["RS"] ** 1.83) / ((merged["RS"] ** 1.83) + (merged["RA"] ** 1.83)), 3
    )
    return merged

####################### Run It #######################

# Stat pull for the stats in STAT_TRANSFORMS
stat_list = list(STAT_TRANSFORMS.keys())
raw_stats = threaded_stat_fetch(stat_list, max_workers=10)
baseball_stats = clean_and_merge(raw_stats, STAT_TRANSFORMS)