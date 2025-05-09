import requests
from bs4 import BeautifulSoup
from PIL import Image # type: ignore
from io import BytesIO # type: ignore
import pandas as pd

# --- ELO Ratings ---
url = 'https://www.warrennolan.com/baseball/2025/elo'

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('table', class_='normal-grid alternating-rows stats-table')

if table:
    headers = [th.text.strip() for th in table.find('thead').find_all('th')]
    headers.insert(1, "Team Link")  # Adding extra column for team link
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

# Pull URL
# You need elo_data without the team replacements - which is the Warren Nolan notation of teams
BASE_URL = "https://www.warrennolan.com"
team = "Arkansas"
team_url = BASE_URL + elo_data[elo_data['Team'] == team]['Team Link'].values[0]
response = requests.get(team_url)
soup = BeautifulSoup(response.text, 'html.parser')
img_tag = soup.find("img", class_="team-menu__image")
img_src = img_tag.get("src")
image_url = BASE_URL + img_src
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))
# Can then use img on matplotlib or whatever you want to use