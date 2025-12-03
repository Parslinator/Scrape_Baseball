import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
import pandas as pd
import matplotlib.font_manager as fm
custom_font = fm.FontProperties(fname="./PEAR/trebuc.ttf")
plt.rcParams['font.family'] = custom_font.get_name()
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
from datetime import datetime
import pytz
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

def get_soup(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")

def download_team_logos(url, logo_dir="./PEAR/PEAR Baseball/logos"):
    """Download team logos from a website's table"""
    os.makedirs(logo_dir, exist_ok=True)
    
    all_logos = []
    page_num = 1

    while page_num < 7:
        page_url = url if page_num == 1 else f"{url}/p{page_num}"
        
        try:
            soup = get_soup(page_url)
            table = soup.find("table")
            if not table:
                break

            for row in table.find_all("tr")[1:]:  # Skip header row
                # Find team name from <a> tag with class="school"
                team_link = row.find("a", class_="school")
                if not team_link:
                    continue
                
                team_name = team_link.text.strip()
                
                # Find image in the row
                img_tag = row.find("img")
                if img_tag and img_tag.get("src"):
                    img_src = img_tag.get("src")
                    # Handle relative URLs
                    if img_src.startswith("//"):
                        img_src = "https:" + img_src
                    elif img_src.startswith("/"):
                        img_src = BASE_URL + img_src
                    
                    all_logos.append((team_name, img_src))

        except requests.exceptions.HTTPError:
            break
        except Exception as e:
            print(f"Error on page {page_num}: {e}")
            break

        page_num += 1

    # Download all logos
    def save_logo(team_data):
        team_name, img_url = team_data
        
        try:
            # Add headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            img_response = requests.get(img_url, timeout=10, headers=headers)
            img_response.raise_for_status()
            
            # Check if it's an SVG file
            if img_url.endswith('.svg') or 'svg' in img_response.headers.get('content-type', ''):
                # Save SVG directly
                file_path = os.path.join(logo_dir, f"{team_name}.svg")
                with open(file_path, 'wb') as f:
                    f.write(img_response.content)
                print(f"Saved SVG logo for {team_name}")
                return team_name, True
            else:
                # Handle raster images (PNG, JPG, etc.)
                img = Image.open(BytesIO(img_response.content))
                
                # Convert to RGBA
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # Upscale for better quality
                upscale_factor = 2
                new_size = (img.width * upscale_factor, img.height * upscale_factor)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save with high quality
                file_path = os.path.join(logo_dir, f"{team_name}.png")
                img.save(file_path, format='PNG', optimize=False, compress_level=0)
                print(f"Saved PNG logo for {team_name}")
                return team_name, True
            
        except Exception as e:
            print(f"Error downloading logo for {team_name}: {e}")
            print(f"  URL: {img_url}")
            return team_name, False
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(save_logo, all_logos))
    
    print(f"\nSuccessfully saved {sum(success for _, success in results)}/{len(all_logos)} logos.")

from cairosvg import svg2png
import glob

def convert_all_svgs_to_png(logo_dir="./PEAR/PEAR Baseball/logos"):
    """Convert all SVG logos to high-quality PNG and delete SVG files"""
    svg_files = glob.glob(os.path.join(logo_dir, "*.svg"))
    
    for svg_path in svg_files:
        png_path = svg_path.replace('.svg', '.png')
        # scale=4 gives you 4x resolution for high quality
        svg2png(url=svg_path, write_to=png_path, scale=4)
        print(f"Converted {os.path.basename(svg_path)}")
        
        # Delete the SVG file after conversion
        os.remove(svg_path)
        print(f"Deleted {os.path.basename(svg_path)}")
    
    print(f"\nConverted and deleted {len(svg_files)} SVG files")

# Usage:
# download_team_logos("https://www.ncaa.com/stats/baseball/...")
BASE_URL = "https://www.ncaa.com"
download_team_logos("https://www.ncaa.com/stats/baseball/d1/current/team/496")
convert_all_svgs_to_png()