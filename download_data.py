import os
import urllib.request
import json

# List of Wikipedia pages to fetch
titles = [
    "Steins;Gate", "Fate/stay_night", "Clannad_(visual_novel)", "The_House_in_Fata_Morgana",
    "Higurashi_When_They_Cry", "Umineko_When_They_Cry", "Doki_Doki_Literature_Club!",
    "Phoenix_Wright:_Ace_Attorney", "999:_Nine_Hours,_Nine_Persons,_Nine_Doors",
    "Solo_Leveling", "Tower_of_God", "Omniscient_Reader's_Viewpoint", "The_Beginning_After_the_End",
    "Lore_Olympus", "Visual_novel", "Digital_comic", "Webtoon", "Manga", "Manhwa", "Manhua"
]

output_dir = "data/sample_docs"
os.makedirs(output_dir, exist_ok=True)

for title in titles:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            extract = data.get('extract', '')
            clean_title = title.replace(':', '_').replace('/', '_').lower()
            
            with open(f"{output_dir}/{clean_title}.txt", "w", encoding="utf-8") as f:
                f.write(f"# {data.get('title', title)}\n\n{extract}")
            print(f"Saved: {clean_title}.txt")
    except Exception as e:
        print(f"Failed to fetch {title}: {e}")