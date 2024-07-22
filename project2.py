import urllib.parse
import requests

from bs4 import BeautifulSoup
import os
import urllib

url = "https://docs.llamaindex.ai/en/stable/"

outputdir = "./llamaindex_docs/"

os.makedirs(outputdir, exist_ok=True)

response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

links = soup.find_all("a", href=True)

print(len(links))

for key, link in enumerate(links):

    href = link["href"]

    if not href.startswith("http"):
        href = urllib.parse.urljoin(url, href)

    file_response = requests.get(href)

    file_name = f"{outputdir}{key+1}.html"

    with open(file_name, "w", encoding="utf-8") as file:
        print("Saving doc")

        file.write(file_response.text)
