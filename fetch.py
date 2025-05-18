import requests
from bs4 import BeautifulSoup

response = requests.get("https://www.kongju.ac.kr/bbs/KNU/2132/409495/artclView.do").text
soup = BeautifulSoup(response, "html.parser")
notices = [tag.text.strip() for tag in soup.select(".view-con")]
print(notices)
