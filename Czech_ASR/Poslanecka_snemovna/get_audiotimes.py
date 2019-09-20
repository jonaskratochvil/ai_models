#!/usr/bin/env python3
from bs4 import BeautifulSoup
import urllib.request
import requests
import sys

arg = sys.argv[1]
month = sys.argv[2]
year = sys.argv[3]

url = f"https://www.psp.cz/eknih/2017ps/audio/{year}/{month}/{arg}/"
res = requests.get(url)
html_page = res.content

soup = BeautifulSoup(html_page, 'html.parser')
text = soup.find_all(text=True)

output = ''
blacklist = [
    '[document]',
    'noscript',
    'header',
    'html',
    'meta',
    'head',
    'input',
    'script',
    'body'
    # there may be more elements you don't want, such as "style", etc.
]

for t in text:
    if t.parent.name not in blacklist:
        output += '{} '.format(t)

fnl_txt = ''
for line in output:
    fnl_txt += line

sys.stdout.write(fnl_txt)
