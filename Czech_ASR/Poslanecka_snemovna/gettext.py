# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from bs4 import BeautifulSoup
import urllib.request
import requests
import sys

arg = sys.argv[1]
meeting_num = sys.argv[2]
# We need to distinguish between single digit and more than 2 digits
if int(arg) < 10:
    url = f"https://www.psp.cz/eknih/2017ps/stenprot/0{meeting_num}schuz/s0{meeting_num}00{arg}.htm"
elif int(arg) > 99:
    url = f"https://www.psp.cz/eknih/2017ps/stenprot/0{meeting_num}schuz/s0{meeting_num}{arg}.htm"
else:
    url = f"https://www.psp.cz/eknih/2017ps/stenprot/0{meeting_num}schuz/s0{meeting_num}0{arg}.htm"

# Request URL here
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
