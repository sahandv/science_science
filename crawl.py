#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:39:55 2020

@author: github.com/sahandv
"""


import requests
from bs4 import BeautifulSoup

URL = 'https://en.wikipedia.org/wiki/Cybernetics'
page = requests.get(URL)

soup = BeautifulSoup(page.content, 'html.parser')
content = soup.find(id='mw-content-text').prettify()