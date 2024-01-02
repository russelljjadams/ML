# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:49:52 2023

@author: rjadams
"""

import requests

# Define the URL
url = 'https://www.chessdb.cn/cdb.php?action=querypv&board=r1bqk1nr/pppp1ppp/2n5/1Bb1p3/4P3/5N2/PPPP1PPP/RNBQK2R%20w%20KQkq%20-%200%201&json=1'

# Make a GET request to fetch the data
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # Access the information you need
    best_move = data.get('bestmove')
    depth = data.get('depth')
    principal_variation = data.get('pv')

    # Print the results
    print(f'Best Move: {best_move}')
    print(f'Depth: {depth}')
    print(f'Principal Variation: {principal_variation}')
else:
    print('Failed to retrieve data')