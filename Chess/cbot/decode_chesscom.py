import requests
import re

# The PGN data with unicode escape sequences (replace this string with actual data you have)
pgn_data = r'CurrentPosition\u0020\u0022R5k1\/8\/6KP\/8\/8\/8\/8\/8\u0020b\u0020\u002D\u0020\u002D\u0022\u005D\n\u005BTimezone\u0020\u0022UTC\u0022\u005D\n\u005BECO\u0020\u0022B51\u0022\u005D\n\u005BECOUrl\u0020\u0022https'

# Decode escaped unicode characters to get the actual PGN string
decoded_pgn = pgn_data.encode().decode('unicode-escape')

# Remove any escaping from slashes and correct the FEN string
corrected_fen = decoded_pgn.replace('\/', '/')

# Extract the FEN part using a regular expression
fen_match = re.search(r'CurrentPosition\s"(.+?)"', corrected_fen)

if fen_match:
    fen = fen_match.group(1)
    print("FEN:", fen)

    # URL encode the FEN for use in an API request
    encoded_fen = requests.utils.quote(fen)

    # The API URL with the encoded FEN
    api_url = f'https://www.chessdb.cn/cdb.php?action=querypv&board={encoded_fen}&json=1'

    # Make the API request
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        best_move = data.get('bestmove')
        depth = data.get('depth')
        principal_variation = data.get('pv')

        # Output the information retrieved from the API
        print('Best Move:', best_move)
        print('Depth:', depth)
        print('Principal Variation:', principal_variation)
    else:
        print('Failed to retrieve data from the API:', response.status_code)
else:
    print('Could not find a valid FEN in the PGN data')