import requests
import time

artists = [
    # RAP US / TRAP

    "21 Savage",

    "Young Thug",

    "Gunna",

    "Lil Baby",

    "Lil Uzi Vert",

    "Playboi Carti",

    "Don Toliver",

    "Baby Keem",

    "Metro Boomin",

    "Yeat",

    # HIP-HOP ALTERNATIF

    "JID",

    "Denzel Curry",

    "Joey Bada$$",

    "Mac Miller",

    "Isaiah Rashad",

    "Vince Staples",

    "Earl Sweatshirt",

    "Freddie Gibbs",

    "JPEGMAFIA",

    "Aminé",

    # R&B / NEO-SOUL

    "PARTYNEXTDOOR",

    "Bryson Tiller",

    "Giveon",

    "Miguel",

    "Kali Uchis",

    "Snoh Aalegra",

    "Ravyn Lenae",

    "Steve Lacy",

    "Cleo Sol",

    "The Internet"
    ]

url = "https://musicbrainz.org/ws/2/artist/"

for artist in artists:

    response = requests.get(

        "https://musicbrainz.org/ws/2/artist/",

        params={

            "query": f'artist:"{artist}"',

            "fmt": "json",

            "limit": 3

        },

        headers={

            "User-Agent": "MusicRecoSystem/0.1 (contact: ton-email@example.com)"

        },

        timeout=15

    )

    response.raise_for_status()

    results = response.json().get("artists", [])

    if results:

        print(artist, ":", results[0]["id"])

    time.sleep(1.1)