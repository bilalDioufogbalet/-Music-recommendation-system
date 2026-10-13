import requests
import json
import pandas as pd
import time 
from paths import RAW_DIR


# Identification de notre application auprès de ListenBrainz
headers = {
    "User-Agent": "MusicRecoSystem/0.1 (bilaldioufogbalet@gmail.com)"
}

def get_with_retry(url, params=None, max_attempts=3):

    for attempt in range(max_attempts):

        try:
            time.sleep(1.1)

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                return response

            print(f"Tentative {attempt + 1} : erreur {response.status_code}")

        except requests.exceptions.RequestException as error:
            print(f"Tentative {attempt + 1} : {error}")

        time.sleep(2 * (attempt + 1))

    return None


# Artistes utilisés comme points de départ pour trouver des utilisateurs
anchor_artists = {
    "Dave": "f93bac9e-b20a-403e-abc9-06a2fc151df1",
    
    "J Hus" : "eca85dd0-f1fb-4e83-8c47-61403f479254",

    "Knucks": "43806ec0-bc26-4df7-8c41-8341d2354c9c",

    "Jim Legxacy": "9ca998cf-c1cd-493b-9a5d-a32fd33859ea",

    "Drake" : "9fff2f8a-21e6-47de-a2b8-7f449929d43f",
    
    "Kendrick Lamar" : "381086ea-f511-4aba-bdf9-71c753dc5077",
    
    "Travis Scott" : "e4a51f17-a57b-47b1-b37b-f552d0f8e9e6",
    
    "J. Cole" : "875203e1-8e58-4b86-8dcb-7190faf411c5",
    
    "Central Cee": "b0337af1-8d93-4671-b6c9-ba306bf942bf",
    
    "Little Simz" : "3cdb40fe-a63e-4bb9-b40d-17cda5f50979",
    
    "Skepta" : "38474d18-9204-4f60-a137-1fcc6750b869",
    
    "Bonobo " : "9a709693-b4f8-4da9-8cc1-038c911a61be",
    
    "The Weeknd" : "c8b03190-306c-4120-bb0b-6f2ebfc06ea9",

    "SZA" : "272989c8-5535-492d-a25c-9f58803e027f",

    "Frank Ocean" : "e520459c-dff4-491d-a6e4-c97be35e0044",
    
    "Odeal" : "47e10b14-4b74-4650-bc2b-34424c1c6bcb",

    "Jorja Smith" : "d62614aa-ef2c-44cd-8a7b-91292755a1d0",

    "Burna Boy" : "78a19169-ac75-4868-b504-7e2e073118e0",

    "Tyler, The Creator": "f6beac20-5dfe-4d1f-ae02-0b0a740aafd6",

    "A$AP Rocky": "25b7b584-d952-4662-a8b9-dd8cdfbfeb64",

    "Future": "48262e82-db9f-4a92-b650-dfef979b73ec",

    "Smino": "96800e05-9924-4321-859a-76e0d4750f2f",

    "Brent Faiyaz": "201412c6-7ced-472b-87a9-9259ec3c07d3",

    "Daniel Caesar": "b46e298d-d3a5-4695-a03e-63e8c523b23e",

    "Summer Walker": "9167f739-da47-4495-9d63-9c13b40114ac",

    "21 Savage" : "37b2cb82-ef79-4d46-a184-a549450aa231",

    "Young Thug" : "800760de-bdf8-43a2-8fe0-44a2401a5515",

    "Gunna" : "100b8734-f4f0-4536-9960-47f7c59d1b4c",

    "Lil Baby" : "d42a5768-ada1-4fb6-85fd-f61aba6264e0",

    "Lil Uzi Vert" : "8837b875-3689-4646-b3fd-d8b53815c7a8",

    "Playboi Carti" : "2baf3276-ed6a-4349-8d2e-f4601e7b2167",

    "PARTYNEXTDOOR": "f23ef341-33bd-47c1-b83c-846a78581f05",

    "Bryson Tiller": "d8fd8d9b-473b-4f06-83c8-869b1bb9de89",

    "Giveon": "ad322b8b-31b2-437d-8fd8-1af13c1d7b91",

    "Miguel": "285e1285-aecc-478f-a8da-0c8cf69e1217",

    "Kali Uchis": "d0c7656d-8169-4f77-9dbe-b8f24e40105d",

    "Snoh Aalegra": "636e0fb6-f2ad-45fc-b385-4d994665e7e3",

    "Ravyn Lenae": "94c64122-dbd4-4927-b2fa-f204b0b4b0c2",

    "Steve Lacy": "b29020d6-5dbf-4c99-b16f-c009f8a712ff",

    "The Internet": "45804465-4271-4e6a-b881-ce668ef09530",
    
    "Don Toliver": "a0723a3c-4135-438e-85c5-012712144ede",

    "Baby Keem": "81d12ca7-5b89-4d21-8736-860b7400b4eb",

    "Metro Boomin": "59db3d82-86ea-451f-881f-dffc8ec387c9",

    "Yeat": "9e6a6a2f-f696-4cc0-87f1-e4f712e46802"


}


# Nombre d'utilisateurs que l'on veut récupérer
NB_USERS = 100


# On stocke ici tous les utilisateurs similaires trouvés
all_candidates = []


# ---------------------------------------------------------
# 1. Trouver les top listeners de chaque artiste
# ---------------------------------------------------------

for artist_name, artist_mbid in anchor_artists.items():

    url = f"https://api.listenbrainz.org/1/stats/artist/{artist_mbid}/listeners"

    
    response = get_with_retry(url)

    if response is None:

        print(f"Impossible de récupérer les top listeners de {artist_name}")

        continue

    print(artist_name, response.status_code)

    if response.status_code != 200:
        print(f"Impossible de traiter {artist_name}")
        continue

    data = response.json()

    listeners = data["payload"]["listeners"]

    print(f"{len(listeners)} top listeners trouvés pour {artist_name}")

    top_users = []

    for listener in listeners:
        top_users.append(listener["user_name"])


    # ---------------------------------------------------------
    # 2. Trouver les utilisateurs similaires aux top listeners
    # ---------------------------------------------------------

    for user in top_users:

        url = f"https://api.listenbrainz.org/1/user/{user}/similar-users"
        

        response = get_with_retry(url)

        if response is None:

            print(f"Impossible de récupérer les utilisateurs similaires de {user}")

            continue

        print(user, response.status_code)

        if response.status_code != 200:
            continue

        similar_data = response.json()

        similar_users = similar_data["payload"]

        print(len(similar_users))

        for similar_user in similar_users:

            all_candidates.append({
                "anchor_artist": artist_name,
                "source_user": user,
                "user_name": similar_user["user_name"],
                "similarity": similar_user["similarity"]
            })


# ---------------------------------------------------------
# 3. Classer les utilisateurs trouvés
# ---------------------------------------------------------

candidates_df = pd.DataFrame(all_candidates)

candidate_groups = candidates_df.groupby("user_name")

candidate_stats = candidate_groups["similarity"].agg(["count", "mean"])


# On privilégie :
# 1. les utilisateurs trouvés plusieurs fois
# 2. puis ceux avec la meilleure similarité moyenne

ranked_candidates = candidate_stats.sort_values(
    ["count", "mean"],
    ascending=[False, False]
)


print("\nMeilleurs candidats :")
print(ranked_candidates.head(10))


# On sélectionne les NB_USERS meilleurs utilisateurs
selected_users = ranked_candidates.head(NB_USERS).index.tolist()

print(f"\n{len(selected_users)} utilisateurs sélectionnés")


# ---------------------------------------------------------
# 4. Télécharger les 100 dernières écoutes de chaque utilisateur
# ---------------------------------------------------------

for username in selected_users:

    file_path = RAW_DIR / f"{username}_listens.json"

    # Si l'utilisateur a déjà été téléchargé, on passe au suivant

    if file_path.exists():

        print(f"{username} déjà téléchargé")

        continue

    url = f"https://api.listenbrainz.org/1/user/{username}/listens"

    response = get_with_retry(
        url,
        params={"count": 100}
    )
    if response is None:

        print(f"Impossible de récupérer les écoutes de {username}")

        continue

    data = response.json()

    listens = data["payload"]["listens"]

    print(f"{username} : {len(listens)} écoutes récupérées")

    # ---------------------------------------------------------
    # 5. Sauvegarder les écoutes dans data/raw
    # ---------------------------------------------------------

    with open(file_path, "w", encoding="utf-8") as file:

        json.dump(data, file, indent=4, ensure_ascii=False)

    print(f"{username} sauvegardé")