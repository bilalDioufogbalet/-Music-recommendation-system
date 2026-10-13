import json 
from paths import RAW_DIR
from paths import PROCESSED_DIR
import pandas as pd 


#permet de lister tous les fichiers JSON dans le répertoire RAW_DIR qui se terminent par "_listens.json" et de les stocker dans une liste files. Ensuite, il affiche la liste des fichiers et le nombre total de fichiers trouvés.
files = list(RAW_DIR.glob("*_listens.json"))

rows = []
for file_path in files:
    
    with open(file_path, "r") as file:
        data = json.load(file) #permet de charger les données du fichier JSON dans une variable Python
        listens = data["payload"]["listens"]

        for listen in listens: 
            track_metadata = listen ["track_metadata"]
            mapping = track_metadata.get("mbid_mapping")

            if mapping is not None and mapping.get(("artist_mbids")):
                artist_mbids = mapping["artist_mbids"]
            else:
                artist_mbids = None
                

            user = listen["user_name"] 
            artist = track_metadata["artist_name"]
            track = track_metadata["track_name"]
            release = release = track_metadata.get("release_name") # car peut être manquant
            listened_at = listen["listened_at"]
            recording_msid = listen["recording_msid"]
            row = {
                "user" : user,
                "artist" : artist,
                "artist_mbids" : artist_mbids,
                "track" : track,
                "release" : release,
                "listened_at" : listened_at,
                # À activer plus tard pour gérer les potentiels doublons
                #"recording_msid" : recording_msid 

            }
            rows.append(row)

    print(file_path.name, len(listens))

df = pd.DataFrame(rows) #conversion de la liste de dictionnaires en DataFrame pandas

df["listened_at"] = pd.to_datetime(
    df["listened_at"], 
    unit ="s", 
    utc=True) #convertit la colonne "listened_at" en format datetime, en utilisant l'unité de temps "secondes" et en spécifiant que les dates sont en temps universel coordonné (UTC).




#Créer un chemin pour indiquer où on voudra enregistrer le fichier
output_path = PROCESSED_DIR / "listens.parquet"

df.to_parquet(output_path, index=False) #Permet de sauvergarder nos data sous la forme d'un parquet, cela nous permet de conserver la structure du DataFrame




