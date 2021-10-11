Try using: "SELECT Track.title, Album.title, Artist.name FROM Track JOIN Album JOIN Artist ON Track.album_id=Album.id AND Album.artist_id = Artist.id"

# This is a Sqllite 3 database of an XML file found in I-tunes music library containing details about the songs such as Artist, Tracks, number of times a song has been played etc
# This database contains 3 dependent tables (Track --> Album --> Artist)
# databse has been architechtures keeping in mind the Normalisation Process of Relational databases
# Python code ensures that there is no Duplication of entries so as to remove redundancy
