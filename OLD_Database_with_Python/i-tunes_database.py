#!/usr/bin/env python
# coding: utf-8

# In[89]:


import sqlite3
import xml.etree.ElementTree as ET


# In[92]:


conn = sqlite3.connect('apple_i-tunes.sqllite3')
cur = conn.cursor()


# In[93]:


cur.executescript('''
DROP TABLE IF EXISTS Artist;
DROP TABLE IF EXISTS Album;
DROP TABLE IF EXISTS Track;

CREATE TABLE Artist(
id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
name TEXT UNIQUE);

CREATE TABLE Album(
id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
artist_id INTEGER,
title TEXT UNIQUE);

CREATE TABLE Track(
id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
title TEXT UNIQUE,
album_id INTEGER,
Count INTEGER,
rating INTEGER,
len INTEGER
);
''')


# In[94]:


fname = input('Enter the file:')

if len(fname) < 1:
    fname = 'Library.xml'

print(fname)


# In[95]:


def lookup(d,key):
    found = False
    for child in d:
        if found:
            return child.text
        if child.tag == 'key' and child.text == key:
            found = True
    return None


# In[96]:


stuff = ET.parse(fname)
all = stuff.findall('dict/dict/dict')
print('dict len:', len(all))
for entry in all:
    if((lookup(entry, 'Name'))  is None):
        continue
    name = lookup(entry,'Name')
    artist = lookup(entry,'Artist')
    album = lookup(entry,'Album')
    count = lookup(entry,'Play Count')
    rating = lookup(entry,'Rating')
    length = lookup(entry,'Length')
    
    if name is None or artist is None or album is None:
        continue
        
    cur.execute('INSERT OR IGNORE INTO Artist (name) VALUES (?)', (artist, ))
    cur.execute('SELECT id FROM Artist WHERE name = ?', (artist,))
    artist_id = cur.fetchone()[0]
    
    cur.execute('INSERT OR IGNORE INTO Album (title,artist_id) VALUES (?,?)', (album,artist_id,))
    cur.execute('SELECT id FROM Album WHERE title = ?', (album,))
    album_id = cur.fetchone()[0]
    
    cur.execute('INSERT OR REPLACE INTO Track(title,album_id, len,rating,count) VALUES ( ?,?,?,?,?)', (name, album_id, length, rating, count))
    
    conn.commit()
    
    


# In[ ]:




