import warnings
from text_unidecode import unidecode
from collections import deque

warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from node2vec import Node2Vec

sns.set_style('whitegrid')

data = pd.read_csv('./FullData.csv', usecols=['Name', 'Club', 'Club_Position', 'Rating'])
# print(data['Name'])

data.columns = list(map(str.lower, data.columns))
reformat_string = lambda x: unidecode(str.lower(x).replace(' ', '_'))
# print(reformat_string)

data['name'] = data['name'].apply(reformat_string)
data['club'] = data['club'].apply(reformat_string)

# print(data['name'])

# Lowercase position
data['club_position'] = data['club_position'].str.lower()

# Ignore substitutes and reserves
data = data[(data['club_position'] != 'sub') & (data['club_position'] != 'res')]

# Fix lcm rcm -> cm cm
fix_positions = {'rcm' : 'cm', 'lcm': 'cm', 'rcb': 'cb', 'lcb': 'cb', 'ldm': 'cdm', 'rdm': 'cdm'}
data['club_position'] = data['club_position'].apply(lambda x: fix_positions.get(x, x))

# For example sake we will keep only 7 clubs
clubs = {'real_madrid', 'manchester_utd',
         'manchester_city', 'chelsea', 'juventus',
         'fc_bayern', 'napoli'}

data = data[data['club'].isin(clubs)]

# Verify we have 11 player for each team
assert all(n_players == 11 for n_players in data.groupby('club')['name'].nunique())

print(data)






