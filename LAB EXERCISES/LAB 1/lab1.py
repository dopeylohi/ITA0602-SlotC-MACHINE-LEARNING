# ==============================
# LAB 1 - EXPERIMENT 1
# FIND-S ALGORITHM
# ==============================

import csv
import io

data = """Sky,AirTemp,Humidity,Wind,Water,Forecast,EnjoySport
Sunny,Warm,Normal,Strong,Warm,Same,Yes
Sunny,Warm,High,Strong,Warm,Same,Yes
Rainy,Cold,High,Strong,Warm,Change,No
Sunny,Warm,High,Strong,Cool,Change,Yes
"""

reader = csv.reader(io.StringIO(data))
header = next(reader)
rows = list(reader)

hypothesis = ['0'] * (len(header) - 1)

for row in rows:
    if row[-1] == 'Yes':
        for i in range(len(hypothesis)):
            if hypothesis[i] == '0':
                hypothesis[i] = row[i]
            elif hypothesis[i] != row[i]:
                hypothesis[i] = '?'

print("Final Hypothesis:", hypothesis)


# ==============================
# LAB 1 - EXPERIMENT 2
# CANDIDATE ELIMINATION
# ==============================

import csv
import io

data = """Sky,AirTemp,Humidity,Wind,Water,Forecast,EnjoySport
Sunny,Warm,Normal,Strong,Warm,Same,Yes
Sunny,Warm,High,Strong,Warm,Same,Yes
Rainy,Cold,High,Strong,Warm,Change,No
Sunny,Warm,High,Strong,Cool,Change,Yes
"""

reader = csv.reader(io.StringIO(data))
header = next(reader)
rows = list(reader)

n = len(header) - 1
S = ['0'] * n
G = [['?'] * n]

for row in rows:
    x = row[:-1]
    target = row[-1]

    if target == 'Yes':
        G = [g for g in G if all(
            g[i] == '?' or g[i] == x[i] for i in range(n)
        )]

        for i in range(n):
            if S[i] == '0':
                S[i] = x[i]
            elif S[i] != x[i]:
                S[i] = '?'

    else:
        new_G = []

        for g in G:
            for i in range(n):
                if g[i] == '?':
                    if S[i] != '?' and S[i] != '0':
                        h = g.copy()
                        h[i] = S[i]
                        new_G.append(h)

        G = new_G

print("Specific Boundary:", S)
print("General Boundary:", G)
