import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'sg':1.010, 'al':2.0, 'sc':1.4,'hemo':11.6,'pcv':19,'htn':0})

print(r.json())