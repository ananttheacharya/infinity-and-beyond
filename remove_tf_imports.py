import json
import os

notebook_path = r"c:\Users\anant\Downloads\zero and already behind\repos\digital-twin-for-aircraft-engine-maintenance\Deep_learning_model.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The first cell is at index 0
source = nb['cells'][0]['source']
new_source = []
for line in source:
    if 'tensorflow' in line:
        new_source.append('# ' + line)
    else:
        new_source.append(line)

nb['cells'][0]['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully commented out TensorFlow imports in the notebook.")
