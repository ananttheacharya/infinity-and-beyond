import urllib.request
import xml.etree.ElementTree as ET
import os

query = 'all:"gas turbine" AND all:"physics-informed"'
url = f'http://export.arxiv.org/api/query?search_query={urllib.parse.quote(query)}&start=0&max_results=3'

response = urllib.request.urlopen(url)
data = response.read()

root = ET.fromstring(data)
ns = {'atom': 'http://www.w3.org/2005/Atom'}

os.makedirs('public/papers', exist_ok=True)

for i, entry in enumerate(root.findall('atom:entry', ns)):
    title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
    summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
    
    # Find PDF link
    pdf_url = None
    for link in entry.findall('atom:link', ns):
        if link.attrib.get('title') == 'pdf':
            pdf_url = link.attrib['href']
            break
            
    if pdf_url:
        filename = f"paper_{i+1}.pdf"
        print(f"Downloading: {title}\nTo: {filename}\n")
        urllib.request.urlretrieve(pdf_url + ".pdf", f"public/papers/{filename}")
        
        # Write a summary text file
        with open(f"public/papers/paper_{i+1}_summary.txt", 'w', encoding='utf-8') as f:
            f.write(f"Title: {title}\n\nSummary:\n{summary}")
