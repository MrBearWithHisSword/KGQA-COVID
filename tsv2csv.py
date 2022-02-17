import re
from tqdm import tqdm

files_names = ['data/merged/merged-kg_nodes','data/merged/merged-kg_edges' ]
for name in files_names:
    # reading given tsv file
    with open(name+".tsv", 'r') as myfile:  
      with open(name+".csv", 'w') as csv_file:
        for line in tqdm(myfile):

          # Replace every tab with comma
          fileContent = re.sub("\t", ",", line)

          # Writing into csv file
          csv_file.write(fileContent)
    
# output
print("Successfully made csv file")