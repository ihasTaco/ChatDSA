import os
from summa import keywords
import json

context_dir = "context"
keywords_dict = {}

# Loop through all files in the context folder
for filename in os.listdir(context_dir):
    filepath = os.path.join(context_dir, filename)
    
    # Open the file and extract keywords
    with open(filepath, "r") as f:
        context_text = f.read()
        extracted_keywords = keywords.keywords(context_text)
    
    # Add file name and extracted keywords to dictionary
    keywords_dict[filename] = extracted_keywords.split("\n")
    
# Save dictionary as JSON file
with open("keywords.json", "w") as f:
    json.dump(keywords_dict, f, indent=4)