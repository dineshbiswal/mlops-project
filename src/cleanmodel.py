import os
# Specify the folder path
folder_path = "models/"
# Iterate through the files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Check if it's a file and delete it
    if os.path.isfile(file_path):
        os.remove(file_path)