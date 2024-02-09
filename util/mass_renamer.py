folder_path = '/tiles/'
substring_to_remove = '.tif'

import os
# Reiterating the approach to ensure it's clear the code is intended for both files and folders

# Walking through the directory, including subdirectories
for root, dirs, files in os.walk(folder_path, topdown=False):
    # Renaming files
    for name in files:
        if substring_to_remove in name:
            new_name = name.replace(substring_to_remove, '')
            os.rename(os.path.join(root, name), os.path.join(root, new_name))
            print(f'Renamed file {name} to {new_name}')

    # Renaming directories
    for name in dirs:
        if substring_to_remove in name:
            new_name = name.replace(substring_to_remove, '')
            os.rename(os.path.join(root, name), os.path.join(root, new_name))
            print(f'Renamed directory {name} to {new_name}')

