import os

def print_directory_structure(root_path, indent=""):
    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)
        print(indent + "|-- " + item)
        if os.path.isdir(item_path):
            print_directory_structure(item_path, indent + "    ")

# Usage
folder_path = r"E:\legeslative bot"  # replace this with your folder path
print_directory_structure(folder_path)
