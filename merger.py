import os

def merge_txt_files(input_folder, output_file='merged_output.txt'):
    if not os.path.exists(input_folder):
        print("Folder not found.")
        return

    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    
    if not txt_files:
        print("No .txt files found in the folder.")
        return

    txt_files.sort()  # Optional: Sort alphabetically

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for fname in txt_files:
            file_path = os.path.join(input_folder, fname)
            with open(file_path, 'r', encoding='utf-8') as infile:
                content = infile.read()
                outfile.write(f"{content}\n")
                outfile.write(f"{'_' * 70}\n")  # Border line

    print(f"{len(txt_files)} file(s) merged into {output_file}")

# Example usage
folder_path = r"E:\legeslative bot\DATA"
merge_txt_files(folder_path)
