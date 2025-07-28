import os

def rename_images(directory):
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            # Extract the numeric part from the filename (without .png)
            file_path = os.path.join(directory, filename)
            stem = filename[:-4]  # Remove .png extension
            
            try:
                # Convert stem to integer (handles leading zeros)
                number = int(stem)
                # Format new filename with leading zeros
                new_name = f"{number:07d}.png"
                new_path = os.path.join(directory, new_name)
                
                # Rename the file
                os.rename(file_path, new_path)
                print(f"Renamed: {filename} → {new_name}")
            
            except ValueError:
                print(f"Skipping non-numeric file: {filename}")

if __name__ == "__main__":
    target_dir = input("Enter directory path (or press Enter for current folder): ").strip()
    if not target_dir:
        target_dir = os.getcwd()
    
    rename_images(target_dir)
    print("Renaming completed.")