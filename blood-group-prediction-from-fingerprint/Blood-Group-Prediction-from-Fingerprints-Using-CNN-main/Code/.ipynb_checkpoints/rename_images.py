from PIL import Image
import os

data_dir = r"C:\Users\Bharath R\Downloads\archive\dataset_blood_group"  # Replace with YOUR path

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        for file_name in os.listdir(class_path):
            if file_name.lower().endswith(".bmp"):
                bmp_path = os.path.join(class_path, file_name)
                img = Image.open(bmp_path)
                new_file_name = os.path.splitext(file_name)[0] + ".png"  # Convert to PNG
                new_path = os.path.join(class_path, new_file_name)
                img.save(new_path)
                os.remove(bmp_path) # Remove the old BMP file
                print(f"Converted {bmp_path} to {new_path}")

print("Conversion complete!")