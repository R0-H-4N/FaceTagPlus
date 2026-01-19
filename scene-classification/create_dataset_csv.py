import os
import csv


def create_dataset_csv(data_folder='data', output_csv='dataset.csv'):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, data_folder)
    # Fixed: added output_csv filename
    output_path = os.path.join(script_dir, output_csv)

    if not os.path.exists(data_path):
        print(f"Error: Data folder '{data_path}' does not exist.")
        return

    image_data = []

    for label_folder in os.listdir(data_path):
        label_path = os.path.join(data_path, label_folder)

        if not os.path.isdir(label_path):
            continue

        for image_file in os.listdir(label_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):

                image_data.append({
                    'image_name': image_file,
                    'label': label_folder
                })

    # Write to CSV
    if image_data:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image_name', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(image_data)

        print(f"✓ Total images: {len(image_data)}")
        print(
            f"✓ Total labels: {len(set(item['label'] for item in image_data))}")

        label_counts = {}
        for item in image_data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1

        print("\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count} images")
    else:
        print("No images found in the data folder structure.")


if __name__ == "__main__":
    create_dataset_csv()
