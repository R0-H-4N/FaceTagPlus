import os
import open_clip
from PIL import Image
import torch
import csv


class SceneRecognition:
    def __init__(self, scenes_csv_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model_name = 'ViT-B-32'  # ViT-L-14, ViT-B-16, etc.
        # options: "openai", "laion400m_e32", "laion2b_s34b_b79k", etc.
        pretrained_dataset = 'laion400m_e32'

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained_dataset
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device)

        # Load scene prompts from CSV
        if scenes_csv_path is None:
            scenes_csv_path = os.path.join(
                os.path.dirname(__file__), 'scenes.csv')

        self.scene_prompts = self.load_scenes_from_csv(scenes_csv_path)

        self.tokenized_text = self.tokenizer(self.scene_prompts)

    def load_scenes_from_csv(self, csv_path):
        """Load scene categories from CSV file"""
        scenes = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Replace underscores with spaces for better readability
                scene_name = row['scene_name'].replace('_', ' ')
                scenes.append(scene_name)
        return scenes

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        preprocessed_image = self.preprocess(
            image).unsqueeze(0).to(self.device)

        return preprocessed_image

    # def tokenize_text(self):
    #     return self.tokenizer(self.scene_prompts)

    def classify_scene(self, image):
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            if not hasattr(self, 'text_features'):
                self.text_features = self.model.encode_text(
                    self.tokenized_text)
                self.text_features = self.text_features / \
                    self.text_features.norm(dim=-1, keepdim=True)
            # Normalize

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Similarities
            similarity = (100 * image_features @ self.text_features.T)
            probs = similarity.softmax(dim=-1).cpu().numpy()

        return probs

    def load_images_batch(self, batch_size=32):
        image_folder = os.path.join(os.getcwd(), "test_images")
        image_paths = [f for f in os.listdir(
            image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []

            for img_path in batch_paths:
                full_path = os.path.join(image_folder, img_path)
                image = Image.open(full_path)
                preprocessed = self.preprocess(image)
                batch_images.append(preprocessed)

            batch_tensor = torch.stack(batch_images).to(self.device)
            yield batch_tensor, batch_paths

    def process_images_batch(self):
        """Process all images efficiently in batches"""
        results = []

        for batch_images, batch_paths in self.load_images_batch():
            probs = self.classify_scene(batch_images)

            for i, path in enumerate(batch_paths):
                results.append({
                    'image': path,
                    'confidence': f'{probs[i].max() * 100:.2f}%',
                    'top_scene': self.get_top_prediction(probs[i])
                })

        return results

    def get_top_prediction(self, probs):
        top_idx = probs.argmax()
        return self.scene_prompts[top_idx]


if __name__ == "__main__":
    scene_recognizer = SceneRecognition()
    results = scene_recognizer.process_images_batch()

    for result in results:
        print(f"Image: {result['image']}")
        print(f"Top Scene: {result['top_scene']}")
        print(f"Confidence: {result['confidence']}\n")
