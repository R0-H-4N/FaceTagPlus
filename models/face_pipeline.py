from insightface.app import FaceAnalysis
import cv2
import numpy as np

class FacePipeline:
    def __init__(self, match_threshold=0.3):
        self.threshold = match_threshold
        self.reference_embeddings_mapping = None
        self.reference_embeddings = None
        self.app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def set_reference_embeddings(self, reference_embeddings):
        self.reference_embeddings = []
        self.reference_embeddings_mapping = []
        for i, (name, embedding) in enumerate(reference_embeddings.items()):
            self.reference_embeddings_mapping.append(name)
            self.reference_embeddings.append(embedding)
        self.reference_embeddings = self.normalize(np.stack(self.reference_embeddings))


    def get_faces(self, img):
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        faces = self.app.get(img_rgb)
        return faces
        
    def normalize(self, A):
        # Handle both 1D and 2D arrays
        if A.ndim == 1:
            A = A.reshape(1, -1)
        mag = np.linalg.norm(A, axis=1, keepdims=True)
        return A / mag
    
    def get_boxes(self, img):
        faces = self.get_faces(img)
        boxes = np.array([face.bbox for face in faces])
        return boxes

    def get_embeddings(self, img):
        faces = self.get_faces(img)
        if len(faces) == 0:
            return np.array([])
        embeddings = np.array([face.embedding for face in faces])
        # Ensure embeddings is always 2D
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings
    
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def get_labels(self, img):
        embeddings_raw = self.get_embeddings(img)
        
        # Handle case when no faces are detected
        if len(embeddings_raw) == 0:
            return [], 0
        
        embeddings = self.normalize(embeddings_raw)
        embeddings_matrix = (embeddings @ self.reference_embeddings.T)
        print(embeddings_matrix)
        best_matches_idx = np.argmax(embeddings_matrix, axis=1).reshape(-1).astype(int)
        labels = []
        duplicate = {}
        i = 0
        unknown_faces = 0
        for each in best_matches_idx:
            if embeddings_matrix[i][each] >= self.threshold:
                if not duplicate.get(each, False):
                    labels.append(self.reference_embeddings_mapping[each])
                duplicate[each] = True
            else:
                unknown_faces += 1
            i += 1
        return labels, unknown_faces