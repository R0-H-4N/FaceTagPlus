from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from face_pipeline import FacePipeline
from typing import List
import json
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PIL import Image
import io
from place import categories, img_transform

app = FastAPI()
pipeline = FacePipeline()
scene_recognizer = None
places_model = None    
# Load T5 model for natural language to MongoDB query conversion
import gdown

# Google Drive folder URL
folder_url = "https://drive.google.com/drive/folders/1o83NUytG1M-JG5SWHaWzJrGWCIlM6YYy?usp=drive_link"
folder_name = "mongo_query_t5"

# Check if folder already exists
if os.path.exists(folder_name):
    print(f"✅ Folder '{folder_name}' already exists. Skipping download.")
else:
    print(f"📥 Downloading '{folder_name}' from Google Drive...")
    gdown.download_folder(url=folder_url, output=folder_name, quiet=False, use_cookies=False, remaining_ok=True)
    print("✅ Done!")



MODEL_PATH = os.path.join(os.path.dirname(__file__), "mongo_query_t5")
try:
    print(f"Loading T5 model from {MODEL_PATH}...")
    t5_tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    t5_model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    print("✅ T5 model loaded successfully")
except Exception as e:
    print(f"⚠️ Failed to load T5 model: {e}")
    t5_tokenizer = None
    t5_model = None

### first send request to "/setmodel" to tell it what photo belongs to which person,
### by sending two separate lists of potraits and their names/unique_id in the same order
###
### only then it can accept photos at /getlabels to label them with people name


model_to_use_is_clip = False
model_to_use_is_resnet = True
@app.on_event("startup")

async def load_model():
    global scene_recognizer
    global places_model
    if model_to_use_is_clip:
        try:
            from scene_recognition import SceneRecognition
            scene_recognizer = SceneRecognition()
            print("✅ Scene Recognition model loaded successfully from CLIP")
        except Exception as e:
            print(f"⚠️ Failed to load Scene Recognition model: {e}")
            scene_recognizer = None
    if model_to_use_is_resnet:
        try:
            from place import Ensemble
            places_model = Ensemble(1, 0.7)
            print("✅ Scene Recognition model loaded successfully from Places365")
        except Exception as e:
            print(f"⚠️ Failed to load Scene Recognition model: {e}")
            places_model = None

@app.post("/getlabels")
async def get_labels(files: List[UploadFile] = File(...), reference_embeddings: str = Form(...)):
    """
    input:
    list of images to be processed

    example usage:
    curl.exe -X POST "http://localhost:8000/setmodel" `
    -F "files=@C:/Users/rohan/Desktop/datasets/50.jpg" 

    """
    

    results = []
    ref_emb = json.loads(reference_embeddings)
    pipeline.set_reference_embeddings(ref_emb)

    for file in files:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        labels, unknown_faces_count = pipeline.get_labels(img)
        results.append({
            "filename": file.filename,
            "labels": labels,
            "unknown_faces": unknown_faces_count
        })

    return {"results": results}

@app.post("/getembeddings")
async def upload_image(files: List[UploadFile] = File(...), labels: List[str] = Form(...)):
    """
    files = list of images of ;
    labels = list of names of people of images in string;

    they should be in order, labels[i]'s potrait should be files[i]

    Example request in curl:
    curl.exe -X POST "http://localhost:8000/setmodel" `
    -F "files=@C:/Users/rohan/Desktop/datasets/50.jpg" `
    -F "files=@C:/Users/rohan/Desktop/datasets/48.jpg" `
    -F "labels=Rohan" `
    -F "labels=Someone" 


    output, a json with keys:
    embeddings = embedding of the people mentions in labels (dict dype, embeddings[person_name] = embeddings_list)
    error = error message dictionary, error[person_name] = error encountered during getting embedding of that person name
    if no error: error will just be string

    """
    reference_embeddings = {}
    err_message = {}
    for file, label in zip(files, labels):
        content = await file.read()
        np_arr = np.frombuffer(content, np.uint8)
        potrait = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        embedding = pipeline.get_embeddings(potrait)
        if embedding.shape[0] != 1:
            err_message[label] = "Please send clear photo with only face of the subject in it"
        else:
            reference_embeddings[label] = [float(x) for x in embedding[0]]

    error_found = len(err_message.items()) > 0
    pipeline.set_reference_embeddings(reference_embeddings)
    for face, emb in reference_embeddings.items():
        print(face, ":", len(emb))
    return {
        'embeddings': reference_embeddings,
        'error': 'No error' if (not error_found) else err_message
    }

@app.post("/getquery")
async def get_query(natural_language: str = Form(...)):
    """
    Convert natural language to MongoDB query using fine-tuned T5 model.
    
    Input:
        natural_language: Natural language query (e.g., "I need photo of prabin")
    
    Output:
        {
            "query": "faceEmbedding.identifier:['prabin']",
            "parsed": {
                "faceEmbeddings.identifiedUser": "prabin"
            },
            "success": true
        }
    
    Example usage:
    curl -X POST "http://localhost:8000/getquery" \
         -F "natural_language=I need photo of prabin"
    """
    
    if t5_model is None or t5_tokenizer is None:
        return {
            "success": False,
            "error": "T5 model not loaded. Please check model path.",
            "query": None,
            "parsed": None
        }
    
    try:
        # Add the task prefix that the model was trained with
        input_text = "translate text to mongo query: " + natural_language
        
        # Tokenize the input
        inputs = t5_tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Generate the MongoDB query
        outputs = t5_model.generate(
            **inputs,
            max_length=128  # Use same max_length as in training
        )
        
        # Decode the output
        generated_query = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Natural Language: {natural_language}")
        print(f"Generated Query: {generated_query}")
        
        # Check for gibberish output
        if len(generated_query) > 500 or generated_query.count('face') > 10:
            print(f"⚠️ T5 model generated gibberish output!")
            print(f"This usually means the model doesn't recognize this username.")
            print(f"Consider retraining with more diverse examples.")
        
        # Try to parse the generated query directly as JSON/Python dict
        parsed_query = parse_t5_query(generated_query)
        
        return {
            "success": True,
            "natural_language": natural_language,
            "query": generated_query,
            "parsed": parsed_query
        }
        
    except Exception as e:
        print(f"Error generating query: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": None,
            "parsed": None
        }

@app.post("/classifyscene")
async def classify_scene(files: List[UploadFile] = File(...)):
    """
    Classify the scene in uploaded images.

    Input:
        files: List of image files to classify

    Output:
        {
            "results": [
                {
                    "filename": "image.jpg",
                    "top_scene": "beach",
                    "confidence": "95.32%"
                }
            ]
        }

    Example usage:
    curl -X POST "http://localhost:8000/classifyscene" \
         -F "files=@/path/to/image.jpg"
    """

    global scene_recognizer, places_model
    
    # Check if at least one model is loaded
    if scene_recognizer is None and places_model is None:
        return {
            "success": False,
            "error": "Scene Recognition model not loaded. Please install dependencies: torch, torchvision, open-clip-torch, Pillow",
            "results": []
        }

    try:
        import torch
        results = []

        for file in files:
            # Read image
            contents = await file.read()

            # Convert to PIL Image
            image = Image.open(io.BytesIO(contents)).convert('RGB')

            result_places = None
            result_clip = None
            
            # Try ResNet/Places365 model
            if model_to_use_is_resnet and places_model is not None:
                try:
                    # Transform image to tensor
                    image_tensor = img_transform(image).unsqueeze(0)
                    
                    # Move to same device as model
                    image_tensor = image_tensor.to(places_model.device)
                    
                    # Get predictions
                    with torch.no_grad():
                        probs, indices = places_model.predict(image_tensor)
                    
                    indices = indices[0]
                    probs = probs[0]
                    
                    # Process results
                    if len(indices) == 0:
                        top_scene = 'uncertain'
                        confidence = ''
                    elif len(indices) == 1:
                        top_scene = categories[indices[0].item()]
                        confidence = f'{probs[0].item() * 100:.2f}%'
                        
                        # Check confidence threshold
                        if probs[0].item() < 0.1:
                            top_scene = 'uncertain'
                            confidence = ''
                    else:
                        top_scene = [categories[i.item()] for i in indices]
                        confidence = [f'{p.item() * 100:.2f}%' for p in probs]

                    top_scene = top_scene.strip().replace(' ', '_')
                    
                    result_places = {
                        "filename": file.filename,
                        "top_scene": top_scene,
                        "confidence": confidence,
                    }
                    
                    print(f"Places365 result: {top_scene}, confidence: {confidence}")
                    
                except Exception as e:
                    print(f"Error with Places365 model: {e}")
                    result_places = None

            # Try CLIP model
            if model_to_use_is_clip and scene_recognizer is not None:
                try:
                    preprocessed_image = scene_recognizer.preprocess_image(image)
                    probs = scene_recognizer.classify_scene(preprocessed_image)

                    # Get top prediction
                    top_idx = probs[0].argmax()
                    top_scene = scene_recognizer.scene_prompts[top_idx]
                    confidence = f'{probs[0].max() * 100:.2f}%'

                    # Check confidence threshold
                    if probs[0].max() < 0.1:
                        top_scene = "uncertain"
                        confidence = ''

                    top_scene = top_scene.strip().replace(' ', '_')
                    result_clip = {
                        "filename": file.filename,
                        "top_scene": top_scene,
                        "confidence": confidence,
                    }
                    
                    print(f"CLIP result: {top_scene}, confidence: {confidence}")
                    
                except Exception as e:
                    print(f"Error with CLIP model: {e}")
                    result_clip = None

            # Decide which result to use
            final_result = None
            
            if result_places is not None and result_clip is not None:
                # Both models succeeded - prioritize based on confidence
                places_has_result = result_places['confidence'] != ''
                clip_has_result = result_clip['confidence'] != ''
                
                if places_has_result and not clip_has_result:
                    final_result = result_places
                elif clip_has_result and not places_has_result:
                    final_result = result_clip
                elif places_has_result and clip_has_result:
                    # Both have confident results - prefer Places365 by default
                    final_result = result_places
                else:
                    # Neither is confident
                    final_result = {
                        "filename": file.filename,
                        "top_scene": "uncertain",
                        "confidence": "",
                        "model": "none"
                    }
            elif result_places is not None:
                final_result = result_places
            elif result_clip is not None:
                final_result = result_clip
            else:
                # Both models failed
                final_result = {
                    "filename": file.filename,
                    "top_scene": "error",
                    "confidence": "",
                    "model": "none",
                    "error": "Both models failed to process image"
                }
            
            results.append(final_result)

        print(f"Final results: {results}")

        return {
            "success": True,
            "results": results
        }

    except Exception as e:
        print(f"Error classifying scene: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "results": []
        }
        
def remove_at_symbols(obj):
    """
    Recursively remove @ symbols from username values in a dict/list.
    Example: {"identifiedUser": "@prajwol"} -> {"identifiedUser": "prajwol"}
    """
    if isinstance(obj, dict):
        return {k: remove_at_symbols(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [remove_at_symbols(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith('@'):
        return obj[1:]  # Remove the @ symbol
    else:
        return obj


def parse_t5_query(generated_query: str):
    """
    Parse the T5 model output directly into a MongoDB query.
    The T5 model is trained to generate MongoDB-like queries.
    
    Examples:
    - '"faceEmbeddings.identifiedUser": "@prajwol"' 
    - '"faceEmbeddings.identifiedUser": "$in": ["@prajwol", "@prabin"]'
    - '"faceEmbeddings" "$size": 1, "$elemMatch" "identifiedUser": "@prajwol"' (for "only")
    - '"tags": "$in": ["nature", "sunset"]'
    - '"description": "$regex": "beach"'
    """
    import json
    import re
    
    try:
        cleaned_query = generated_query.strip()
        print(f"Parsing T5 query: {cleaned_query}")
        
       
        if len(cleaned_query) > 500 or cleaned_query.count('face') > 10:
            print(f"⚠️ Detected gibberish/repetitive output from T5 model")
            print(f"Query length: {len(cleaned_query)}, 'face' count: {cleaned_query.count('face')}")
            # Return empty dict - will show all photos in group
            return {}
       
        cleaned_query = cleaned_query.replace('"facefaceEmbeddings"', '"faceEmbeddings"')
        
     
        if not cleaned_query.startswith('{'):
            cleaned_query = '{' + cleaned_query + '}'
        
        # Fix common T5 output issues:
        # 1. Replace single quotes with double quotes for JSON compatibility
        cleaned_query = cleaned_query.replace("'", '"')
        
        # 2. Fix operators with arrays: "$in", "$nin", "$all"
        # Pattern: "field": "$operator": [values] -> "field": {"$operator": [values]}
        cleaned_query = re.sub(
            r'("faceEmbeddings\.identifiedUser"\s*:\s*)"(\$(?:in|nin))"\s*:\s*(\[[^\]]+\])',
            r'\1{"\2": \3}',
            cleaned_query
        )
        
        # 3. Fix $ne format: "field":"$ne":"value" -> "field":{"$ne":"value"}
        cleaned_query = re.sub(
            r'("faceEmbeddings\.identifiedUser"\s*:\s*)"(\$ne)"\s*:\s*("[^"]+")',
            r'\1{"\2": \3}',
            cleaned_query
        )
        
        # 4. Fix $and format with proper handling of nested conditions
        # Find $and and properly extract its array content considering nested brackets
        and_start = cleaned_query.find('"$and"')
        
        if and_start != -1:
            # Find the opening bracket after "$and":
            bracket_start = cleaned_query.find('[', and_start)
            
            if bracket_start != -1:
                # Now find the matching closing bracket
                bracket_depth = 0
                bracket_end = -1
                
                for i in range(bracket_start, len(cleaned_query)):
                    if cleaned_query[i] == '[':
                        bracket_depth += 1
                    elif cleaned_query[i] == ']':
                        bracket_depth -= 1
                        if bracket_depth == 0:
                            bracket_end = i
                            break
                
                if bracket_end != -1:
                    and_content = cleaned_query[bracket_start + 1:bracket_end]
                    
                    fixed_conditions = []
                    # Split by comma but respect nested braces and brackets
                    depth = 0
                    bracket_depth = 0
                    current_condition = ""
                    
                    for char in and_content:
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                        elif char == '[':
                            bracket_depth += 1
                        elif char == ']':
                            bracket_depth -= 1
                        elif char == ',' and depth == 0 and bracket_depth == 0:
                            if current_condition.strip():
                                cond = current_condition.strip()
                                if not cond.startswith('{'):
                                    cond = '{' + cond + '}'
                                fixed_conditions.append(cond)
                            current_condition = ""
                            continue
                        current_condition += char
                    
                    # Don't forget the last condition
                    if current_condition.strip():
                        cond = current_condition.strip()
                        if not cond.startswith('{'):
                            cond = '{' + cond + '}'
                        fixed_conditions.append(cond)
                    
                    # Reconstruct the $and part
                    and_array = ', '.join(fixed_conditions)
                    
                    # Check if there are conditions after $and (outside the array)
                    after_and = cleaned_query[bracket_end + 1:]
                    if after_and.strip().startswith(','):
                        # There are more conditions after $and, merge them into $and
                        remaining = after_and.strip()[1:].strip()
                        if remaining.endswith('}'):
                            remaining = remaining[:-1].strip()
                        
                        if remaining:
                            if not remaining.startswith('{'):
                                remaining = '{' + remaining + '}'
                            fixed_conditions.append(remaining)
                            and_array = ', '.join(fixed_conditions)
                    
                    cleaned_query = f'{{"$and": [{and_array}]}}'
     
        # 5. Fix $size + $elemMatch pattern for "only" queries
        size_elemMatch_pattern = r'("faceEmbeddings")\s*:\s*("\$size")\s*:\s*(\d+)\s*,\s*("\$elemMatch")\s*:\s*("identifiedUser"\s*:\s*"[^"]+")'
        match = re.search(size_elemMatch_pattern, cleaned_query)
        
        if match:
            field = match.group(1)
            size_value = match.group(3)
            elemMatch_content = match.group(5)
            cleaned_query = f'{{{field}: {{"$size": {size_value}, "$elemMatch": {{{elemMatch_content}}}}}}}'
        
        # 6. Fix $size + $all pattern for "only X and Y" queries
        # Pattern: "faceEmbeddings": "$size": 2, "$all": ["$elemMatch": "identifiedUser": "user", ...]
        size_all_start = cleaned_query.find('"faceEmbeddings"')
        
        if size_all_start != -1:
            # Check if this is a $size + $all pattern
            size_match = re.search(r'"faceEmbeddings"\s*:\s*"\$size"\s*:\s*(\d+)', cleaned_query[size_all_start:])
            if size_match:
                size_value = size_match.group(1)
                
                # Find the $all array
                all_start = cleaned_query.find('"$all"', size_all_start)
                if all_start != -1:
                    bracket_start = cleaned_query.find('[', all_start)
                    
                    if bracket_start != -1:
                        # Find matching closing bracket
                        bracket_depth = 0
                        bracket_end = -1
                        
                        for i in range(bracket_start, len(cleaned_query)):
                            if cleaned_query[i] == '[':
                                bracket_depth += 1
                            elif cleaned_query[i] == ']':
                                bracket_depth -= 1
                                if bracket_depth == 0:
                                    bracket_end = i
                                    break
                        
                        if bracket_end != -1:
                            all_content = cleaned_query[bracket_start + 1:bracket_end]
                            
                            # Parse $all array items - each should be wrapped as {"$elemMatch": {"identifiedUser": "user"}}
                            all_items = []
                            
                            # Split by "$elemMatch" occurrences
                            parts = all_content.split('"$elemMatch"')
                            
                            for i, part in enumerate(parts):
                                if i == 0 and not part.strip():
                                    continue  # Skip empty first part
                                
                                # Extract the identifiedUser value
                                # Pattern: : "identifiedUser": "value"
                                user_match = re.search(r':\s*"identifiedUser"\s*:\s*"([^"]+)"', part)
                                if user_match:
                                    username = user_match.group(1)
                                    all_items.append(f'{{"$elemMatch": {{"identifiedUser": "{username}"}}}}')
                            
                            if all_items:
                                all_array = ', '.join(all_items)
                                # Check if there are other fields after the $all pattern
                                after_all = cleaned_query[bracket_end + 1:]
                                if after_all.strip().startswith(','):
                                    # Extract additional fields
                                    additional = after_all.strip()[1:].strip()
                                    if additional.endswith('}'):
                                        additional = additional[:-1].strip()
                                    if additional:
                                        cleaned_query = f'{{"faceEmbeddings": {{"$size": {size_value}, "$all": [{all_array}]}}, {additional}}}'
                                    else:
                                        cleaned_query = f'{{"faceEmbeddings": {{"$size": {size_value}, "$all": [{all_array}]}}}}'
                                else:
                                    cleaned_query = f'{{"faceEmbeddings": {{"$size": {size_value}, "$all": [{all_array}]}}}}'
                else:
                    # Just $size without $all (e.g., photos with no faces: $size: 0)
                    # Check if there's no $all or $elemMatch after $size
                    remaining_after_size = cleaned_query[size_all_start + size_match.end():]
                    if '"$all"' not in remaining_after_size and '"$elemMatch"' not in remaining_after_size:
                        # Check for additional fields after $size
                        # Look for comma after the size value
                        size_end_in_full = size_all_start + size_match.end()
                        after_size = cleaned_query[size_end_in_full:]
                        
                        if after_size.strip().startswith(','):
                            # There are additional fields
                            additional = after_size.strip()[1:].strip()
                            if additional.endswith('}'):
                                additional = additional[:-1].strip()
                            if additional:
                                cleaned_query = f'{{"faceEmbeddings": {{"$size": {size_value}}}, {additional}}}'
                            else:
                                cleaned_query = f'{{"faceEmbeddings": {{"$size": {size_value}}}}}'
                        else:
                            cleaned_query = f'{{"faceEmbeddings": {{"$size": {size_value}}}}}'
        
        # 7. Fix remaining operators with arrays: "$in", "$nin" (general case)
        cleaned_query = re.sub(
            r'("[^"]+"\s*:\s*)"(\$(?:in|nin))"\s*:\s*(\[[^\]]+\])',
            r'\1{"\2": \3}',
            cleaned_query
        )
        
        # 8. Fix $regex format: "field": "$regex": "value" -> "field": {"$regex": "value"}
        cleaned_query = re.sub(
            r'("[^"]+"\s*:\s*)"(\$regex)"\s*:\s*("[^"]+")',
            r'\1{"\2": \3}',
            cleaned_query
        )
        
        if '"$sort"' in cleaned_query or '"$limit"' in cleaned_query:
            # Extract only the filter fields, ignore sort/limit/etc
            filter_match = re.search(r'\{([^}]*?)(?:,\s*"\$(?:sort|limit|skip)".*?)?\}', cleaned_query, re.DOTALL)
            if filter_match:
                cleaned_query = '{' + filter_match.group(1) + '}'
        
        print(f"Cleaned query: {cleaned_query}")
        
        # Try to parse as JSON
        try:
            parsed = json.loads(cleaned_query)
            
           
            parsed = remove_at_symbols(parsed)
            
            
            if 'faceEmbeddings' in parsed and isinstance(parsed['faceEmbeddings'], dict):
                if '$elemMatch' in parsed['faceEmbeddings']:
                    parsed['unknownFacesCount'] = 0
                    print(f"Added unknownFacesCount: 0 for 'only' query")
            
            print(f"Successfully parsed: {parsed}")
            return parsed
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Attempted to parse: {cleaned_query}")
            
            # Fallback: return empty dict (shows all photos in group)
            return {}
        
    except Exception as e:
        print(f"Error in parse_t5_query: {e}")
        return {}