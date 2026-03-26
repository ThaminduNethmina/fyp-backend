import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from model import ComplexityFusionModel
from features import clean_code, get_python_features, get_java_features
from explainer import generate_shap_explanation

# API SETUP
app = FastAPI(title="Code Complexity XAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map = {0: 'CONSTANT', 1: 'LINEAR', 2: 'LOGN', 3: 'NLOGN', 4: 'QUADRATIC', 5: 'CUBIC', 6: 'NP'}
REPO_ID = "himansha2001/algox" 

print("Booting up backend services...")
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
model = ComplexityFusionModel(model_name="microsoft/unixcoder-base", num_labels=7, num_static_features=5)

safetensors_path = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors")
state_dict = load_file(safetensors_path)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()
print("API is ready for inference!")

class CodeRequest(BaseModel):
    code: str
    language: str

@app.get("/")
async def health_check():
    """
    Root endpoint to verify the API is online and the model is loaded.
    """
    return {
        "status": "online",
        "message": "Code Complexity XAI API is running successfully.",
        "model_loaded": True,
        "version": "1.0.0"
    }

@app.post("/predict")
async def predict_complexity(request: CodeRequest):
    """
    Endpoint to predict the complexity of the provided code and generate an explanation.
    """

    lang = request.language.lower()
    
    # Prepare Data
    cleaned_code = clean_code(request.code, lang)
    if lang == 'python':
        feats = get_python_features(request.code)
    elif lang == 'java':
        feats = get_java_features(request.code)
    else:
        raise HTTPException(status_code=400, detail="Language must be 'java' or 'python'")
        
    request_static_features = torch.tensor([feats], dtype=torch.float32).to(device)
    
    # Tokenize & Predict
    inputs = tokenizer(cleaned_code, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            static_features=request_static_features
        )
        probs = F.softmax(logits, dim=1)
        
    pred_idx = probs.argmax().item()
    confidence = probs.max().item()
    prediction = label_map[pred_idx]
    
    # Generate SHAP Explanation
    shap_explanation = generate_shap_explanation(
        cleaned_code=cleaned_code,
        model=model,
        tokenizer=tokenizer,
        static_features_tensor=request_static_features,
        device=device,
        pred_idx=pred_idx,
        label_map=label_map
    )
        
    # Return Response
    return {
        "complexity": prediction,
        "confidence": float(confidence),
        "static_features": {
            "max_depth": feats[0], 
            "branch_count": feats[1], 
            "has_recursion": bool(feats[2]), 
            "has_log_math": bool(feats[3]), 
            "has_sort": bool(feats[4])
        },
        "shap_explanation": shap_explanation
    }