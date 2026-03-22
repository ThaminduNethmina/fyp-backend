import token

import torch
import torch.nn as nn
import ast
import re
import javalang
import shap
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download

# MODEL ARCHITECTURE 
class ComplexityFusionModel(nn.Module):
    def __init__(self, model_name, num_labels, num_static_features):
        super(ComplexityFusionModel, self).__init__()
        self.codebert = AutoModel.from_pretrained(model_name)
        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(768 + 16, num_labels)
        
    def forward(self, input_ids, attention_mask, static_features):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = outputs.last_hidden_state[:, 0, :]
        static_output = self.static_mlp(static_features)
        combined = torch.cat((bert_output, static_output), dim=1)
        logits = self.classifier(combined)
        return logits

# FEATURE EXTRACTORS 
def get_python_features(code):
    try:
        tree = ast.parse(code)
    except:
        return [0, 0, 0, 0]

    max_depth = 0
    branch_count = 0
    has_recursion = 0
    has_log_math = 0
    
    class DepthVisitor(ast.NodeVisitor):
        def __init__(self):
            self.max_depth = 0
            self.current_depth = 0
        def visit_For(self, node):
            self.current_depth += 1
            self.max_depth = max(self.max_depth, self.current_depth)
            self.generic_visit(node)
            self.current_depth -= 1
        def visit_While(self, node):
            self.current_depth += 1
            self.max_depth = max(self.max_depth, self.current_depth)
            self.generic_visit(node)
            self.current_depth -= 1
            
    try:
        v = DepthVisitor()
        v.visit(tree)
        max_depth = v.max_depth
    except: pass

    current_functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            branch_count += 1
        if isinstance(node, ast.FunctionDef):
            current_functions.append(node.name)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in current_functions:
                has_recursion = 1
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.RShift, ast.Mult, ast.LShift)):
                has_log_math = 1
        if isinstance(node, ast.AugAssign):
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.RShift, ast.Mult, ast.LShift)):
                has_log_math = 1
                
    return [max_depth, branch_count, has_recursion, has_log_math]

def get_java_features(code):
    has_log_math = 0
    if re.search(r'[\w\]]+\s*[/|*|%]=\s*', code): has_log_math = 1
    if re.search(r'>>|<<|>>>', code): has_log_math = 1
    if re.search(r'\(\s*[\w\s\+\-]+\s*\)\s*/\s*2', code): has_log_math = 1

    try:
        if "class " not in code:
             tokens = javalang.tokenizer.tokenize("class Dummy { " + code + " }")
        else:
             tokens = javalang.tokenizer.tokenize(code)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
    except:
        return [0, 0, 0, has_log_math]

    real_max_depth = 0
    for path, node in tree.filter(javalang.tree.ForStatement):
        real_max_depth = max(real_max_depth, 1)
        
    branch_count = 0
    for path, node in tree.filter(javalang.tree.IfStatement):
        branch_count += 1
        
    has_recursion = 0
    methods = [node.name for path, node in tree.filter(javalang.tree.MethodDeclaration)]
    for path, node in tree.filter(javalang.tree.MethodInvocation):
        if node.member in methods:
            has_recursion = 1
            
    return [real_max_depth, branch_count, has_recursion, has_log_math]

def clean_code(code, lang):
    code = re.sub(r'\n\s*\n', '\n', code)
    if lang == 'java':
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    return code.strip()

# API
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load Model
device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
label_map = {0: 'CONSTANT', 1: 'LINEAR', 2: 'LOGN', 3: 'NLOGN', 4: 'QUADRATIC', 5: 'CUBIC', 6: 'NP'}

print("Downloading model weights...")
model_path = hf_hub_download(repo_id="himansha2001/algox", filename="model.pth")

print("Loading model...")
model = ComplexityFusionModel("microsoft/unixcoder-base", 7, 4)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully!")

class CodeRequest(BaseModel):
    code: str
    language: str = "java"


@app.post("/predict")
async def predict_complexity(request: CodeRequest):
    
    # Prepare
    lang = request.language.lower()
    cleaned_code = clean_code(request.code, lang)
    if lang == 'python':
        feats = get_python_features(request.code)
    else:
        feats = get_java_features(request.code)
        
    request_static_features = torch.tensor([feats], dtype=torch.float).to(device)
    
    # Predict
    inputs = tokenizer(cleaned_code, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'], request_static_features)
        probs = torch.nn.functional.softmax(logits, dim=1)
        
    pred_idx = probs.argmax().item()
    confidence = probs.max().item()
    prediction = label_map[pred_idx]
    
    def get_predictor(static_feats):
        def predictor(texts):
            inputs = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True).to(device)
            batch_size = inputs['input_ids'].shape[0]
            expanded_static = static_feats.repeat(batch_size, 1)
            with torch.no_grad():
                logits = model(inputs['input_ids'], inputs['attention_mask'], expanded_static)
            return torch.nn.functional.softmax(logits, dim=1).detach().numpy()
        return predictor

    # Explain (SHAP)
    masker = shap.maskers.Text(tokenizer, mask_token="<mask>")
    explainer = shap.Explainer(get_predictor(request_static_features), masker, output_names=list(label_map.values()))
    
    # Run SHAP
    shap_values = explainer([cleaned_code], max_evals=10)
    
    tokens = shap_values.data[0]
    scores = shap_values.values[0, :, pred_idx]
    
    encoding = tokenizer(cleaned_code, return_offsets_mapping=True, truncation=True, max_length=512)
    offsets = encoding["offset_mapping"]
    
    token_data = []
    
    for i, (t, s) in enumerate(zip(tokens, scores)):
        start_char = offsets[i][0] if i < len(offsets) else 0
        end_char = offsets[i][1] if i < len(offsets) else 0
        
        token_data.append({
            "token": t, 
            "score": float(s),
            "start_char": start_char,
            "end_char": end_char
        })
        
    return {
        "complexity": prediction,
        "confidence": float(confidence),
        "static_features": {
            "depth": feats[0], "branches": feats[1], "recursion": feats[2], "log_hint": feats[3]
        },
        "shap_explanation": token_data
    }