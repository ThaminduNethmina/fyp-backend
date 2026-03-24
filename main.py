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

class ComplexityFusionModel(nn.Module):
    def __init__(self, model_name, num_labels, num_static_features):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, static_features=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  
        static_vec = self.static_mlp(static_features)
        
        # Scaling matches your training setup
        fused = torch.cat((cls_embedding * 0.5, static_vec * 2.0), dim=1)
        logits = self.classifier(fused)
        return logits

def get_python_features(code):
    try:
        tree = ast.parse(code)
    except:
        return [0, 0, 0, 0, 0]

    max_depth = 0
    branch_count = 0
    has_recursion = 0
    has_log_math = 0
    has_sort = 0

    function_names = []

    class DepthVisitor(ast.NodeVisitor):
        def __init__(self):
            self.current = 0
            self.max_depth = 0

        def visit_For(self, node):
            self.current += 1
            self.max_depth = max(self.max_depth, self.current)
            self.generic_visit(node)
            self.current -= 1

        def visit_While(self, node):
            self.current += 1
            self.max_depth = max(self.max_depth, self.current)
            self.generic_visit(node)
            self.current -= 1

        def visit_ListComp(self, node):
            self.current += len(node.generators)
            self.max_depth = max(self.max_depth, self.current)
            self.generic_visit(node)
            self.current -= len(node.generators)

    dv = DepthVisitor()
    dv.visit(tree)
    max_depth = dv.max_depth

    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.AsyncFor)):
            branch_count += 1

        if isinstance(node, ast.FunctionDef):
            function_names.append(node.name)

        if isinstance(node, ast.Call):
            # recursion detection
            if isinstance(node.func, ast.Name) and node.func.id in function_names:
                has_recursion = 1

            if isinstance(node.func, ast.Attribute):
                if node.func.attr in function_names:
                    has_recursion = 1

            # sort detection
            if isinstance(node.func, ast.Name) and node.func.id == "sorted":
                has_sort = 1

            if isinstance(node.func, ast.Attribute) and node.func.attr == "sort":
                has_sort = 1

        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.RShift, ast.LShift)):
                has_log_math = 1

    return [max_depth, branch_count, has_recursion, has_log_math, has_sort]

def get_java_features(code):
    try:
        if "class " not in code:
            code = "class Dummy { " + code + " }"

        tokens = javalang.tokenizer.tokenize(code)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
    except:
        return [0, 0, 0, 0, 0]

    max_depth = 0
    branch_count = 0
    has_recursion = 0
    has_log_math = 0
    has_sort = 0

    methods = [node.name for _, node in tree.filter(javalang.tree.MethodDeclaration)]

    for path, node in tree.filter(javalang.tree.ForStatement):
        depth = sum(
            isinstance(p, (javalang.tree.ForStatement,
                           javalang.tree.WhileStatement,
                           javalang.tree.DoStatement))
            for p in path
        )
        max_depth = max(max_depth, depth + 1)

    for _, node in tree.filter(javalang.tree.IfStatement):
        branch_count += 1

    for _, node in tree.filter(javalang.tree.MethodInvocation):
        if node.member in methods:
            has_recursion = 1

        if node.member in ["sort", "parallelSort"]:
            has_sort = 1

    for _, node in tree.filter(javalang.tree.BinaryOperation):
        if node.operator in ['/', '>>', '<<', '>>>']:
            has_log_math = 1

    return [max_depth, branch_count, has_recursion, has_log_math, has_sort]

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

device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
label_map = {0: 'CONSTANT', 1: 'LINEAR', 2: 'LOGN', 3: 'NLOGN', 4: 'QUADRATIC', 5: 'CUBIC', 6: 'NP'}

print("Downloading model weights...")
model_path = hf_hub_download(repo_id="himansha2001/algox", filename="model.pth")

print("Loading model...")

model = ComplexityFusionModel("microsoft/unixcoder-base", 7, 5)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()
print("Model loaded successfully!")

class CodeRequest(BaseModel):
    code: str
    language: str = "java"

@app.post("/predict")
async def predict_complexity(request: CodeRequest):
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
        logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], static_features=request_static_features)
        probs = torch.nn.functional.softmax(logits, dim=1)
        
    pred_idx = probs.argmax().item()
    confidence = probs.max().item()
    prediction = label_map[pred_idx]
    
    # SHAP Wrapper
    def text_prediction_wrapper(texts):
        encodings = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        batch_size = encodings['input_ids'].shape[0]
        expanded_static = request_static_features.repeat(batch_size, 1)
        with torch.no_grad():
            batch_logits = model(input_ids=encodings['input_ids'], attention_mask=encodings['attention_mask'], static_features=expanded_static)
        return torch.nn.functional.softmax(batch_logits, dim=1).cpu().numpy()

    # Explain (SHAP)
    masker = shap.maskers.Text(tokenizer, mask_token="<mask>")
    explainer = shap.Explainer(text_prediction_wrapper, masker, output_names=list(label_map.values()))
    
    shap_values = explainer([cleaned_code], max_evals=100)
    
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
            "depth": feats[0], "branches": feats[1], "recursion": feats[2], "log_hint": feats[3], "has_sort": feats[4]
        },
        "shap_explanation": token_data
    }