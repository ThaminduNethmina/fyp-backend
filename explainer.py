# explainer.py
import shap
import torch
import torch.nn.functional as F

def generate_shap_explanation(
    cleaned_code: str, 
    model: torch.nn.Module, 
    tokenizer, 
    static_features_tensor: torch.Tensor, 
    device: torch.device, 
    pred_idx: int, 
    label_map: dict
):
    """
    Generates SHAP token importance scores for the predicted complexity class.
    """
    
    # SHAP Prediction Wrapper
    def text_prediction_wrapper(texts):
        texts_list = [str(t) for t in texts]
        encodings = tokenizer(
            texts_list, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(device)
        
        # Expand static features to match SHAP permutation batch size
        batch_size = encodings['input_ids'].shape[0]
        expanded_static = static_features_tensor.repeat(batch_size, 1)
        
        with torch.no_grad():
            batch_logits = model(
                input_ids=encodings['input_ids'], 
                attention_mask=encodings['attention_mask'], 
                static_features=expanded_static
            )
        return F.softmax(batch_logits, dim=1).cpu().numpy()

    # Configure SHAP Explainer
    masker = shap.maskers.Text(tokenizer, mask_token=tokenizer.mask_token)
    explainer = shap.Explainer(
        text_prediction_wrapper, 
        masker, 
        output_names=list(label_map.values())
    )
    
    # Calculate Values (max_evals=100 for API speed)
    shap_values = explainer([cleaned_code], max_evals=100)
    
    # Extract the specific tokens and their impact scores for the predicted class
    tokens = shap_values.data[0]
    scores = shap_values.values[0, :, pred_idx]
    
    # Map Character Offsets for Frontend Highlighting
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
        
    return token_data