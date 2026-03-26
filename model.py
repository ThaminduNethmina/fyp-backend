import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

class ComplexityFusionModel(nn.Module):
    def __init__(self, model_name, num_labels, num_static_features, static_hidden_dim=16):
        super(ComplexityFusionModel, self).__init__()
        
        # Load config and base model
        self.config = AutoConfig.from_pretrained(model_name)
        self.codebert = AutoModel.from_pretrained(model_name)

        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, static_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        fusion_dim = self.config.hidden_size + static_hidden_dim
        self.classifier = nn.Linear(fusion_dim, num_labels)

    def forward(self, input_ids=None, attention_mask=None, static_features=None):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = outputs.last_hidden_state[:, 0, :]

        static_output = self.static_mlp(static_features)

        combined_features = torch.cat((bert_output, static_output), dim=1)

        logits = self.classifier(combined_features)

        return logits