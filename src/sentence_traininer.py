import torch
import torch.nn.functional as F
from typing import Optional, Union, Any, Tuple
from transformers import Trainer
from transformers import AutoModel


def extract_sentence_embedding_from_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Extract the sentence embedding from the hidden states.
    """
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = torch.sum(mask_expanded, dim=1)
    return sum_embeddings / sum_mask.clamp(min=1e-9)

# Compute metrics function for evaluation
def compute_metrics(eval_pred: Tuple[torch.Tensor, torch.Tensor]) -> dict:
    """
    Compute Mean Squared Error (MSE) for evaluating model performance using PyTorch.
    Measures how close predicted similarity scores are to true labels.
    """
    predictions, labels = eval_pred
    
    # Convert numpy arrays to PyTorch tensors for consistent computation
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    # Calculate MSE using PyTorch
    mse = F.mse_loss(predictions_tensor, labels_tensor, reduction='mean')
    
    # Calculate RMSE (Root Mean Squared Error) for interpretability
    rmse = torch.sqrt(mse)
    
    # Calculate MAE (Mean Absolute Error) as an additional metric
    mae = F.l1_loss(predictions_tensor, labels_tensor, reduction='mean')
    
    # Calculate R² score (coefficient of determination)
    # R² = 1 - (SS_res / SS_tot)
    ss_res = torch.sum((labels_tensor - predictions_tensor) ** 2)
    ss_tot = torch.sum((labels_tensor - torch.mean(labels_tensor)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else torch.tensor(0.0)
    
    return {
        "mse": mse.item(),
        "rmse": rmse.item(),
        "mae": mae.item(),
        "r2_score": r2_score.item(),
    }

# Custom trainer class that handles sentence pair training
class SentencePairTrainer(Trainer):

    def compute_loss(self, model: AutoModel, inputs: dict[str, Union[torch.Tensor, Any]], return_outputs: bool = False, num_items_in_batch: Optional[torch.Tensor] = None):
        """Custom loss computation for sentence pairs"""

        input_ids_1 = inputs.get("input_ids_1")
        attention_mask_1 = inputs.get("attention_mask_1")
        input_ids_2 = inputs.get("input_ids_2")
        attention_mask_2 = inputs.get("attention_mask_2")
        labels = inputs.get("labels")

        try:
            # Get embeddings for sentence 1
            outputs1 = model(input_ids=input_ids_1, attention_mask=attention_mask_1)
            hidden_states1 = outputs1.last_hidden_state
            embeddings1 = extract_sentence_embedding_from_hidden_states(hidden_states1, attention_mask_1)
        except Exception as e:
            print(f"Some error happened for sentence 1 {input_ids_1} {attention_mask_1}")
            raise e
        
        try:
            # Get embeddings for sentence 2
            outputs2 = model(input_ids=input_ids_2, attention_mask=attention_mask_2)
            hidden_states2 = outputs2.last_hidden_state
            embeddings2 = extract_sentence_embedding_from_hidden_states(hidden_states2, attention_mask_2)
        except Exception as e:
            print(f"Some error happened for sentence 2 {input_ids_2} {attention_mask_2}")
            raise e
        
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(embeddings1, embeddings2)
        
        # Scale similarity to [0, 1] range
        cos_sim_scaled = (cos_sim + 1) / 2
        
        # Ensure tensors are properly shaped for loss computation
        cos_sim_scaled = cos_sim_scaled.squeeze()
        labels_float = labels.float().squeeze()
        
        # Binary cross entropy loss
        loss = F.mse_loss(cos_sim_scaled, labels_float, reduction='mean')
        
        return (loss, {"cos_sim": cos_sim_scaled}) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Custom prediction step to return cosine similarity scores for metrics computation.
        """
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            # Compute loss and get outputs
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            cos_sim_scaled = outputs["cos_sim"]
            
        if prediction_loss_only:
            return (loss, None, None)
        
        # Return predictions (cosine similarities) and labels
        labels = inputs.get("labels")
        return (loss, cos_sim_scaled, labels)
