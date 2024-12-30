
from huggingface_hub import login
from dynamic_predictor.dust3r.model import AsymmetricCroCo3DStereo

login(token="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")


model = AsymmetricCroCo3DStereo.from_pretrained("das3r_checkpoint-last.pth")
model_name = "das3r_checkpoint-last.pth"
model.push_to_hub(model_name)