import clip
import torch

def get_language_clip(model, Text, device):
    text = clip.tokenize([Text]).to(device)
    with torch.no_grad():
        text_feature = model.encode_text(text)
    return text_feature

if __name__ == '__main__': 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, process = clip.load("ViT-B/32", device=device)
    game = 'AirRaid'
    result = get_language_clip(model, game, device)
    print(result.shape)



