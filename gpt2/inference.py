# import os
# import torch


# def inference():
#     # load pretrained model weights(checkpoints)
#     model_weights_path = os.path.join("log", "gpt2_v1.pt")
#     checkpoint = torch.load(model_weights_path, map_location="cpu")
#     csd = checkpoint["model"]
#     model = GPT(GPTConfig(vocab_size=50304)).to(device)
#     model.load_state_dict(csd, strict=False)


# if __name__ == "__main__":
#     inference()
