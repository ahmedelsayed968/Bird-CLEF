from transformers import AutoImageProcessor, AutoModel
import torch
from peft import LoraConfig, get_peft_model
from copy import deepcopy
from torch import nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self,transformer,processor,n_out,device):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = transformer
#         for params in self.transformer.parameters():
#             params.requires_grad = False
            
        self.processor = processor
        self.n_out = n_out
        self.device = device
        self.classifier = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, self.n_out))
        self.softmax    =  nn.Softmax(dim=1)
    def forward(self, x):
        x = self.processor(x,return_tensors="pt",do_rescale=False)
#         print(x)
        x = x.to(device)
#         print(x.pixel_values)
        x = self.transformer(**x)
        x =x.pooler_output
        x = nn.Flatten()(x)
        logits = self.classifier(x)
#         print(logits.shape)

        predications = self.softmax(logits)
#         print(predications.shape)
        return predications

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoImageProcessor.from_pretrained('/kaggle/input/dinov2/pytorch/large/1/',)
    model = AutoModel.from_pretrained('/kaggle/input/dinov2/pytorch/large/1/').to(device)
    target_modules = ['query',
                 'key',
                 'value'
                 ]
    for params in model.parameters():
        params.requires_grad = False # Freeze all parameter
        if params.ndim == 1:
            params.data = params.data.to(torch.float32) # cast to float32 for stability

    # Enables the gradients for the input embeddings.
    # This is useful for fine-tuning adapter weights while keeping the model weights fixed.
    model.enable_input_require_grads()
    # reduce number of stored activations
    model.gradient_checkpointing_enable()
    # task = 'CAUSAL_LM'
    desired_rank = 8
    lora_alpha = 32
    lora_dropout = 0.1
    lora_config = LoraConfig(
    #     task_type = task,
        r = desired_rank,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        target_modules=target_modules,
        bias = 'none',
    #     use_dora = True only for DoRA
    )
    train_loader_images = '' # LOAD IT
    train_dataset = '' #LOAD IT
    peft_model = get_peft_model(model,lora_config)
    peft_model.print_trainable_parameters()
    vit = DinoVisionTransformerClassifier(peft_model,processor,train_dataset.c,device).to(device)
    num_epochs = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vit.parameters(), lr=1e-6)
    epoch_losses = []
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)

        batch_losses = []
        for idx,data in enumerate(train_loader_images):
            # get the input batch and the labels
            batch_of_images, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # model prediction
            output = vit(batch_of_images.to(device)).squeeze(dim=1)

            # compute loss and do gradient descent
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            print(f"step:{idx} Loss: {loss.item()}")
        epoch_losses.append(np.mean(batch_losses))
        print(f"Mean epoch loss: {epoch_losses[-1]}")

    print("Finished training!")