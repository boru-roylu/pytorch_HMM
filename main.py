import torch
from models import HMM
from data import get_datasets, read_config
from training import Trainer

torch.autograd.set_detect_anomaly(True)

# Generate datasets from text file
path = "data"
N = 3
config = read_config(N, path)
train_dataset, valid_dataset = get_datasets(config, parent_dir="./data/agent")
checkpoint_path = "."

# Initialize model
model = HMM(config.N, config.M)

# Train the model
num_epochs = 200
trainer = Trainer(model, config, lr=0.03)
trainer.load_checkpoint(checkpoint_path)

split_every = 10

for epoch in range(num_epochs):
    print("========= Epoch %d of %d =========" % (epoch+1, num_epochs))
    train_loss = trainer.train(train_dataset)
    valid_loss = trainer.test(valid_dataset)

    if (epoch != 0) and (epoch % split_every == 0):
        if not trainer.split_state(train_loss):
            trainer.save_checkpoint(epoch, checkpoint_path)
            break

    trainer.save_checkpoint(epoch, checkpoint_path)


    print("========= Results: epoch %d of %d =========" % (epoch+1, num_epochs))
    print("train loss: %.2f| valid loss: %.2f\n" % (train_loss, valid_loss) )
