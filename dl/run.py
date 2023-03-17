import time
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from RNN import Config, Model
from dl.data_loader import DiabetesDataset

from dl.train_eval import train

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

start_time = time.time()

config = Config()
train_dataset = DiabetesDataset(is_train=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_works)
test_dataset = DiabetesDataset(is_train=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_works)

model = Model(config).to(config.device)
print(model)
writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
model.train()
train(config, model, train_loader, test_loader, writer)