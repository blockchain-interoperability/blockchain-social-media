import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.dataset import TwitterDataset




class LinearAutoEncoder(nn.Module):
    def __init__(self):
        super(LinearAutoEncoder, self).__init__()
       
        #Encoder

        self.encoder = nn.Sequential(
            nn.Linear(384, 200),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2,2),
            nn.Linear(200,50),
            nn.ReLU(inplace=True),
            nn.Linear(50,10),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2,2),
        )
        # nn.MaxPool2d(2, 2)
       
        #Decoder
        self.decoder = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200,384),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))



class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
       
        #Encoder

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        # nn.MaxPool2d(2, 2)
       
        #Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))



# lenc = lenc.cuda()
# ebs = ebs.cuda()

# from tqdm.auto import tqdm
# from torch.optim import Adam

# criterion = nn.MSELoss()
# optimizer = Adam(lenc.parameters())

# for epoch in tqdm(range(100)):
#     optimizer.zero_grad()
#     out = lenc(ebs)
#     loss = criterion(out,ebs)
#     loss.backward()
#     optimizer.step()
#     print('loss:',loss.item())
def RMSELoss(yhat,y):
    return (((yhat-y)**2).mean()+1e-7).sqrt()


ENCODER_MAPPING = {
    'linear': LinearAutoEncoder
}



'''
# result for linear:
train_loss: 0.045552572654371355  -- val_loss: 0.046404985962687315                       
train_loss: 0.043293155760902494  -- val_loss: 0.04616103762490743                        
train_loss: 0.042884061317645945  -- val_loss: 0.04580690225160679                        
train_loss: 0.042666319261879714  -- val_loss: 0.04548883599753169                        
train_loss: 0.04255907804678535  -- val_loss: 0.04548084332037909                         
train_loss: 0.042432036920164146  -- val_loss: 0.04527067876894585                        
train_loss: 0.04234932652971398  -- val_loss: 0.045193105498534746                        
train_loss: 0.042281313528406014  -- val_loss: 0.04520549797052541                        
final test performance: 0.045206617563962936
'''

def train_encoder(
    timestamp_path,
    sentiment_path,
    embedding_path,
    whole_text_path = '',
    token_path = '',
    encoder_type = 'linear',
    save_path = '',
    plot_path = '',
    splits = [8,1,1]
):
    dset = TwitterDataset(
        timestamp_path,
        sentiment_path,
        embedding_path = embedding_path,
        whole_text_path = whole_text_path,
        token_path = token_path,
    )

    idxs = torch.arange(len(dset))
    offset = 0
    split_idxs = []
    for s in splits:
        size = int(s/sum(splits) * len(dset))
        split_idxs.append(idxs[offset:size+offset])
        offset += size

    train_idxs,val_idxs,test_idxs = split_idxs

    train_loader = DataLoader(dset,batch_size = 1024,sampler = SequentialSampler(train_idxs))
    val_loader = DataLoader(dset,batch_size = 1024,sampler = SequentialSampler(val_idxs))
    test_loader = DataLoader(dset,batch_size = 1024,sampler = SequentialSampler(test_idxs))    

    model = ENCODER_MAPPING[encoder_type]().cuda()
    criterion = RMSELoss
    optimizer = Adam(model.parameters())

    train_losses = []
    val_losses = []

    old_val_loss = 1000000
    for epoch in tqdm(range(100)):
        train_loss = []
        val_loss = []
        model.train()
        for batch in tqdm(train_loader,leave=False):
            inp = batch['embedding'].cuda()
            optimizer.zero_grad()
            out = model(inp)
            loss = criterion(out,inp)
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()

        model.eval()
        for batch in val_loader:
            inp = batch['embedding'].cuda()
            out = model(inp)
            loss = criterion(out,inp)
            val_loss.append(loss.item())

        # print(f'{epoch:04d} train_loss: {sum(train_loss)/len(train_loss)}  -- val_loss: {sum(val_loss)/len(val_loss)}',)
        train_losses.append(sum(train_loss)/len(train_loss))
        val_losses.append(sum(val_loss)/len(val_loss))
        

        if sum(val_loss)/len(val_loss) > old_val_loss:
            print('I think we are done')
            break
        else: old_val_loss = sum(val_loss)/len(val_loss)


    df = pd.DataFrame()
    # df['Epoch'] = list(range(8))
    df['Train Loss'] = train_loss
    df['Validation Loss'] = val_loss

    fig,ax = plt.subplots(figsize=(6,3))
    sns.lineplot(
        data = df,
        ax = ax
    )
    fig.savefig(plot_path)

    model.eval()
    test_real = []
    test_pred = []
    for batch in tqdm(test_loader,desc='final testing...'):
        inp = batch['embedding'].cuda()
        out = model(inp)
        loss = criterion(out,inp)

        test_real.append(inp.cpu())
        test_pred.append(out.cpu())
    

    
    test_performance = RMSELoss(torch.vstack(test_pred),torch.vstack(test_real))
    print(f'final test performance: {test_performance}')

    save_path = Path(save_path) / f'{model.__class__.__name__}.pkl'
    torch.save(model.state_dict(), save_path)


def load_encoder(
    encoder_path,
    encoder_type
):
    model = ENCODER_MAPPING[encoder_type]()
    model.load_state_dict(torch.load(encoder_path))
    return model