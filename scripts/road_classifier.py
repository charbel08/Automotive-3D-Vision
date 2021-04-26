wimport numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from load_images import load_images
import cv2
import sys


class FCN(nn.Module):
    def __init__(self, n=6):
        super(FCN, self).__init__()
                
        self.n = n
        self.leaky_relu = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(in_channels=6,
                               out_channels=n,
                               kernel_size=5,
                               padding=2)
          
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(in_channels=n,
                               out_channels=n*2,
                               kernel_size=5,
                               padding=2)


        self.conv3 = nn.Conv2d(in_channels=n*2,
                               out_channels=n*2,
                               kernel_size=5,
                               padding=2)
        
        self.pool = nn.MaxPool2d(2, 2)
                
        self.conv4 = nn.Conv2d(in_channels=n*2,
                               out_channels=n*4,
                               kernel_size=5,
                               padding=2)
        
        self.tconv1 = nn.ConvTranspose2d(in_channels=n*4,
                                         out_channels=n*2,
                                         kernel_size=5,
                                         padding=2,
                                         stride=2,
                                         output_padding=1)
        
        self.conv5 = nn.Conv2d(in_channels=n*2,
                               out_channels=n*2,
                               kernel_size=5,
                               padding=2)
        
        self.tconv2 = nn.ConvTranspose2d(in_channels=n*2,
                                         out_channels=n,
                                         kernel_size=5,
                                         padding=2,
                                         stride=2,
                                         output_padding=1)
        
        self.conv6 = nn.Conv2d(in_channels=n,
                               out_channels=2,
                               kernel_size=5,
                               padding=2)
        
        
    def forward(self, x):
        
        x = self.pool(self.leaky_relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.leaky_relu(self.tconv1(x))
        x = self.leaky_relu(self.conv5(x))
        x = self.leaky_relu(self.tconv2(x))
        x = self.conv6(x)
        return x
        

def get_fcn_accuracy(model, X, T):
    model.eval()
    out = model(X).permute(0, 3, 2, 1).reshape(-1, 2)
    pred = out.argmax(dim=1).cpu().numpy()
    t = T.reshape(-1)
    return (pred==t.cpu().numpy()).mean()


def train_fcn(model, X, T, Xval, Tval, batch_size=10, weight_decay=1e-5,
          learning_rate=0.001, num_epochs=250, checkpoint_path=None):
  
    # Defining our loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)
    
    # Initializing lists to track performancee
    iters, train_loss, train_accs, val_accs = [], [], [], []
    
    # Sending model and arrays to GPU (if available)
    model = model.to(device)
    X = torch.Tensor(X).to(device)
    T = torch.Tensor(T).long().to(device)
    Xval = torch.Tensor(Xval).to(device)
    Tval = torch.Tensor(Tval).long().to(device)

    n = 0
    for epoch in range(num_epochs):
        
        # Shuffle data at each epoch
        s = np.arange(X.shape[0])
        np.random.shuffle(s)
        X = X[s]
        T = T[s]
        
        # Plot the prediction for the validation image plotted above
        if (epoch % 3 == 0 and epoch <= 15) or (epoch % 10 == 0 and epoch > 20):
            track = model(Xval[0].unsqueeze(dim=0))
            seg = track.argmax(dim=1).permute(2, 1, 0).squeeze().cpu().numpy()
            plt.imshow(seg)
            plt.title("Prediction at epoch " + str(epoch))
            plt.show()
            
        # Split data into batches
        N, C, W, H = X.shape
        Xbatches = X.reshape(N//batch_size, batch_size, C, W, H)
        Tbatches = T.reshape(N//batch_size, batch_size, H, W)

        for i in range(N//batch_size):
            
            # Extract current batch
            batch = Xbatches[i]
            target = Tbatches[i]
            
            # Annotate model for training
            model.train()
            
            # Forwards pass
            out = model(batch)
            
            # Backpropagation
            loss = criterion(out.permute(0, 3, 2, 1).reshape(-1, 2), target.reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            n += 1
        
        # Checkpointing parameters
        if (checkpoint_path is not None) and n > 0:
            torch.save(model.state_dict(), checkpoint_path.format(epoch), _use_new_zipfile_serialization=False)
        
        # Computing accuracies and loss
        train_acc = get_fcn_accuracy(model, X, T)
        val_acc = get_fcn_accuracy(model, Xval, Tval)
        loss = float(loss)/batch_size
        print("Epoch %d; Loss %f; Train acc %f; Val acc %f" % (epoch+1, loss, train_acc, val_acc))

        iters.append(n)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_loss.append(loss)

    # Plotting
    plt.title("Learning Curve")
    plt.plot(iters, train_loss, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Accuracy")
    plt.plot(iters, train_accs, label="Train")
    plt.plot(iters, val_accs, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.show()


          
def create_data_features(left_imgs, mags, LoGs, gt_road):
    
    data, labels = [], []
    
    for key, img in left_imgs.items():
        if img.shape == (375, 1242, 3) and key != "um_000000":
            
            # Getting the targets
            split = key.split("_")
            split.insert(1, "road")
            target = cv2.cvtColor(gt_road["_".join(split)], cv2.COLOR_BGR2GRAY)//255
            
            # Getting the feature
            img = (left_imgs[key]/255) - 0.5
            mag = mags[key]
            mag = (mag/np.max(mag)) - 0.5
            LoG = LoGs[key]
            LoG = (LoG/np.max(LoG)) - 0.5
            depth = depths[key]
            depth = (depth/np.max(depth)) - 0.5
            
            # Stacking them along the channel dimension and cropping the sky
            features = np.dstack((img, mag, LoG, depth))[127:, 2:, :]
            features = np.transpose(features, (2, 1, 0))
            data.append(features)
            
            # Cropping the sky
            labels.append(target[127:, 2:])

    return np.stack(data), np.stack(labels)


if __name__ == "__main__":

    path = sys.argv[1]
    left_imgs, mags, LoGs = load_images(path, get_mag=True, get_LoG=True)
    gt_road, _, _ = load_images("../data/train/gt_image_left")

    data, labels = create_data_features(left_imgs, mags, LoGs, gt_road)
        
    # Splitting validation and training data
    fcn_train, fcn_ttrain = data[:110], labels[:110]
    fcn_val, fcn_tval = data[110:], labels[110:]
    
    # Code that would train the network, but I already trained it so instead, I just
    # import the pre-trained weights
    
    #fcn = FCN()
    #train_fcn(fcn, fcn_train, fcn_ttrain, fcn_val, fcn_tval,
    #      checkpoint_path='/content/gdrive/My Drive/CSC420/fcn_weights/ckpt-{}.pk')
    
    
    fcn_road_classifier = FCN()
    fcn_road_classifier.load_state_dict(torch.load('../fcn_weights/ckpt-240.pk', map_location=torch.device('cpu')))
    
    
    inp = np.expand_dims(data[0], axis=0)
    inp = np.transpose(inp, (0, 3, 2, 1))
    out = fcn_road_classifier(torch.Tensor(inp).to(device))

    pred = out.argmax(dim=1).permute(2, 1, 0).squeeze().cpu().numpy()
    plt.imshow(pred)
    plt.title("Predicted road pixels (FCN)")
