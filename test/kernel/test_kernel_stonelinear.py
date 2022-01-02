import torch
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from UnarySim.kernel import FSULinearStoNe
from UnarySim.stream import RNG, BinGen, BSGen


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_stonelinear():
    time_step = 10
    hwcfg={
            "mode" : "bipolar",
            "format" : "float32",
            "widthw" : 8,
            "scale" : 512,
            "depth" : 20,
            "leak" : 1,
            "widthg" : 1.25
        }
    data = "mnist"

    def PoissonGen(inp, rescale_fac=1.0):
        rand_inp = torch.rand_like(inp)
        return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))

    class MLPStoNe(torch.nn.Module):
        def __init__(self,time_step, hwcfg):
            super(MLPStoNe, self).__init__()
            
            self.time_step = time_step
            
            self.fc_1 = FSULinearStoNe(in_features=28*28, out_features=512,bias=False, hwcfg=hwcfg)
            self.fc_out = FSULinearStoNe(in_features=512, out_features=10,bias=False, hwcfg=hwcfg)

            self.rng = RNG(
                        hwcfg={
                            "width" : 12,
                            "rng" : "Sobol",
                            "dimr" : 1
                        },
                        swcfg={
                            "rtype" : torch.float
                        })()
        
        def forward(self, inp):
            torch.autograd.set_detect_anomaly(False)
            u_out = 0
            inp = inp.view(inp.shape[0],-1)
            binary = BinGen(
                        inp, 
                        hwcfg={
                            "width" : 12,
                            "mode" : "unipolar"
                        },
                        swcfg={
                            "rtype" : torch.float
                        })()
            bsgen = BSGen(
                        binary, 
                        self.rng, 
                        swcfg={
                            "stype" : torch.float
                        })
            for t in range(self.time_step):
                
                # spike_inp = PoissonGen(inp)
                spike_inp = bsgen(torch.zeros_like(inp, dtype=torch.long)+t)
                x, _  = self.fc_1(spike_inp)
                _, U = self.fc_out(x)
                u_out = u_out + U
            return u_out



    batch_size_train = 256
    batch_size_test = 256

    train_loader_mnist = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/mnt/ssd1/data/', train=True, download=False,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size_train, shuffle=True,drop_last=True)

    test_loader_mnist = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/mnt/ssd1/data/', train=False, download=False,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size_test, shuffle=True,drop_last=True)


    

    model = MLPStoNe(time_step, hwcfg).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if data == "cifar10":
        train_loader = train_loader_cifar10
        test_loader = test_loader_cifar10
    elif data == "mnist":
        train_loader = train_loader_mnist
        test_loader = test_loader_mnist
        
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(30):
            model.train()
            running_loss = 0.0
            for i, (data, target) in enumerate(train_loader):
                # running_loss = 0.0
                
                # zero the parameter gradients
                optimizer.zero_grad()
                data, target = data.to(device), target.to(device)

                out = model(data)
                loss = criterion(out, target)
                # torch.autograd.set_detect_anomaly(True)
                
                loss.backward(retain_graph=True)
                optimizer.step()
                
                # print statistics
                running_loss += loss.item()
                # if i % 100 == 99:    # print every 100 mini-batches
                #     print('[%d, %5d] current average batch loss: %.3f' %
                #         (epoch + 1, i + 1, running_loss / (i+1)))
            print('[%d, %5d] overall average batch loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / (i+1)))
            
            ## Test each epoch
            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                model.eval()
                for data in test_loader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    # calculate outputs by running images through the network
                    outputs = model(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %.2f %%' % (
                100 * correct / total))

    print('Finished Training')






if __name__ == '__main__':
    test_stonelinear()


