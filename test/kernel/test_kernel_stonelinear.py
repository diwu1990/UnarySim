import torch
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from UnarySim.kernel import FSULinearStoNe
from UnarySim.stream import RNG, BinGen, BSGen
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_stonelinear(epochs=1):
    dataset = "mnist"
    time_step = 10
    mode = "bipolar"
    widthw = 12
    batch_size_train = 256
    batch_size_test = 256
    print("Dataset: " + dataset)

    hwcfg={
            "mode" : mode,
            "format" : "bfloat16",
            "widthw" : widthw,
            "scale" : 1.2,
            "depth" : 20,
            "leak" : 0.5,
            "widthg" : 1.25
        }

    class MLPStoNe(torch.nn.Module):
        def __init__(self,time_step, hwcfg):
            super(MLPStoNe, self).__init__()
            
            self.time_step = time_step
            
            self.fc_1 = FSULinearStoNe(in_features=28*28, out_features=512,bias=False, hwcfg=hwcfg)
            self.fc_out = FSULinearStoNe(in_features=512, out_features=10,bias=False, hwcfg=hwcfg)

            self.rng = RNG(
                        hwcfg={
                            "width" : widthw,
                            "rng" : "Sobol",
                            "dimr" : 1
                        },
                        swcfg={
                            "rtype" : torch.float
                        })()
        
        def forward(self, inp):
            u_out = 0
            inp = inp.view(inp.shape[0],-1)
            binary = BinGen(
                        inp, 
                        hwcfg={
                            "width" : widthw,
                            "mode" : mode
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
            u1_list = [0]
            u2_list = [0]
            for t in range(self.time_step):
                spike_inp = bsgen(torch.zeros_like(inp, dtype=torch.long)+t)
                x, _, u1  = self.fc_1(spike_inp, u1_list[-1])
                u1_list.append(u1)
                _, us, u2 = self.fc_out(x, u2_list[-1])
                u2_list.append(u2)
                u_out = u_out + us
            return u_out

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10(root='/mnt/ssd1/data/', train=True,
                                        download=True, transform=transform_train)
    train_loader_cifar10 = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                            shuffle=True, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='/mnt/ssd1/data/', train=False,
                                        download=True, transform=transform_test)
    test_loader_cifar10 = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                         shuffle=False, num_workers=4, pin_memory=True)
                                         
    train_loader_mnist = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/mnt/ssd1/data/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size_train, shuffle=True,drop_last=True)

    test_loader_mnist = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/mnt/ssd1/data/', train=False, download=True,
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

    if dataset == "cifar10":
        train_loader = train_loader_cifar10
        test_loader = test_loader_cifar10
    elif dataset == "mnist":
        train_loader = train_loader_mnist
        test_loader = test_loader_mnist
        
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            for i, (data, target) in enumerate(train_loader):
                # running_loss = 0.0
                # print(i)
                # zero the parameter gradients
                optimizer.zero_grad()
                data, target = data.to(device), target.to(device)

                out = model(data)
                loss = criterion(out, target)
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # if i % 20 == 19:    # print every 20 mini-batches
                #     print('[%d, %5d] current average batch loss: %.3f' %
                #         (epoch + 1, i + 1, running_loss / (i+1)))
            scheduler.step()
            print('Epoch-%3d overall average batch loss: %.3f' %
                    (epoch + 1, running_loss / (i+1)))
            
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
            print('Accuracy of the network on the 10000 test images: %.2f %% (Epoch runtime: %.2f secs)' % (
                100 * correct / total, time.time()-start_time))

    print('Finished Training')


if __name__ == '__main__':
    test_stonelinear(epochs=30)

