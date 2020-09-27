from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

############ Style Loss ############
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class StyleTransfer():
  def __init__(self, content_img, style_img, iterations):
    self.args = {
        'img_size': 512,
        'num_steps': iterations,
        'style_weight': 1000000, 
        'content_weight': 1
    }
    self.args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = nn.Sequential()
    self.content_losses = []
    self.style_losses   = []


    self.loader = transforms.Compose([
      transforms.Resize((self.args['img_size'])),  # scale imported image
      transforms.CenterCrop(((self.args['img_size']), (self.args['img_size']))),
      transforms.ToTensor(),  # transform it into a torch tensor
      # transforms.Normalize(
      #     [0.485, 0.456, 0.406],
      #     [0.229, 0.224, 0.225]
      # )
      ])

    self.show_imgs   = True
    self.content_img = self.image_loader(content_img, 'Entrada')
    self.style_img   = self.image_loader(style_img, 'Estilo')

  def image_loader(self, image_name, type_):
    image = Image.open(image_name)
    image = self.loader(image)
    
    if self.show_imgs:
      plt.figure(figsize=(7,7))
      plt.imshow(image.permute(1,2,0).cpu().detach())
      plt.title(type_ + ': ' + image_name)
      plt.axis('off')
      plt.show()
    
    return image.unsqueeze(0).to(self.args['device'], torch.float)

  def build_model(self):
    cnn = models.vgg19(pretrained=True).features.to(self.args['device']).eval()

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    content_losses = []
    style_losses = []

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        self.model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = self.model(self.content_img).detach()
            content_loss = ContentLoss(target)
            self.model.add_module("content_loss_{}".format(i), content_loss)
            self.content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = self.model(self.style_img).detach()
            style_loss = StyleLoss(target_feature)
            self.model.add_module("style_loss_{}".format(i), style_loss)
            self.style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(self.model) - 1, -1, -1):
        if isinstance(self.model[i], ContentLoss) or isinstance(self.model[i], StyleLoss):
            break

    self.model = self.model[:(i + 1)]

  def run(self,):
    self.build_model()

    #### Otimizador recebe a imagem original para otimizar
    input_img = self.content_img.clone()
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    print('Optimizing..')
    run = [0]
    while run[0] <= self.args['num_steps']-100:

      def closure():
          # correct the values of updated input image
        input_img.data.clamp_(0, 1)

        optimizer.zero_grad()

        self.model(input_img)
        style_score = 0
        content_score = 0

        for sl in self.style_losses:
            style_score += sl.loss
        for cl in self.content_losses:
            content_score += cl.loss

        style_score *= self.args['style_weight']
        content_score *= self.args['content_weight']

        loss = style_score + content_score
        loss.backward()

        run[0] += 1
        if run[0] % 50 == 0:
            print("run {}:".format(run))
            print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                style_score.item(), content_score.item()))
            print()

        return style_score + content_score

      optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)
    return input_img[0].permute(1,2,0).cpu().detach()
    







