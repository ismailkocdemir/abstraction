import torch
from torch import nn
from torch import optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import argparse

from autograd_hacks import *
import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='Information Discarding')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
					choices=model_names,
					help='model architecture: ' + ' | '.join(model_names) +
					' (default: vgg19)')

parser.add_argument('-c', '--checkpoint', type=str, 
					help='checkpoint file')
parser.add_argument('-n', '--sample-size', default=1, type=int,
					metavar='N', help='mini-batch size (default: 1)')

normalization_values_per_layer = {}

class Subnetwork(nn.Module): 
	def __init__(self, model, layer_name):
		super(Subnetwork, self).__init__()
		
		self.layer_name = layer_name
		self._module_list = []

		flag = True
		for _name, _module in model.named_modules():
			if isinstance(_module, (nn.Sequential, models.PreActBlock, models.PreActResNet, models.VGG)):
				continue

			if isinstance(_module, nn.Linear) and flag:
				flag = False
				self._module_list.append(models.Flatten())
			
			self._module_list.append(_module)
			
			if _name == layer_name:
				break
		
		self.subnetwork = nn.Sequential(*self._module_list)

	def forward(self, x):
		for idx, _module in enumerate(self._module_list):
			
			x = _module(x)
		
		#return self.subnetwork(x)
		return x

def calculate_normalization(sampled_x, model, reduced_axes=None):
	add_hooks(model, grad1=False)

	_ = model(sampled_x)

	print("Calculating normalization for all layers")
	activations_dict = get_activations()
	for layer_name, layer_act in activations_dict.items():
		normalization_values_per_layer[layer_name] = np.std(layer_act.cpu().numpy(), axis=0)

	remove_hooks(model)

	return

class InfoDiscarding(nn.Module):
	def __init__(self, x, subnetwork, scale=0.5, rate=0.1, normalization=None):

		super(InfoDiscarding, self).__init__()
		self.shape = x.size()
		self.ratio = nn.Parameter(torch.randn(self.shape), requires_grad=True).to(models.device)

		self.normalization = normalization

		if self.normalization is not None:
			self.normalization = nn.Parameter(torch.tensor(self.normalization).to(models.device), requires_grad=False)

		self.scale = scale
		self.rate = rate
		self.x = x
		self.subnetwork = subnetwork

	def reparameterize(self, ):
		_scale = torch.sigmoid(self.ratio)
		_std = torch.exp(0.5*torch.log(_scale))
		_eps = torch.randn_like(_std)
		return _eps * _std * self.scale

	def forward(self,):
		x = self.x + 0.  
		x_tilde = x + self.reparameterize()  #ratios * torch.randn(self.shape).to(models.device) * self.scale
		
		features = self.subnetwork(x) 
		features_tilde = self.subnetwork(x_tilde)
		loss = (features_tilde - features) ** 2
		
		if self.normalization is not None:
			loss = torch.mean(loss / self.normalization ** 2)
		else:
			loss = torch.mean(loss) / torch.mean(features ** 2)

		ratios = torch.sigmoid(self.ratio)  
		return loss - torch.mean(torch.log(ratios)) * self.rate

	def optimize(self, iteration=1000, lr=0.01, show_progress=False):
		minLoss = None
		state_dict = None
		optimizer = optim.Adam(self.parameters(), lr=lr)
		self.train()
		func = (lambda x: x) if not show_progress else tqdm
		for _ in func(range(iteration)):
			optimizer.zero_grad()
			loss = self()
			loss.backward()
			optimizer.step()
			if minLoss is None or minLoss > loss:
				state_dict = {k:self.state_dict()[k] + 0. for k in self.state_dict().keys()}
				minLoss = loss
		self.eval()
		self.load_state_dict(state_dict)

	def get_sigma(self):
		ratios = torch.sigmoid(self.ratio)  # S * 1
		return ratios.detach().cpu().numpy() * self.scale
	
	def visualize(self, save_path="info_discarding_temp.png", title="Information Discarding"):
		_sigma = self.get_sigma()[0].transpose(1,2,0).mean(axis=2) / self.scale

		fig = plt.figure()
		plt.title(title)
		plt.imshow(_sigma, cmap='viridis', vmin=0, vmax=1)
		plt.colorbar()
		plt.tight_layout()
		plt.savefig(save_path)
		plt.close(fig)


def sample_data_from_CIFAR10(sample_size, workers=4):
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										 std=[0.229, 0.224, 0.225])
	_transforms_val = transforms.Compose([
								transforms.ToTensor(),
								normalize,
							])


	_transforms_val_un = transforms.Compose([
								transforms.ToTensor(),
							])


	dataset_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10(root='./data', train=False, transform=_transforms_val, download=True),
						batch_size=sample_size, shuffle=False,
						num_workers=workers, pin_memory=True)


	dataset_loader_un = torch.utils.data.DataLoader(
		datasets.CIFAR10(root='./data', train=False, transform=_transforms_val_un, download=True),
						batch_size=sample_size, shuffle=False,
						num_workers=workers, pin_memory=True)


	_samples = None
	for _input, _ in dataset_loader:
		_samples = _input
		break

	_samples_un = None
	for _input, _ in dataset_loader_un:
		_samples_un = _input
		break

	return _samples, _samples_un
	
def plot_layerwise_discarding(arch, epoch, idx, sid_dict, scale):
	layers = list(sid_dict.keys())
	values = list(sid_dict.values())

	fig = plt.figure(figsize=(8,5))

	plt.ylabel("Avg. Information Discarding")
	plt.xlabel("Layer")
	plt.xticks(rotation=45)
   
	#plt.ylim((0,scale))
	plt.subplots_adjust(bottom=0.25)

	plt.title("{}, epoch {}, sample-{}".format(arch, epoch, idx))    
	plt.plot(layers, values)

	plt.savefig("figures/information_discarding/layerwise_discarding/{}_epoch{}_sample{}.png".format(arch, epoch, idx))
	plt.close(fig)    

def plot_sample(sample, arch, idx):

	fig = plt.figure()
	plt.imshow(sample.cpu().numpy().transpose(1,2,0))
	plt.savefig("figures/information_discarding/samples/{}_{}.png".format(arch, idx))
	plt.close(fig)

def _calculater_sid_per_layer_for_checkpoint(arch, checkpoint_file, sample_size):
	global normalization_values_per_layer

	dataset, dataset_un = sample_data_from_CIFAR10(sample_size)
	epoch = checkpoint_file.split("/")[-1].split("_")[-1].split(".")[0]

	print("Calculating SID for layers in model {}, from checkpoint {}".format(arch, epoch))
	
	pretrained_model = models.__dict__[arch](num_classes=10)
	checkpoint = torch.load(checkpoint_file)
	pretrained_model.load_state_dict(checkpoint['state_dict'])

	normalization = calculate_normalization(dataset, pretrained_model)

	#for idx, sample in enumerate(dataset_un):
	#	plot_sample(sample, arch, idx)

	val_img_indices = [9,55,60]
	total_size = len(val_img_indices)
	
	for idx, sample in enumerate(dataset):
		if idx not in val_img_indices:
			continue

		print("Sample {} in progress".format(idx+1))

		_single_item_batch = torch.from_numpy(np.array(sample)[np.newaxis, :]).to(models.device)
		sid_per_layer = {}
		for layer_name, layer_normalization in normalization_values_per_layer.items():
			subnetwork = Subnetwork(pretrained_model, layer_name).to(models.device)

			info_dis = InfoDiscarding(_single_item_batch, subnetwork, normalization=layer_normalization, scale=10, rate=1.5)
			info_dis.optimize()
			sid = info_dis.get_sigma().mean()
			sid_per_layer[layer_name] = sid
			print("avg. SID of {}: {}".format(layer_name, sid))

			save_path = "figures/information_discarding/heatmaps/ID_{}_{}_sample_{}.png".format(arch, layer_name, idx)
			title = "{} {} sample-{}".format(arch, layer_name, idx)
			
			info_dis.visualize(save_path, title)
		
		plot_layerwise_discarding(arch, epoch, idx, sid_per_layer, scale=10)

	
	return sid_per_layer

if __name__ == '__main__':
	args = parser.parse_args()

	_calculater_sid_per_layer_for_checkpoint(arch=args.arch, checkpoint_file=args.checkpoint, sample_size=args.sample_size)


"""
class InfoDiscarding(nn.Module):
	def __init__(self, sigma_shape, sub_network, delta_f):

		self.sub_network = sub_network.to(device)
		self.sub_network.requires_grad = False

		self.C = torch.zeros(sigma_shape, requires_grad=False) 
					+ 0.5 * math.log(2*math.pi*math.e).to(device)

		self.sigma = torch.DoubleTensor(sigma_shape, 
			requires_grad=True).to(device)

		self.tau = 1e-2
		self.alpha = 1.5
		self.delta_f = delta_f

	def _reparameterize(self):
		_std = torch.exp(0.5*torch.log(self.sigma))
		_eps = torch.randn_like(_std)
		return _eps * _std

	def forward(self, input):
		_noise = self._reparameterize()
		_perturbed_input = input + _noise
		_features = self.sub_network(_perturbed_input)
		return _features
	
	def entropy_loss(input_features, target_features):
		_mse = F.mse_loss(input_features, output_features) 
				#torch.mean( (input_features - target_features)**2 , dim=0 )
		
		_normalizationd_mse = ( 1/self.delta_f**2 ) * _mse
		_entropy_sum = torch.sum(torch.log(self.sigma) + + self.C) 

		with torch.no_grad():
			_lambda = (self.alpha*self.delta_f**2) / _entropy_sum

		_adjusted_lagrangian = _normalizationd_mse - _lambda * _entropy_sum
		return  _adjusted_lagrangian


"""



