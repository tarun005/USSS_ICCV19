import pdb

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

# BatchNorm = nn.BatchNorm2d


# __all__ = ['DRN', 'drn26', 'drn42', 'drn58']


webroot = 'https://tigress-web.princeton.edu/~fy/drn/models/'

model_urls = {
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'drn-c-26': webroot + 'drn_c_26-ddedf421.pth',
	'drn-c-42': webroot + 'drn_c_42-9d336e8c.pth',
	'drn-c-58': webroot + 'drn_c_58-0a53a92c.pth',
	'drn-d-22': webroot + 'drn_d_22-4bd2f8ea.pth',
	'drn-d-38': webroot + 'drn_d_38-eebb45f0.pth',
	'drn-d-54': webroot + 'drn_d_54-0e0534ff.pth',
	'drn-d-105': webroot + 'drn_d_105-12b40979.pth'
}


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None,
				 dilation=(1, 1), residual=True):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride,
							 padding=dilation[0], dilation=dilation[0])
		self.bn1 = BatchNorm(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes,
							 padding=dilation[1], dilation=dilation[1])
		self.bn2 = BatchNorm(planes)
		self.downsample = downsample
		self.stride = stride
		self.residual = residual

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)
		if self.residual: ## Noresidual connection in degridding networks
			out += residual
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None,
				 dilation=(1, 1), residual=True):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = BatchNorm(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=dilation[1], bias=False,
							   dilation=dilation[1])
		self.bn2 = BatchNorm(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = BatchNorm(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual ## Always there is a residual connection
		out = self.relu(out)

		return out


class DRN(nn.Module):

	def __init__(self, block, layers, num_classes=1000,
				 channels=(16, 32, 64, 128, 256, 512, 512, 512),
				 out_map=False, out_middle=False, pool_size=28, arch='D'):
		super(DRN, self).__init__()
		self.inplanes = channels[0]
		self.out_map = out_map
		self.out_dim = channels[-1]
		self.out_middle = out_middle
		self.arch = arch

		if arch == 'C':
			self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
								   padding=3, bias=False)
			self.bn1 = BatchNorm(channels[0])
			self.relu = nn.ReLU(inplace=True)

			self.layer1 = self._make_layer(
				BasicBlock, channels[0], layers[0], stride=1)
			self.layer2 = self._make_layer(
				BasicBlock, channels[1], layers[1], stride=2)
		elif arch == 'D':
			self.layer0 = nn.Sequential(
				nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3,
						  bias=False),
				BatchNorm(channels[0]),
				nn.ReLU(inplace=True)
			)

			self.layer1 = self._make_conv_layers(
				channels[0], layers[0], stride=1)
			self.layer2 = self._make_conv_layers(
				channels[1], layers[1], stride=2)

		self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
		self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
		self.layer5 = self._make_layer(block, channels[4], layers[4],
									   dilation=2, new_level=False)
		self.layer6 = None if layers[5] == 0 else \
			self._make_layer(block, channels[5], layers[5], dilation=4,
							 new_level=False)

		if arch == 'C':
			self.layer7 = None if layers[6] == 0 else \
				self._make_layer(BasicBlock, channels[6], layers[6], dilation=2,
								 new_level=False, residual=False)
			self.layer8 = None if layers[7] == 0 else \
				self._make_layer(BasicBlock, channels[7], layers[7], dilation=1,
								 new_level=False, residual=False)
		elif arch == 'D':
			self.layer7 = None if layers[6] == 0 else \
				self._make_conv_layers(channels[6], layers[6], dilation=2)
			self.layer8 = None if layers[7] == 0 else \
				self._make_conv_layers(channels[7], layers[7], dilation=1)

		if num_classes > 0:
			self.avgpool = nn.AvgPool2d(pool_size)
			self.fc = nn.Conv2d(self.out_dim, num_classes, kernel_size=1,
								stride=1, padding=0, bias=True)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, BatchNorm):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
					new_level=True, residual=True):
		assert dilation == 1 or dilation % 2 == 0
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				BatchNorm(planes * block.expansion),
			)

		layers = list()
		layers.append(block(
			self.inplanes, planes, stride, downsample,
			dilation=(1, 1) if dilation == 1 else (
				dilation // 2 if new_level else dilation, dilation),
			residual=residual))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, residual=residual,
								dilation=(dilation, dilation)))

		return nn.Sequential(*layers)

	def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
		modules = []
		for i in range(convs):
			modules.extend([
				nn.Conv2d(self.inplanes, channels, kernel_size=3,
						  stride=stride if i == 0 else 1,
						  padding=dilation, bias=False, dilation=dilation),
				BatchNorm(channels),
				nn.ReLU(inplace=True)])
			self.inplanes = channels
		return nn.Sequential(*modules)

	def forward(self, x):
		y = list()

		if self.arch == 'C':
			x = self.conv1(x)
			x = self.bn1(x)
			x = self.relu(x)
		elif self.arch == 'D':
			x = self.layer0(x)

		x = self.layer1(x)
		y.append(x)
		x = self.layer2(x)
		y.append(x)

		x = self.layer3(x)
		y.append(x)

		x = self.layer4(x)
		y.append(x)

		x = self.layer5(x)
		y.append(x)

		if self.layer6 is not None:
			x = self.layer6(x)
			y.append(x)

		if self.layer7 is not None:
			x = self.layer7(x)
			y.append(x)

		if self.layer8 is not None:
			x = self.layer8(x)
			y.append(x)

		if self.out_map:
			x = self.fc(x)
		else:
			x = self.avgpool(x)
			x = self.fc(x)
			x = x.view(x.size(0), -1)

		if self.out_middle:
			return x, y
		else:
			return x


class DRN_A(nn.Module):

	def __init__(self, block, layers, num_classes=1000):
		self.inplanes = 64
		super(DRN_A, self).__init__()
		self.out_dim = 512 * block.expansion
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
									   dilation=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
									   dilation=4)
		self.avgpool = nn.AvgPool2d(28, stride=1)
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, BatchNorm):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

		# for m in self.modules():
		#     if isinstance(m, nn.Conv2d):
		#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
		#     elif isinstance(m, nn.BatchNorm2d):
		#         nn.init.constant_(m.weight, 1)
		#         nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes,
								dilation=(dilation, dilation)))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


def drn_a_50(pretrained=False, **kwargs):
	model = DRN_A(Bottleneck, [3, 4, 6, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
	return model


def drn_c_26(pretrained=False, **kwargs):
	model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-c-26']))
	return model


def drn_c_42(pretrained=False, **kwargs):
	model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-c-42']))
	return model


def drn_c_58(pretrained=False, **kwargs):
	model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-c-58']))
	return model


def drn_d_22(pretrained=False, **kwargs):
	model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='D', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-d-22']))
	return model


def drn_d_24(pretrained=False, **kwargs):
	model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 2, 2], arch='D', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-d-24']))
	return model


def drn_d_38(pretrained=False, **kwargs):
	model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-d-38']))
		print("Loading pretrained model on ImageNet")
	return model


def drn_d_40(pretrained=False, **kwargs):
	model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-d-40']))
	return model


def drn_d_54(pretrained=False, **kwargs):
	model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-d-54']))
	return model


def drn_d_56(pretrained=False, **kwargs):
	model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-d-56']))
	return model


def drn_d_105(pretrained=False, **kwargs):
	model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 1, 1], arch='D', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-d-105']))
	return model


def drn_d_107(pretrained=False, **kwargs):
	model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 2, 2], arch='D', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-d-107']))
	return model


def fill_up_weights(up):
	w = up.weight.data
	f = math.ceil(w.size(2) / 2)
	c = (2 * f - 1 - f % 2) / (2. * f)
	for i in range(w.size(2)):
		for j in range(w.size(3)):
			w[0, 0, i, j] = \
				(1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
	for c in range(1, w.size(0)):
		w[c, 0, :, :] = w[0, 0, :, :]


class Net(nn.Module):

	def __init__(self, classes, embed_dim, resnet, pretrained_model=None,
				 pretrained=True, use_torch_up=False):
		super().__init__()
		assert(isinstance(classes , dict)), f"num_labels should be dict, got {type(classes)}"
		self.datasets = list(classes.keys())
		self.embed_dim = embed_dim

		resnet_archs = {'resnet_18':drn_d_22 , 'resnet_34':drn_d_38
                        , 'resnet_50':drn_d_54 , 'resnet_101':drn_d_105}
		arch = resnet_archs[resnet]
		
		# model = model_name(pretrained=pretsrained, num_classes=1000)
		model = arch(pretrained=pretrained, num_classes=1000)
		pmodel = nn.DataParallel(model)
		if pretrained_model is not None:
			pmodel.load_state_dict(pretrained_model)

		self.base = nn.Sequential(*list(model.children())[:-2]) ## Encoder. 
		self.seg = nn.ModuleList() ## Decoder 1d conv
		self.up = nn.ModuleList() ## Decoder upsample (non-trainable)

		for n_labels in classes.values():
			m = nn.Conv2d(model.out_dim, n_labels, kernel_size=1, bias=True)
			n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			m.weight.data.normal_(0, math.sqrt(2. / n))
			m.bias.data.zero_()
			self.seg.append(m)

			if use_torch_up:
				self.up.append(nn.UpsamplingBilinear2d(scale_factor=8))
			else:
				up = nn.ConvTranspose2d(n_labels, n_labels, 16, stride=8, padding=4,
										output_padding=0, groups=n_labels,
										bias=False)
				fill_up_weights(up)
				up.weight.requires_grad = False
				self.up.append(up)

		## Encoder output module
		m = nn.Conv2d(model.out_dim , self.embed_dim , kernel_size=1, bias=True)
		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
		m.weight.data.normal_(0, math.sqrt(2. / n))
		m.bias.data.zero_()
		self.en_map = m
		self.en_up = nn.ConvTranspose2d(self.embed_dim , self.embed_dim , 16, stride=8, padding=4
													,output_padding=0,groups=self.embed_dim, bias=False)
		
		fill_up_weights(self.en_up)
		self.en_up.weight.requires_grad = False

	def forward(self, x, enc=True, finetune=False):

		y_encoder = self.base(x)

		if finetune:
			y_encoder = y_encoder.detach()
		
		output_dict = {key:None for key in self.datasets}
		for seg_layer , up_layer , key in zip(self.seg , self.up , self.datasets):
			y = seg_layer(y_encoder)
			y = up_layer(y)
			output_dict[key] = y

		if enc:
			y_encoder = self.en_map(y_encoder)
			y_encoder = self.en_up(y_encoder)
			return output_dict , y_encoder
		else:
			return output_dict


	def optim_parameters(self, memo=None):
		for param in self.base.parameters():
			yield param
		for param in self.seg.parameters():
			yield param
		for param in self.en_map.parameters():
			yield param
