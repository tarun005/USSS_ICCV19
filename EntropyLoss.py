import torch
import torch.nn.functional as F
import pdb , sys
from collections import OrderedDict 

class EmbeddingLoss(torch.nn.Module):

	def __init__(self , n_labels , embed_dim, label_embedding, loss_criterion):
		super().__init__()

		assert(isinstance(n_labels , dict)), f"num_labels should be dict, got {type(classes)}"

		self.n_labels = n_labels
		self.datasets = list(self.n_labels.keys())
		self.embed_dim = embed_dim
		self.t = 1
		self.dnum = {key:i for i,key in enumerate(self.datasets)}
		self.label_embedding = label_embedding
		self.loss_criterion = loss_criterion
		
		self.conv_module_dict = torch.nn.Sequential(OrderedDict([
			(self.datasets[0], torch.nn.Conv2d(embed_dim , label_embedding[self.datasets[0]].size(0) , kernel_size=1, bias=False)), 
			(self.datasets[1], torch.nn.Conv2d(embed_dim , label_embedding[self.datasets[1]].size(0) , kernel_size=1, bias=False))
		]))

		self._fill_conv_weights()

	def _fill_conv_weights(self):

		for d in self.datasets:
			e_w = F.normalize(self.label_embedding[d] , p=2 , dim=1)
			# layer = self.conv_module_dict[d]
			layer = self.conv_module_dict._modules[d]
			layer.weight.data.copy_(e_w.unsqueeze(-1).unsqueeze(-1))
			layer.weight.requires_grad = False

	def forward(self , encoder_op , d_set, alpha=0, beta=0, targets=None, delta=0):

		self.d = d_set

		if targets is None:
			if len(self.datasets) == 1:
				return self._self_entropy(encoder_op, beta)
			else:
				return self._similarity(encoder_op, alpha, beta)

		centroids, nlabels = self._update_centroids(encoder_op, targets)

		return centroids , nlabels

	def _update_centroids(self, encoder_op, targets):

		embeds = {key:torch.zeros(1, self.n_labels[key], self.embed_dim).cuda() for key in self.datasets}
		nlabels = {key:torch.zeros(1 , self.n_labels[key] ,1, dtype=torch.long).cuda() for key in self.datasets}

		for key in self.datasets:
			
			if self.dnum[key] in self.d_id:

				encoder_partial = torch.index_select(encoder_op , 0 , self.dataset_index[key])
				labels = torch.index_select(targets , 0 , self.dataset_index[key])
				labels_onehot = torch.zeros_like(labels).cuda()
				labels_onehot = labels_onehot.repeat(1,self.n_labels[key],1,1).scatter_(1,labels,1)
				nlabels[key] = labels_onehot.sum(-1).sum(-1).sum(0).view(1,-1,1).data

				# ########## ---- Memory Intensive ------######
				# centroids = (labels_onehot.float().unsqueeze(2) * encoder_partial.unsqueeze(1)).data
				# embeds[key] = centroids.sum(0 , keepdim=True).sum(3 , keepdim=True).sum(4 , keepdim=True).squeeze(-1).squeeze(-1) # 1 x C x E
				# ########## ---- Memory Intensive ------######

				########## ----Time Intensive ------######
				centroids = []
				for l_mapid in range(self.n_labels[key]):
					label_map = labels_onehot[:,l_mapid,:,:].unsqueeze(1).float() ## N x 1 x H x W
					centroid_map = label_map * encoder_partial ## N x E x H x W
					centroid_map = centroid_map.sum(0 , keepdim=True).sum(2,keepdim=True).sum(3,keepdim=True).view(1,-1) # 1 x E
					centroids.append(centroid_map)

				embeds[key] = torch.cat(centroids , dim=0)
				embeds[key] = embeds[key].unsqueeze(0).data
				########## ----Time Intensive ------######

		return embeds, nlabels			

	def _similarity(self, encoder_op, alpha, beta):

		within_conv_layer = self.conv_module_dict._modules[self.d]
		other_d = [d for d in self.datasets if d != self.d][0]
		
		cross_conv_layer = self.conv_module_dict._modules[other_d]

		within_domain_map = within_conv_layer(encoder_op)
		within_domain_map = within_domain_map[:,:-1] ## Ignore the last "None" class
		tensor_prob = F.softmax(within_domain_map, dim=1) ## Temperature = 1
		log_prob = F.log_softmax(within_domain_map, dim=1)
		loss_within = torch.mean(-1* torch.sum(log_prob * tensor_prob , dim=1)).view(-1)

		cross_domain_map = cross_conv_layer(encoder_op)
		cross_domain_map = cross_domain_map[:,:-1]
		tensor_prob = F.softmax(cross_domain_map/self.t, dim=1)
		log_prob = F.log_softmax(cross_domain_map/self.t, dim=1)
		loss_cross = torch.mean(-1* torch.sum(log_prob * tensor_prob , dim=1)).view(-1)

		loss_unsup = alpha*loss_cross + beta*loss_within
		return loss_unsup

