"""Utilities related to model visualization."""
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict

try:
	# pydot-ng is a fork of pydot that is better maintained.
	import pydot_ng as pydot
except ImportError:
# pydotplus is an improved version of pydot
	try:
		import pydotplus as pydot
	except ImportError:
# Fall back on pydot if necessary.
		try:
			import pydot
		except ImportError:
			pydot = None


def _check_pydot():
	try:
# Attempt to create an image of a blank graph
#  to check the pydot/graphviz installation.
		pydot.Dot.create(pydot.Dot())
	except Exception:
# pydot raises a generic Exception here,
# so no specific class can be caught.
		raise ImportError('Failed to import pydot. You must install pydot '
		                  'and graphviz for `pydotprint` to work.')





def model_to_dot(model, input_shape,show_shapes=False,
                 show_layer_names=True,rankdir='TB'):
	def register_hook(module):
		def hook(module, input, output):
			layer=[]
			class_name = str(module.__class__).split('.')[-1].split("'")[0]
			module_idx = len(summary)
			if hasattr(module, 'weight'):
				m_key = '%i-%s(%s)' % (module_idx + 1, class_name, module.weight.data.numpy().shape)
			else:
				m_key = '%i-%s' % (module_idx + 1, class_name)
			layer.append(str(m_key))
			layer.append(list(input[0].size()))
			if isinstance(output, (list, tuple)):
				layer.append([[-1] + list(o.size())[1:] for o in output])
			else:
				layer.append(list(output.size()))
			
			params = 0
			if hasattr(module, 'weight'):
				params += torch.prod(torch.LongTensor(list(module.weight.size())))
			
			if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
				params += torch.prod(torch.LongTensor(list(module.bias.size())))
			layer.append(params)
			summary.append(layer)
		if (not isinstance(module, nn.Sequential) and
			    not isinstance(module, nn.ModuleList) and
			    not (module == model)):
			hooks.append(module.register_forward_hook(hook))
	
	dtype = torch.FloatTensor
	# check if there are multiple inputs to the network
	if isinstance(input_shape[0], (list, tuple)):
		x = [Variable(torch.rand(*in_size)).type(dtype) for in_size in input_shape]
	else:
		x = Variable(torch.rand(*input_shape)).type(dtype)
	
	# print(type(x[0]))
	# create properties
	summary = []
	hooks = []
	# register hook
	model.apply(register_hook)
	# make a forward pass
	# print(x.shape)
	model(x)
	# print(summary)
	# remove these hooks
	for h in hooks:
		h.remove()
	
	_check_pydot()
	dot = pydot.Dot()
	dot.set('rankdir', rankdir)
	dot.set('concentrate', True)
	dot.set_node_defaults(shape='record')
	layers=[]
	for layer in range(len(summary)):
		layer_list=[]
		# Create node's label.
		if show_layer_names:
			label = '{}'.format(summary[layer][0])
		else:
			label = str(summary[layer][0])
		
		# Rebuild the label as a table including input/output shapes.
		if show_shapes:
			outputlabels = str(summary[layer][2])
			inputlabels = str(summary[layer][1])
			label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label,
			                                               inputlabels,
			                                               outputlabels)
		node = pydot.Node(str(summary[layer][0]), label=label)
		dot.add_node(node)
		layers.append(summary[layer][0])
		# layers.append(layer_list)
	for layer in range(len(layers)):
		if layer>=1:
			dot.add_edge(pydot.Edge(str(layers[layer-1]), str(layers[layer])))

	return dot




def plot_model(model,input_size,
               to_file='model.png',
               show_shapes=False,
               show_layer_names=True,
               rankdir='TB'):
	"""Converts a Keras model to dot format and save to a file.

	# Arguments
	    model: A Keras model instance
	    to_file: File name of the plot image.
	    show_shapes: whether to display shape information.
	    show_layer_names: whether to display layer names.
	    rankdir: `rankdir` argument passed to PyDot,
	        a string specifying the format of the plot:
	        'TB' creates a vertical plot;
	        'LR' creates a horizontal plot.
	"""
	dot = model_to_dot(model,input_size, show_shapes, show_layer_names, rankdir)
	_, extension = os.path.splitext(to_file)
	if not extension:
		extension = 'png'
	else:
		extension = extension[1:]
	dot.write(to_file, format=extension)