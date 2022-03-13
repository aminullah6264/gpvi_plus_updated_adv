# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Networks with input parameters."""

from builtins import print
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import alf
from alf.initializers import variance_scaling_init
from alf.layers import ParamFC, ParamConv2D
from alf.networks.network import Network
from alf.tensor_specs import TensorSpec
from alf.utils import common


@alf.configurable
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation=torch.relu_,
                 use_bias=False,
                 use_norm=None,
                 n_groups=None,
                 kernel_initializer=None):
        super(BasicBlock, self).__init__()


        if use_bias is None:
            use_bias = use_norm is None
        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                mode='fan_in',
                distribution='truncated_normal',
                nonlinearity=activation)

        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self._param_length = None

        


        self.conv1 = ParamConv2D(self.in_planes, self.planes, 3, activation=activation, strides=self.stride,
                     padding=1, use_bias=use_bias, use_norm=use_norm,
                     n_groups=n_groups, kernel_initializer=kernel_initializer)
        # self.conv1 = ParamConv2D(self.in_planes, self.planes, 1, activation=activation, strides=self.stride,
        #              padding=0, use_bias=use_bias, use_norm=use_norm,
        #              n_groups=n_groups, kernel_initializer=kernel_initializer)

        self.conv2 = ParamConv2D(self.planes, self.planes, 3, activation=activation, strides= 1,
                     padding=1, use_bias=use_bias, use_norm=use_norm,
                     n_groups=n_groups, kernel_initializer=kernel_initializer)

        

        if self.stride != 1 or self.in_planes != self.planes:
            self.shortcut = ParamConv2D(self.in_planes, self.expansion* self.planes, 1, activation=activation, strides=self.stride,
                            padding= 0, use_bias=use_bias, use_norm=use_norm,
                            n_groups=n_groups, kernel_initializer=kernel_initializer)

  
    
    @property
    def param_length(self):
        """Get total number of parameters for all layers. """
        if self._param_length is None:
            length = 0

            length = length + self.conv1.param_length
            length = length + self.conv2.param_length

            if self.stride != 1 or self.in_planes != self.planes:
                length = length + self.shortcut.param_length

            self._param_length = length
        return self._param_length

    def set_parameters(self, theta, reinitialize=False):
        """Distribute parameters to corresponding layers.

        Args:
            theta (torch.Tensor): with shape ``[D] (groups=1)``
                                        or ``[B, D] (groups=B)``
                where the meaning of the symbols are:
                - ``B``: batch size
                - ``D``: length of parameters, should be self.param_length
                When the shape of inputs is ``[D]``, it will be unsqueezed
                to ``[1, D]``.
            reinitialize (bool): whether to reinitialize parameters of
                each layer.
        """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        assert (theta.ndim == 2 and theta.shape[1] == self.param_length), (
            "Input theta has wrong shape %s. Expecting shape (, %d)" %
            self.param_length)
        pos = 0
        param_length = self.conv1.param_length
        self.conv1.set_parameters(
                theta[:, pos:pos + param_length], reinitialize=reinitialize)
        pos = pos + param_length

        param_length = self.conv2.param_length
        self.conv2.set_parameters(
                theta[:, pos:pos + param_length], reinitialize=reinitialize)
        pos = pos + param_length

        if self.stride != 1 or self.in_planes != self.planes:
            param_length = self.shortcut.param_length
            self.shortcut.set_parameters(
                    theta[:, pos:pos + param_length], reinitialize=reinitialize)
            pos = pos + param_length

        self._output_spec = None


    def forward(self, x):
        out = self.conv1(x, keep_group_dim=False)
        out = self.conv2(out, keep_group_dim=False)

        if self.stride != 1 or self.in_planes != self.planes:
            out = out.clone() + self.shortcut(x, keep_group_dim=False)
            out = torch.relu_(out)

        return out

@alf.configurable
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, activation=torch.relu_,
                 use_bias=False,
                 use_norm=None,
                 n_groups=None,
                 kernel_initializer=None,
                 flatten_output= False):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.activation= activation
        self.use_bias= use_bias
        self.use_norm= use_norm
        self.n_groups= n_groups
        self.kernel_initializer= kernel_initializer
        self.flatten_output= flatten_output
        self._param_length = None

        if self.use_bias is None:
            self.use_bias = self.use_norm is None
        if self.kernel_initializer is None:
            self.kernel_initializer = functools.partial(
                variance_scaling_init,
                mode='fan_in',
                distribution='truncated_normal',
                nonlinearity=activation)

        self.conv1 = ParamConv2D(3, self.in_planes, 3, activation=self.activation, strides=1,
                     padding=1, use_bias=self.use_bias, use_norm= self.use_norm,
                     n_groups=self.n_groups, kernel_initializer=self.kernel_initializer)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, activation=self.activation,
                 use_bias=self.use_bias,
                 use_norm=self.use_norm,
                 n_groups=self.n_groups,
                 kernel_initializer=self.kernel_initializer))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)


    @property
    def param_length(self):
        """Get total number of parameters for all layers. """
        if self._param_length is None:
            length = 0

            length = length + self.conv1.param_length

            for conv_l in self.layer1:
                length = length + conv_l.param_length

            for conv_l in self.layer2:
                length = length + conv_l.param_length

            for conv_l in self.layer3:
                length = length + conv_l.param_length

            self._param_length = length
        return self._param_length


    
    def set_parameters(self, theta, reinitialize=False):
        """Distribute parameters to corresponding layers.

        Args:
            theta (torch.Tensor): with shape ``[D] (groups=1)``
                                        or ``[B, D] (groups=B)``
                where the meaning of the symbols are:
                - ``B``: batch size
                - ``D``: length of parameters, should be self.param_length
                When the shape of inputs is ``[D]``, it will be unsqueezed
                to ``[1, D]``.
            reinitialize (bool): whether to reinitialize parameters of
                each layer.
        """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        assert (theta.ndim == 2 and theta.shape[1] == self.param_length), (
            "Input theta has wrong shape %s. Expecting shape (, %d)" %
            self.param_length)
        pos = 0
        param_length = self.conv1.param_length
        self.conv1.set_parameters(
                theta[:, pos:pos + param_length], reinitialize=reinitialize)
        pos = pos + param_length

        for conv_l in self.layer1:
            param_length = conv_l.param_length
            conv_l.set_parameters(
                theta[:, pos:pos + param_length], reinitialize=reinitialize)
            pos = pos + param_length

        for conv_l in self.layer2:
            param_length = conv_l.param_length
            conv_l.set_parameters(
                theta[:, pos:pos + param_length], reinitialize=reinitialize)
            pos = pos + param_length
        
        for conv_l in self.layer3:
            param_length = conv_l.param_length
            conv_l.set_parameters(
                theta[:, pos:pos + param_length], reinitialize=reinitialize)
            pos = pos + param_length

        self._output_spec = None

    def forward(self, x, state=()):
        out = self.conv1(x, keep_group_dim=False)

        # print(out.shape)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)


        out = F.avg_pool2d(out, out.size()[3])
        if self.flatten_output:
            out = out.view(out.shape[0], -1, self.in_planes)
        
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out, state




@alf.configurable
class ParamNetwork(Network):
    def __init__(self,
                 input_tensor_spec,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 use_conv_bias=False,
                 use_conv_norm=None,
                 use_fc_bias=True,
                 use_fc_norm=None,
                 n_groups=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 last_layer_size=None,
                 last_activation=None,
                 last_use_bias=True,
                 last_use_norm=None,
                 name="ParamNetwork"):
        """A network with Fc and conv2D layers that does not maintain its own
        network parameters, but accepts them from users. If the given parameter
        tensor has an extra batch dimension (first dimension), it performs
        parallel operations.

        Args:
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then ``preprocessing_combiner`` must not be
                None.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format
                ``(filters, kernel_size, strides, padding, pooling_kernel)``,
                where ``padding`` and ``pooling_kernel`` are optional.
            fc_layer_params (tuple[int]): a tuple of integers
                representing FC layer sizes.
            use_conv_bias (bool|None): whether use bias for conv layers. If None, 
                will use conv_bias if ``use_norm`` is None.
            use_conv_norm (str): which normalization to apply to conv layers, options 
                are [``bn`, ``ln``]. Default: None, no normalization applied.
            use_fc_bias (bool): whether use bias for fc layers.
            use_fc_norm (str): which normalization to apply to fc layers, options 
                are [``bn`, ``ln``]. Default: None, no normalization applied.
            n_groups (int): number of parallel groups, must be specified if ``use_bn``
            activation (torch.nn.functional): activation for all the layers
            kernel_initializer (Callable): initializer for all the layers.
            last_layer_size (int): an optional size of an additional layer
                appended at the very end. Note that if ``last_activation`` is
                specified, ``last_layer_size`` has to be specified explicitly.
            last_activation (nn.functional): activation function of the
                additional layer specified by ``last_layer_param``. Note that if
                ``last_layer_param`` is not None, ``last_activation`` has to be
                specified explicitly.
            last_use_bias (bool): whether use bias for the additional layer.
            last_use_norm (str): which normalization to apply to the additional layer, 
                options are [``bn`, ``ln``]. Default: None, no normalization applied.
            name (str):
        """

        super().__init__(input_tensor_spec=input_tensor_spec, name=name)

        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                mode='fan_in',
                distribution='truncated_normal',
                nonlinearity=activation)

        self._param_length = None
        self._conv_net = None

        block_size = 3         # block_size = {3, 5, 7, 9}, leading to 20, 32, 44, and 56-layer networks. 
        self._conv_net = ResNet(BasicBlock, [block_size, block_size, block_size],
            activation=activation,
            use_bias=use_conv_bias,
            use_norm=use_conv_norm,
            n_groups=n_groups,
            kernel_initializer=kernel_initializer,
            flatten_output=True)
        input_size = self._conv_net.in_planes
        
        # if conv_layer_params:
        #     assert isinstance(conv_layer_params, tuple), \
        #         "The input params {} should be tuple".format(conv_layer_params)
        #     assert input_tensor_spec.ndim == 3, \
        #         "The input shape {} should be like (C,H,W)!".format(
        #             input_tensor_spec.shape)
        #     input_channels, height, width = input_tensor_spec.shape

            
            

            # self._conv_net = ParamConvNet(
            #     input_channels, (height, width),
            #     conv_layer_params,
            #     activation=activation,
            #     use_bias=use_conv_bias,
            #     use_norm=use_conv_norm,
            #     n_groups=n_groups,
            #     kernel_initializer=kernel_initializer,
            #     flatten_output=True)
            # input_size = self._conv_net.output_spec.shape[-1]
        # else:
        #     assert input_tensor_spec.ndim == 1, \
        #         "The input shape {} should be like (N,)!".format(
        #             input_tensor_spec.shape)
        #     input_size = input_tensor_spec.shape[0]

        self._fc_layers = nn.ModuleList()
        if fc_layer_params is None:
            fc_layer_params = []
        else:
            assert isinstance(fc_layer_params, tuple)
            fc_layer_params = list(fc_layer_params)

        for size in fc_layer_params:
            self._fc_layers.append(
                ParamFC(
                    input_size,
                    size,
                    activation=activation,
                    use_bias=use_fc_bias,
                    use_norm=use_fc_norm,
                    n_groups=n_groups,
                    kernel_initializer=kernel_initializer))
            input_size = size

        if last_layer_size is not None or last_activation is not None:
            assert last_layer_size is not None and last_activation is not None, \
            "Both last_layer_param and last_activation need to be specified!"
            self._fc_layers.append(
                ParamFC(
                    input_size,
                    last_layer_size,
                    activation=last_activation,
                    use_bias=last_use_bias,
                    use_norm=last_use_norm,
                    n_groups=n_groups,
                    kernel_initializer=kernel_initializer))
            input_size = last_layer_size

        self._output_spec = TensorSpec((input_size, ),
                                       dtype=self._input_tensor_spec.dtype)

    @property
    def param_length(self):
        """Get total number of parameters for all layers. """
        if self._param_length is None:
            length = 0
            if self._conv_net is not None:
                length += self._conv_net.param_length
            for fc_l in self._fc_layers:
                length = length + fc_l.param_length
            self._param_length = length
        return self._param_length

    def set_parameters(self, theta, reinitialize=False):
        """Distribute parameters to corresponding layers.

        Args:
            theta (torch.Tensor): with shape ``[D] (groups=1)``
                                        or ``[B, D] (groups=B)``
                where the meaning of the symbols are:
                - ``B``: batch size
                - ``D``: length of parameters, should be self.param_length
                When the shape of inputs is ``[D]``, it will be unsqueezed
                to ``[1, D]``.
            reinitialize (bool): whether to reinitialize parameters of
                each layer.
        """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        assert (theta.ndim == 2 and theta.shape[1] == self.param_length), (
            "Input theta has wrong shape %s. Expecting shape (, %d)" %
            self.param_length)
        if self._conv_net is not None:
            split = self._conv_net.param_length
            conv_theta = theta[:, :split]
            self._conv_net.set_parameters(
                conv_theta, reinitialize=reinitialize)
            fc_theta = theta[:, self._conv_net.param_length:]
        else:
            fc_theta = theta

        pos = 0
        for fc_l in self._fc_layers:
            param_length = fc_l.param_length
            fc_l.set_parameters(
                fc_theta[:, pos:pos + param_length], reinitialize=reinitialize)
            pos = pos + param_length

    def forward(self, inputs, state=()):
        """
        Args:
            inputs (Tensor):
            state: not used, just keeps the interface same with other networks.
        """
        x = inputs
        if self._conv_net is not None:
            x, state = self._conv_net(x, state=state)
        
        # import ipdb; ipdb.set_trace()
        for fc_l in self._fc_layers:
            x = fc_l(x)
        return x, state
