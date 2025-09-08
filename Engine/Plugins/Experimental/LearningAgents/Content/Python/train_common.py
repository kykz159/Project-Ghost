# -*- coding: utf-8 -*-
'''
Copyright Epic Games, Inc. All Rights Reserved.
'''

import struct
import time
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

import logging
logger = logging.getLogger("LearningAgents")

UE_LEARNING_DEVELOPMENT = False
UE_LEARNING_PROFILE = True


def check_array(check_name, arr):
    """ Checks an array for nans/infs """
    
    if not UE_LEARNING_DEVELOPMENT: return
    
    with torch.no_grad():
        arr_numpy = arr.detach().cpu().numpy()

        if np.any(~np.isfinite(arr_numpy)):

            logger.info("%s %f %f %f %f %i %i %i %i %i %s" % (
                check_name,
                arr_numpy.min(), 
                arr_numpy.max(), 
                arr_numpy.mean(), 
                arr_numpy.std(), 
                np.any(~np.isfinite(arr_numpy)),
                np.any(np.isinf(arr_numpy)),
                np.any(np.isposinf(arr_numpy)),
                np.any(np.isneginf(arr_numpy)),
                np.any(np.isnan(arr_numpy)),
                str(arr_numpy)))
                
            logging.shutdown()
            
            raise Exception()

def check_network(check_name, network):
    """ Checks a network for nans/infs """
    
    with torch.no_grad():
        for name, param in network.named_parameters():
            check_array(check_name + '_' + name, param)

def check_network_gradients(check_name, network):
    """ Checks a network's gradients for nans/infs """
    
    with torch.no_grad():
        for name, param in network.named_parameters():
            if param.grad is not None:
                check_array(check_name + '_' + name + '_grad', param.grad)

def check_scalar(check_name, value):
    """ Checks a scalar for nans/infs """
    
    with torch.no_grad():
        value_item = value.item()
        if not np.isfinite(value_item):
            logger.info(check_name, value_item)

# Profile

class Profile:

    def __init__(self, name):
        self.name = name
        self.enabled = logger.isEnabledFor(logging.INFO) and UE_LEARNING_PROFILE

    def __enter__(self):
        if self.enabled:
            self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        if self.enabled:
            gib_conversion = 1024 ** 3
            allocated = torch.cuda.memory_allocated() / gib_conversion
            reserved = torch.cuda.memory_reserved() / gib_conversion
            elapsed_ms = round((time.time() - self.start_time) * 1000)
            message = (
                f"Profile| {self.name:<25s} {elapsed_ms:6d}ms   "
                f"GPU Usage| Allocated: {allocated:.2f} GiB, Reserved: {reserved:.2f} GiB"
            )
            logger.info(message)

# Network Snapshots

# Magic Number of Neural Network Snapshots. Should match what is given in LearningNeuralNetwork.cpp
UE_LEARNING_NEURAL_NETWORK_MAGIC_NUMBER = 0x1e9b0c80

# Version Number of Neural Network Snapshots. Should match what is given in LearningNeuralNetwork.cpp
UE_LEARNING_NEURAL_NETWORK_VERSION_NUMBER = 1

# Functions for loading and saving snapshot from binary data into a python neural network object

def load_snapshot(data, network):

    magic = struct.unpack("I", data[0:4].tobytes())[0]
    assert magic == UE_LEARNING_NEURAL_NETWORK_MAGIC_NUMBER
    version = struct.unpack("I", data[4:8].tobytes())[0]
    assert version == UE_LEARNING_NEURAL_NETWORK_VERSION_NUMBER
    input_num = struct.unpack("I", data[8:12].tobytes())[0]
    output_num = struct.unpack("I", data[12:16].tobytes())[0]
    compatibility_hash = struct.unpack("I", data[16:20].tobytes())[0]
    network_filedata_size = struct.unpack("I", data[20:24].tobytes())[0]

    # For convenience we just attach these additional properties to the network object
    network.load_from_filedata(data[24:])
    network.input_num = input_num
    network.output_num = output_num
    network.compatibility_hash = compatibility_hash
    
def save_snapshot(data, network):
    data[ 0: 4] = np.frombuffer(struct.pack("I", UE_LEARNING_NEURAL_NETWORK_MAGIC_NUMBER), np.uint8)
    data[ 4: 8] = np.frombuffer(struct.pack("I", UE_LEARNING_NEURAL_NETWORK_VERSION_NUMBER), np.uint8)
    data[ 8:12] = np.frombuffer(struct.pack("I", network.input_num), np.uint8)
    data[12:16] = np.frombuffer(struct.pack("I", network.output_num), np.uint8)
    data[16:20] = np.frombuffer(struct.pack("I", network.compatibility_hash), np.uint8)
    data[20:24] = np.frombuffer(struct.pack("I", network.get_filedata_size()), np.uint8)
    network.save_to_filedata(data[24:])

def get_snapshot_byte_num(network):
    return (4 + 4 + 4 + 4 + 4 + 4 + network.get_filedata_size())

def save_snapshot_to_file(network, filename):
    with open(filename, 'wb') as f:
        data = np.zeros([get_snapshot_byte_num(network)], dtype=np.uint8)
        save_snapshot(data, network)
        f.write(data.tobytes())
    

# Completion

UE_COMPLETION_RUNNING = 0
UE_COMPLETION_TRUNCATED = 1
UE_COMPLETION_TERMINATED = 2


# Trainer Response

UE_RESPONSE_SUCCESS = 0
UE_RESPONSE_UNEXPECTED = 1
UE_RESPONSE_COMPLETED = 2
UE_RESPONSE_STOPPED = 3
UE_RESPONSE_TIMEOUT = 4


def create_task_dir(base_dir, task_name, task_id_override=None):
    # Use task_id_override to keep the snapshot dir in sync with the config dir in case the user has a
    # folder in one place and not the other we won't let them get different ids

    # If the end user puts in any other weird symbols besides slashes in the base_dir or task_name, then that's their problem for now.
    task_name = task_name.strip("/").strip("\\")

    if task_id_override:
        candidate_path = os.path.join(base_dir, f"{task_name}{task_id_override}")
        if not os.path.exists(candidate_path):
            os.makedirs(candidate_path)
            return (candidate_path, task_id_override)
        elif not os.listdir(candidate_path):
            # Path is already made but it's empty so we are good
            return (candidate_path, task_id_override)
        else:
            # Someone already made the dir and left other files in it so error out. Deleting files seems too risky.
            raise Exception(f"Can't make dir with {candidate_path} because it already exists and is not empty.")
    else:
        # Make dir by auto-incrementing until we find a fresh folder name
        task_id = 0
        while True:
            candidate_path = os.path.join(base_dir, f"{task_name}{task_id}") 

            if not os.path.exists(candidate_path):
                os.makedirs(candidate_path)
                return (candidate_path, task_id)

            task_id += 1


# Functions for operating on action schemas

def _mask_to_indices_exclusive(m, device):
    indices = [[] for i in range(m.shape[1])]
    for i in range(m.shape[0]):
        indices[np.where(m[i] == 1)[0][0]].append(i)
    return [torch.as_tensor(i, dtype=torch.long, device=device) for i in indices]


def _mask_to_indices_inclusive(m, device):
    indices = [[] for i in range(m.shape[1])]
    for i in range(m.shape[0]):
        for j in np.where(m[i] == 1)[0]:
            indices[j].append(i)
    return [torch.as_tensor(i, dtype=torch.long, device=device) for i in indices]
    
    
def schema_act_num(act_schema):
    
    act_type = act_schema['Type']
    
    if act_type == 'Null':
        return 0
    
    if act_type == 'Continuous':
        return act_schema['VectorSize']
    
    elif act_type == 'DiscreteExclusive':
        return act_schema['VectorSize']
    
    elif act_type == 'DiscreteInclusive':
        return act_schema['VectorSize']
        
    elif act_type == 'NamedDiscreteExclusive':
        return act_schema['VectorSize']
    
    elif act_type == 'NamedDiscreteInclusive':
        return act_schema['VectorSize']
        
    elif act_type == 'And':
        return sum([schema_act_num(element) for element in act_schema['Elements'].values()])
    
    elif act_type in ('OrExclusive', 'OrInclusive'):
        return len(act_schema['Elements']) + sum([schema_act_num(element) for element in act_schema['Elements'].values()])
    
    elif act_type == 'Array':
        return act_schema['Num'] * schema_act_num(act_schema['Element'])
        
    elif act_type == 'Encoding':
        return schema_act_num(act_schema['Element'])
        
    else:
        raise Exception('Not Implemented')


half_plus_half_log_2pi = 0.5 + 0.5 * np.log(2 * np.pi)
log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

# This works by making very small logits when masked is one and 
# having very little zero effect on the logits when masked is zero
# it has stable gradients compared to using negative inf to mask values
def log_softmax_masked(logits, masked, dim=-1):
    return torch.log_softmax(logits + torch.log((1.0 - masked) + 1e-45), dim=dim)

def independent_normal_entropy(log_std, masked):
    return (1.0 - masked) * (half_plus_half_log_2pi + log_std)

def multinoulli_entropy(logits, masked):
    log_probs = log_softmax_masked(logits, masked, dim=-1)
    probs = torch.exp(torch.clip(log_probs, None, 10))
    return torch.where(masked == 1.0, 0.0, -(probs * log_probs))
    
def bernoulli_entropy(logits, masked):
    neg_prob = torch.sigmoid(-logits)
    pos_prob = torch.sigmoid(+logits)
    neg_log_prob = -F.softplus(+logits)
    pos_log_prob = -F.softplus(-logits)
    return (1.0 - masked) * -(neg_log_prob * neg_prob + pos_log_prob * pos_prob)

def schema_entropy(act_schema, act_dist, act_mod):
    
    act_type = act_schema['Type']
    assert act_schema['DistributionSize'] == act_dist.shape[1]
    assert act_schema['ModifierSize'] == act_mod.shape[1]
    
    if act_type == 'Null':
        return torch.zeros([len(act_dist)], device=act_dist.device)
    
    if act_type == 'Continuous':
        
        assert act_schema['VectorSize'] == act_dist.shape[1]//2
        
        _, act_log_std = act_dist[:,:act_dist.shape[1]//2], act_dist[:,act_dist.shape[1]//2:]
        act_mod_masked = act_mod[:,1:1+act_dist.shape[1]//2]
        
        return independent_normal_entropy(act_log_std, act_mod_masked).sum(dim=-1)
    
    elif act_type == 'DiscreteExclusive':
        
        return multinoulli_entropy(act_dist, act_mod[:,1:]).sum(dim=-1)
    
    elif act_type == 'DiscreteInclusive':
        
        return bernoulli_entropy(act_dist, act_mod[:,1:]).sum(dim=-1)
        
    elif act_type == 'NamedDiscreteExclusive':
        
        return multinoulli_entropy(act_dist, act_mod[:,1:]).sum(dim=-1)
    
    elif act_type == 'NamedDiscreteInclusive':
        
        return bernoulli_entropy(act_dist, act_mod[:,1:]).sum(dim=-1)
        
    elif act_type == 'And':
        
        total = torch.zeros_like(act_dist[:,0])
        
        offset_dist = 0
        offset_mod = 1
        
        for ei, element in enumerate(act_schema['Elements'].values()):
            
            assert ei == element['Index']
            size_dist = element['DistributionSize']
            size_mod = element['ModifierSize']
            
            total += schema_entropy(
                element, 
                act_dist[:,offset_dist:offset_dist+size_dist],
                act_mod[:,offset_mod:offset_mod+size_mod])
            
            offset_dist += size_dist
            offset_mod += size_mod
        
        assert offset_dist == act_schema['DistributionSize']
        assert offset_mod == act_schema['ModifierSize']
        
        return total
    
    elif act_type == 'OrExclusive':

        elem_num = len(act_schema['Elements'])
        elem_entropy = multinoulli_entropy(act_dist[:,-elem_num:], act_mod[:,1:1+elem_num])

        total = torch.zeros_like(act_dist[:,0])
        
        offset_dist = 0
        offset_mod = 1 + elem_num
        
        for ei, element in enumerate(act_schema['Elements'].values()):
            
            assert ei == element['Index']
            size_dist = element['DistributionSize']
            size_mod = element['ModifierSize']
            
            total += (1.0 - act_mod[:,1+ei]) * (elem_entropy[:,ei] + schema_entropy(
                element, 
                act_dist[:,offset_dist:offset_dist+size_dist],
                act_mod[:,offset_mod:offset_mod+size_mod]))
            
            offset_dist += size_dist
            offset_mod += size_mod
        
        assert offset_dist + elem_num == act_schema['DistributionSize']
        assert offset_mod == act_schema['ModifierSize']
        
        return total
        
    elif act_type == 'OrInclusive':
        
        elem_num = len(act_schema['Elements'])
        elem_entropy = bernoulli_entropy(act_dist[:,-elem_num:], act_mod[:,1:1+elem_num])
    
        total = torch.zeros_like(act_dist[:,0])
        
        offset_dist = 0
        offset_mod = 1 + elem_num
        
        for ei, element in enumerate(act_schema['Elements'].values()):
            
            assert ei == element['Index']
            size_dist = element['DistributionSize']
            size_mod = element['ModifierSize']
            
            total += (1.0 - act_mod[:,1+ei]) * (elem_entropy[:,ei] + schema_entropy(
                element, 
                act_dist[:,offset_dist:offset_dist+size_dist],
                act_mod[:,offset_mod:offset_mod+size_mod]))
            
            offset_dist += size_dist
            offset_mod += size_mod
            
        assert offset_dist + elem_num == act_schema['DistributionSize']
        assert offset_mod == act_schema['ModifierSize']

        return total
    
    elif act_type == 'Array':
        batchsize = len(act_dist)
        act_dist_reshape = act_dist.reshape([batchsize * act_schema['Num'], -1])
        act_mod_reshape = act_mod[:,1:].reshape([batchsize * act_schema['Num'], -1])
        entropy = schema_entropy(act_schema['Element'], act_dist_reshape, act_mod_reshape)
        return entropy.reshape([batchsize, act_schema['Num']]).sum(dim=-1)
        
    elif act_type == 'Encoding':
        return schema_entropy(act_schema['Element'], act_dist, act_mod[:,1:])
        
    else:
        raise Exception('Not Implemented')


def independent_normal_log_prob(mean, log_std, value, masked):
    std = torch.exp(torch.clip(log_std, None, 10.0))
    return (1.0 - masked) * (-((value - mean) ** 2) / (2 * (std ** 2)) - log_std - log_sqrt_2pi)

def multinoulli_log_prob(logits, value, masked):
    log_probs = log_softmax_masked(logits, masked, dim=-1)
    return torch.where(masked == 1.0, 0.0, log_probs * value)
   
def bernoulli_log_prob(logits, value, masked):
    neg_log_prob = -F.softplus(+logits)
    pos_log_prob = -F.softplus(-logits)
    return (1.0 - masked) * (neg_log_prob * (1 - value) + pos_log_prob * value)

def schema_log_prob(act_schema, act_dist, act_sample, act_mod):
    
    act_type = act_schema['Type']
    assert act_schema['DistributionSize'] == act_dist.shape[1]
    assert act_schema['VectorSize'] == act_sample.shape[1]
    assert act_schema['ModifierSize'] == act_mod.shape[1]

    if act_type == 'Null':
        return torch.zeros([len(act_dist)], device=act_dist.device)
    
    if act_type == 'Continuous':
        act_mean, act_log_std = act_dist[:,:act_dist.shape[1]//2], act_dist[:,act_dist.shape[1]//2:]
        act_mod_masked = act_mod[:,1:1+act_dist.shape[1]//2]
        assert act_schema['DistributionSize'] == 2 * act_schema['VectorSize']
        assert act_schema['ModifierSize'] == 1 + 2 * act_mean.shape[1]
        assert act_schema['VectorSize'] == act_mean.shape[1]
        assert act_schema['VectorSize'] == act_log_std.shape[1]

        return independent_normal_log_prob(act_mean, act_log_std, act_sample, act_mod_masked).sum(dim=-1)
    
    elif act_type == 'DiscreteExclusive':

        return multinoulli_log_prob(act_dist, act_sample, act_mod[:,1:]).sum(dim=-1)
    
    elif act_type == 'DiscreteInclusive':

        return bernoulli_log_prob(act_dist, act_sample, act_mod[:,1:]).sum(dim=-1)
        
    elif act_type == 'NamedDiscreteExclusive':

        return multinoulli_log_prob(act_dist, act_sample, act_mod[:,1:]).sum(dim=-1)
    
    elif act_type == 'NamedDiscreteInclusive':

        return bernoulli_log_prob(act_dist, act_sample, act_mod[:,1:]).sum(dim=-1)
        
    elif act_type == 'And':
        
        total = torch.zeros_like(act_dist[:,0])
        
        dist_offset = 0
        smpl_offset = 0
        mod_offset = 1
        
        for ei, element in enumerate(act_schema['Elements'].values()):
            
            assert ei == element['Index']
            dist_size = element['DistributionSize']
            smpl_size = element['VectorSize']
            mod_size = element['ModifierSize']
           
            total += schema_log_prob(
                element, 
                act_dist[:,dist_offset:dist_offset+dist_size],
                act_sample[:,smpl_offset:smpl_offset+smpl_size],
                act_mod[:,mod_offset:mod_offset+mod_size])
                
            dist_offset += dist_size
            smpl_offset += smpl_size
            mod_offset += mod_size
            
        assert dist_offset == act_schema['DistributionSize']
        assert smpl_offset == act_schema['VectorSize']
        assert mod_offset == act_schema['ModifierSize']

        return total
    
    elif act_type == 'OrExclusive':
        
        elem_num = len(act_schema['Elements'])
        elem_logp = multinoulli_log_prob(act_dist[:,-elem_num:], act_sample[:,-elem_num:], act_mod[:,1:1+elem_num])
        
        with torch.no_grad():
            elem_mask_np = act_sample[:,-elem_num:].cpu().numpy()
            elem_indices = _mask_to_indices_exclusive(elem_mask_np, device=act_sample.device)
            
        total = torch.zeros_like(act_dist[:,0])
        
        dist_offset = 0
        mod_offset = 1 + elem_num

        for ei, element in enumerate(act_schema['Elements'].values()):
            
            assert ei == element['Index']
            dist_size = element['DistributionSize']
            smpl_size = element['VectorSize']
            mod_size = element['ModifierSize']
        
            if len(elem_indices[ei]) != 0:
                
                total[elem_indices[ei]] += (1.0 - act_mod[elem_indices[ei],1+ei]) * (elem_logp[elem_indices[ei],ei] + schema_log_prob(
                    element, 
                    act_dist[elem_indices[ei],dist_offset:dist_offset+dist_size],
                    act_sample[elem_indices[ei],:smpl_size],
                    act_mod[elem_indices[ei],mod_offset:mod_offset+mod_size]))
        
            dist_offset += dist_size
            mod_offset += mod_size
        
        assert dist_offset + elem_num == act_schema['DistributionSize']
        assert mod_offset == act_schema['ModifierSize']
        
        return total
    
    elif act_type == 'OrInclusive':
        
        elem_num = len(act_schema['Elements'])
        elem_logp = bernoulli_log_prob(act_dist[:,-elem_num:], act_sample[:,-elem_num:], act_mod[:,1:1+elem_num])
        
        with torch.no_grad():
            elem_mask_np = act_sample[:,-elem_num:].cpu().numpy()
            elem_indices = _mask_to_indices_inclusive(elem_mask_np, device=act_sample.device)
        
        total = torch.zeros_like(act_dist[:,0])

        dist_offset = 0
        smpl_offset = 0
        mod_offset = 1 + elem_num

        for ei, element in enumerate(act_schema['Elements'].values()):
            
            assert ei == element['Index']
            dist_size = element['DistributionSize']
            smpl_size = element['VectorSize']
            mod_size = element['ModifierSize']
            
            if len(elem_indices[ei]) != 0:
                
                total[elem_indices[ei]] += (1.0 - act_mod[elem_indices[ei],1+ei]) * (elem_logp[elem_indices[ei],ei] + schema_log_prob(
                    element, 
                    act_dist[elem_indices[ei],dist_offset:dist_offset+dist_size],
                    act_sample[elem_indices[ei],smpl_offset:smpl_offset+smpl_size],
                    act_mod[elem_indices[ei],mod_offset:mod_offset+mod_size]))

            dist_offset += dist_size
            smpl_offset += smpl_size
            mod_offset += mod_size

        assert dist_offset + elem_num == act_schema['DistributionSize']
        assert smpl_offset + elem_num == act_schema['VectorSize']
        assert mod_offset == act_schema['ModifierSize']

        return total
        
    elif act_type == 'Array':
        batchsize = len(act_dist)
        act_dist_reshape = act_dist.reshape([batchsize * act_schema['Num'], -1])
        act_sample_reshape = act_sample.reshape([batchsize * act_schema['Num'], -1])
        act_mod_reshape = act_mod[:,1:].reshape([batchsize * act_schema['Num'], -1])
        logp = schema_log_prob(act_schema['Element'], act_dist_reshape, act_sample_reshape, act_mod_reshape)
        return logp.reshape([batchsize, act_schema['Num']]).sum(dim=-1)

    elif act_type == 'Encoding':
        return schema_log_prob(act_schema['Element'], act_dist, act_sample, act_mod[:,1:])
        
    else:
        raise Exception('Not Implemented')
    
    
def schema_regularization(act_schema, act_dist):
    
    act_type = act_schema['Type']
    assert act_schema['DistributionSize'] == act_dist.shape[1]
    
    if act_type == 'Null':
        return torch.zeros([len(act_dist)], device=act_dist.device)
    
    if act_type == 'Continuous':
        act_mean, act_log_std = act_dist[:,:act_dist.shape[1]//2], act_dist[:,act_dist.shape[1]//2:]
        return abs(act_mean).sum(dim=-1) + abs(act_log_std).sum(dim=-1)
    
    elif act_type == 'DiscreteExclusive':
        return abs(act_dist).sum(dim=-1)
    
    elif act_type == 'DiscreteInclusive':
        return abs(act_dist).sum(dim=-1)
        
    elif act_type == 'NamedDiscreteExclusive':
        return abs(act_dist).sum(dim=-1)
    
    elif act_type == 'NamedDiscreteInclusive':
        return abs(act_dist).sum(dim=-1)
        
    elif act_type == 'And':
        total = torch.zeros_like(act_dist[:,0])
        offset = 0
        for ei, element in enumerate(act_schema['Elements'].values()):
            assert ei == element['Index']
            size = element['DistributionSize']
            total += schema_regularization(element, act_dist[:,offset:offset+size])
            offset += size
        assert offset == act_schema['DistributionSize']

        return total
    
    elif act_type == 'OrExclusive':
        elem_num = len(act_schema['Elements'])

        total = abs(act_dist[:,-elem_num:]).sum(dim=-1)
        offset = 0
        for ei, element in enumerate(act_schema['Elements'].values()):
            assert ei == element['Index']
            size = element['DistributionSize']
            total += schema_regularization(element, act_dist[:,offset:offset+size])
            offset += size
        assert offset + len(act_schema['Elements']) == act_schema['DistributionSize']

        return total
    
    elif act_type == 'OrInclusive':
        elem_num = len(act_schema['Elements'])

        total = abs(act_dist[:,-elem_num:]).sum(dim=-1)
        offset = 0
        for ei, element in enumerate(act_schema['Elements'].values()):
            assert ei == element['Index']
            size = element['DistributionSize']
            total += schema_regularization(element, act_dist[:,offset:offset+size])
            offset += size
        assert offset + len(act_schema['Elements']) == act_schema['DistributionSize']

        return total
    
    elif act_type == 'Array':    
        batchsize = len(act_dist)
        act_dist_reshape = act_dist.reshape([batchsize * act_schema['Num'], -1])
        reg = schema_regularization(act_schema['Element'], act_dist_reshape)
        return reg.reshape([batchsize, act_schema['Num']]).sum(dim=-1)
    
    elif act_type == 'Encoding':
        return schema_regularization(act_schema['Element'], act_dist)
    
    else:
        raise Exception('Not Implemented')


def schema_noise_mask(act_schema, act_sample):
    
    act_type = act_schema['Type']
    assert act_schema['VectorSize'] == act_sample.shape[1]

    if act_type == 'Null':
        return torch.zeros_like(act_sample)
    
    if act_type == 'Continuous':
        return torch.ones_like(act_sample)
    
    elif act_type == 'DiscreteExclusive':
        return torch.zeros_like(act_sample)
        
    elif act_type == 'DiscreteInclusive':
        return torch.zeros_like(act_sample)
        
    elif act_type == 'NamedDiscreteExclusive':
        return torch.zeros_like(act_sample)
        
    elif act_type == 'NamedDiscreteInclusive':
        return torch.zeros_like(act_sample)
        
    elif act_type == 'And':
        
        mask = torch.zeros_like(act_sample)
        
        smpl_offset = 0
        
        for ei, element in enumerate(act_schema['Elements'].values()):
            
            assert ei == element['Index']
            smpl_size = element['VectorSize']
           
            mask[:,smpl_offset:smpl_offset+smpl_size] = schema_noise_mask(
                element, 
                act_sample[:,smpl_offset:smpl_offset+smpl_size])
                
            smpl_offset += smpl_size
            
        assert smpl_offset == act_schema['VectorSize']

        return mask
    
    elif act_type == 'OrExclusive':
        
        elem_num = len(act_schema['Elements'])
        
        with torch.no_grad():
            elem_mask_np = act_sample[:,-elem_num:].cpu().numpy()
            elem_indices = _mask_to_indices_inclusive(elem_mask_np, device=act_sample.device)
        
        mask = torch.zeros_like(act_sample)
        
        for ei, element in enumerate(act_schema['Elements'].values()):
            
            assert ei == element['Index']
            smpl_size = element['VectorSize']
        
            if len(elem_indices[ei]) != 0:
                
                mask[elem_indices[ei],:smpl_size] = schema_noise_mask(
                    element, 
                    act_sample[elem_indices[ei],:smpl_size])
        
        return mask
    
    elif act_type == 'OrInclusive':
        
        elem_num = len(act_schema['Elements'])
        
        with torch.no_grad():
            elem_mask_np = act_sample[:,-elem_num:].cpu().numpy()
            elem_indices = _mask_to_indices_inclusive(elem_mask_np, device=act_sample.device)
        
        mask = torch.zeros_like(act_sample)

        smpl_offset = 0

        for ei, element in enumerate(act_schema['Elements'].values()):
            
            assert ei == element['Index']
            smpl_size = element['VectorSize']
            
            if len(elem_indices[ei]) != 0:
                
                mask[elem_indices[ei],smpl_offset:smpl_offset+smpl_size] = schema_noise_mask(
                    element, 
                    act_sample[elem_indices[ei],smpl_offset:smpl_offset+smpl_size])

            smpl_offset += smpl_size

        assert smpl_offset + elem_num == act_schema['VectorSize']

        return mask
        
    elif act_type == 'Array':
        batchsize = len(act_sample)
        elemsize = act_schema['Element']['VectorSize']
        act_sample_reshape = act_sample.reshape([batchsize * act_schema['Num'], elemsize])
        mask = schema_noise_mask(act_schema['Element'], act_sample_reshape)
        return mask.reshape([batchsize, act_schema['Num'] * elemsize])

    elif act_type == 'Encoding':
        return schema_noise_mask(act_schema['Element'], act_sample)
        
    else:
        raise Exception('Not Implemented')
        
        
def schema_noise_mask_observation(obs_schema, obs_sample):
    
    obs_type = obs_schema['Type']
    assert obs_schema['VectorSize'] == obs_sample.shape[1]

    if obs_type == 'Null':
        return torch.zeros_like(obs_sample)
    
    elif obs_type == 'Continuous':
        return torch.ones_like(obs_sample)
    
    elif obs_type == 'DiscreteExclusive':
        return torch.zeros_like(obs_sample)
    
    elif obs_type == 'DiscreteInclusive':
        return torch.zeros_like(obs_sample)
        
    elif obs_type == 'NamedDiscreteExclusive':
        return torch.zeros_like(obs_sample)
        
    elif obs_type == 'NamedDiscreteInclusive':
        return torch.zeros_like(obs_sample)
    
    elif obs_type == 'And':
        
        mask = torch.zeros_like(obs_sample)
        
        smpl_offset = 0
        
        for ei, element in enumerate(obs_schema['Elements'].values()):
            
            assert ei == element['Index']
            smpl_size = element['VectorSize']
           
            mask[:,smpl_offset:smpl_offset+smpl_size] = schema_noise_mask_observation(
                element, 
                obs_sample[:,smpl_offset:smpl_offset+smpl_size])
                
            smpl_offset += smpl_size
            
        assert smpl_offset == obs_schema['VectorSize']

        return mask
    
    elif obs_type == 'OrExclusive':
        
        elem_num = len(obs_schema['Elements'])
        
        with torch.no_grad():
            elem_mask_np = obs_sample[:,-elem_num:].cpu().numpy()
            elem_indices = _mask_to_indices_inclusive(elem_mask_np, device=obs_sample.device)
        
        mask = torch.zeros_like(obs_sample)
        
        for ei, element in enumerate(obs_schema['Elements'].values()):
            
            assert ei == element['Index']
            smpl_size = element['VectorSize']
        
            if len(elem_indices[ei]) != 0:
                
                mask[elem_indices[ei],:smpl_size] = schema_noise_mask_observation(
                    element, 
                    obs_sample[elem_indices[ei],:smpl_size])
        
        return mask
    
    elif obs_type == 'OrInclusive':
        
        elem_num = len(obs_schema['Elements'])
        
        with torch.no_grad():
            elem_mask_np = obs_sample[:,-elem_num:].cpu().numpy()
            elem_indices = _mask_to_indices_inclusive(elem_mask_np, device=obs_sample.device)
        
        mask = torch.zeros_like(obs_sample)

        smpl_offset = 0

        for ei, element in enumerate(obs_schema['Elements'].values()):
            
            assert ei == element['Index']
            smpl_size = element['VectorSize']
            
            if len(elem_indices[ei]) != 0:
                
                mask[elem_indices[ei],smpl_offset:smpl_offset+smpl_size] = schema_noise_mask_observation(
                    element, 
                    obs_sample[elem_indices[ei],smpl_offset:smpl_offset+smpl_size])

            smpl_offset += smpl_size

        assert smpl_offset + elem_num == obs_schema['VectorSize']

        return mask
        
    elif obs_type == 'Array':
        batchsize = len(obs_sample)
        elemsize = obs_schema['Element']['VectorSize']
        obs_sample_reshape = obs_sample.reshape([batchsize * obs_schema['Num'], elemsize])
        mask = schema_noise_mask_observation(obs_schema['Element'], obs_sample_reshape)
        return mask.reshape([batchsize, obs_schema['Num'] * elemsize])

    elif obs_type == 'Set':
        batchsize = len(obs_sample)
        elemsize = obs_schema['Element']['VectorSize']
        maxnum = obs_schema['MaxNum']
        
        mask = torch.zeros_like(obs_sample)
        
        with torch.no_grad():
            elemnum = obs_sample[:,-maxnum:].cpu().numpy().astype(np.int32).sum(axis=-1)
        
        smpl_offset = 0
        
        for ei in range(maxnum):
            
            mask[ei < elemnum,smpl_offset:smpl_offset+elemsize] = schema_noise_mask_observation(
                obs_schema['Element'], 
                obs_sample[ei < elemnum,smpl_offset:smpl_offset+elemsize])
            
            smpl_offset += elemsize
        
        assert smpl_offset + maxnum == obs_schema['VectorSize']
        
        return mask


    elif obs_type == 'Encoding':
        return schema_noise_mask_observation(obs_schema['Element'], obs_sample)
        
    else:
        raise Exception('Not Implemented')


class AbstractCommunicator(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def get_network(self, network_id:int):
        raise NotImplementedError
        
    @abstractmethod
    def set_network(self, network_id:int, network:torch.nn.Module, version:int):
        raise NotImplementedError

    @abstractmethod
    def receive_experience(self, replay_buffer_id:int, trim_episode_start:int, trim_episode_end:int):
        raise NotImplementedError

    @abstractmethod
    def send_network(self, network_id:int, network:torch.nn.Module, version:int):
        raise NotImplementedError

    @abstractmethod
    def send_complete(self):
        raise NotImplementedError
    
    @abstractmethod
    def send_ping(self):
        raise NotImplementedError

    @abstractmethod
    def has_stop(self):
        raise NotImplementedError
    
    @abstractmethod
    def receive_stop(self):
        raise NotImplementedError

    @abstractmethod
    def get_batch_size(self):
        raise NotImplementedError


# Experiment Tracker

class AbstractExperimentTracker(ABC):
    def __init__(self, config):
        self.config = config
        self.tracker = None
        self.tracker_class = None
        self.tracking = True

    @abstractmethod
    def initialize_tracker(self):
        raise NotImplementedError
    
    @abstractmethod
    def track(self, data: list):
        raise NotImplementedError
    
    @abstractmethod
    def track_snapshots(self, data: list):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


class TensorboardTracker(AbstractExperimentTracker):
    def __init__(self, config):
        super().__init__(config)

    def initialize_tracker(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tracker_class = SummaryWriter
        except ImportError as e:
            logger.warning('Failed to Load TensorBoard: %s. Please add manually to site-packages.' % str(e))
            self.config['UseTensorBoard'] = False
            self.tracking = False

        if self.tracking:
            self.tracker = self.tracker_class(
                log_dir=self.config['TempDirectory'] + "/TensorBoard/runs/" + self.config['TaskName'] + "_" + self.config['TrainerMethod'] + "_" + self.config['CommunicationType'] + "_" + self.config['TimeStamp'],
                max_queue=1000)

    def track(self, data: list):
        if self.tracking:
            for d in data:
                key, value, step = d
                self.tracker.add_scalar(key, value, step)
    
    def track_snapshots(self, data: list):
        pass

    def close(self):
        pass


class MLFlowTracker(AbstractExperimentTracker):
    def __init__(self, config):
        super().__init__(config)


    def initialize_tracker(self):
        if 'PPOSettings' in self.config:
            settings = self.config['PPOSettings']
        else:
            settings = self.config['BehaviorCloningSettings']

        try:
            import mlflow
            # Set the tracking server URI
            logger.info('Logging to MLflow Uri %s...' % settings['MLflowTrackingUri'])
            self.tracker = mlflow
            mlflow.set_tracking_uri(settings['MLflowTrackingUri'])
        except ImportError as e:
            logger.warning('Failed to Load MLflow : %s. Please add manually to site-packages.' % str(e))
            settings['UseMLflow'] = False
            self.tracking = False
    
    
        if self.tracking:
            self.tracker.set_experiment(self.config["TaskName"])
            self.tracker.start_run()
            self.tracker.log_params(self.config)
            self.tracker.log_dict(self.config, "config.json")

    def track(self, data: list):
        if self.tracking:
            for d in data:
                key, value, step = d
                self.tracker.log_metric(key, value, step)

    def track_snapshots(self, data: list):
        if self.tracking:
            for d in data:
                self.tracker.log_artifact(d)

    def close(self):
        if self.tracking:
            self.tracker.end_run()


def get_experiment_trackers(config):
    trackers = []
    if 'PPOSettings' in config:
        settings = config['PPOSettings']
    else:
        settings = config['BehaviorCloningSettings']
    
    if settings['UseTensorBoard']:
        trackers.append(TensorboardTracker(config))
    # if settings['UseMLflow']:
    #     trackers.append(MLFlowTracker(config))
    if not trackers:
        logger.warning('No experiment tracker selected.')
    return trackers 


def get_alive_threads(threads):
    return [t for t in threads if t.is_alive()]


def get_merged_buffer(buffers):
    merged_buffer = {
        'obs':          [np.concatenate(obs) for obs in zip(*[buffer['obs'] for buffer in buffers])],
        'obs_next':     [np.concatenate(obs_next) for obs_next in zip(*[buffer['obs_next'] for buffer in buffers])],
        'act':          [np.concatenate(act) for act in zip(*[buffer['act'] for buffer in buffers])],
        'mod':          [np.concatenate(mod) for mod in zip(*[buffer['mod'] for buffer in buffers])],
        'mem':          [np.concatenate(mem) for mem in zip(*[buffer['mem'] for buffer in buffers])],
        'mem_next':     [np.concatenate(mem_next) for mem_next in zip(*[buffer['mem_next'] for buffer in buffers])],
        'rew':          [np.concatenate(rew) for rew in zip(*[buffer['rew'] for buffer in buffers])],
        'terminated':   np.concatenate([buffer['terminated'] for buffer in buffers]),
        'truncated':    np.concatenate([buffer['truncated'] for buffer in buffers]),}
    return merged_buffer


def get_merged_stats(stats):
    ep_lengths = [stat['experience/avg_episode_length'] for stat in stats]
    merged_stats = {
        'experience/avg_reward':         [sum(rew) / len(rew) for rew in zip(*[stat['experience/avg_reward'] for stat in stats])],
        'experience/avg_reward_sum':     [sum(rew) / len(rew) for rew in zip(*[stat['experience/avg_reward_sum'] for stat in stats])],
        'experience/avg_episode_length': sum(ep_lengths) / len(ep_lengths),}
    return merged_stats


def build_network(device):
    if device.lower() == 'gpu':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            logger.warning('GPU does not support CUDA. Defaulting to CPU training.')
            device = 'cpu'
    elif device.lower() == 'cpu':
        device = 'cpu'
    else:
        logger.warning('Unknown training device "%s". Defaulting to CPU training.' % device)

    from nne_runtime_basic_cpu_pytorch import NeuralNetwork
    return NeuralNetwork(device=device)
