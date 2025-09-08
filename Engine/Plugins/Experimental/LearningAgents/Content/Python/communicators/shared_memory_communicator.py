# -*- coding: utf-8 -*-
'''
Copyright Epic Games, Inc. All Rights Reserved.
'''

import json
import os
import time
import uuid

from collections import OrderedDict

from .shared_memory import SharedMemory
from train_common import (
    UE_RESPONSE_STOPPED, 
    UE_RESPONSE_UNEXPECTED, 
    UE_RESPONSE_SUCCESS,
    UE_LEARNING_DEVELOPMENT, 
    UE_COMPLETION_TERMINATED,
    UE_COMPLETION_TRUNCATED,)
from train_common import AbstractCommunicator
from train_common import save_snapshot, load_snapshot, build_network, get_merged_buffer, get_merged_stats, create_task_dir

import numpy as np

import logging
logger = logging.getLogger("LearningAgents")


UE_SHARED_MEMORY_EXPERIENCE_EPISODE_NUM     = 0
UE_SHARED_MEMORY_EXPERIENCE_STEP_NUM        = 1
UE_SHARED_MEMORY_EXPERIENCE_SIGNAL          = 2
UE_SHARED_MEMORY_CONFIG_SIGNAL              = 3
UE_SHARED_MEMORY_NETWORK_SIGNAL             = 4
UE_SHARED_MEMORY_COMPLETE_SIGNAL            = 5
UE_SHARED_MEMORY_STOP_SIGNAL                = 6
UE_SHARED_MEMORY_PING_SIGNAL                = 7
UE_SHARED_MEMORY_NETWORK_ID                 = 8
UE_SHARED_MEMORY_REPLAY_BUFFER_ID           = 9

UE_SHARED_MEMORY_CONTROL_NUM                = 10


def shared_memory_map_array(guid, shape, dtype, create=False):
    size = int(np.prod(shape) * np.dtype(dtype).itemsize)
    if size > 0:
        was_error = False

        try:
            handle = SharedMemory(guid, create, size)
        except FileExistsError:
            if create:
                # Windows has trouble cleaning up so use the existing shared memory - C++ side will reinitialize it
                handle = SharedMemory(guid, False, size)
                was_error = True

        assert handle is not None
        array = np.frombuffer(handle.buf, dtype=dtype, count=np.prod(shape)).reshape(shape)
        if was_error:
            # if we were trying to create, but ended up mapping, then we are responsible for cleaning the memory and not C++
            array[:] = 0 

        if create:
            logger.info('Created shared memory with name: ' + guid)
        else:
            logger.info('Mapped existing shared memory with name: ' + guid)

        return handle, array
    else:
        return None, np.empty(shape, dtype=dtype)


def shared_memory_receive_experience_behavior_cloning(
    controls,
    replay_buffer_id,
    replay_buffer):

    # Wait until experience is ready
    
    while not controls[UE_SHARED_MEMORY_EXPERIENCE_SIGNAL]:
    
        if controls[UE_SHARED_MEMORY_STOP_SIGNAL]:
            controls[UE_SHARED_MEMORY_STOP_SIGNAL] = 0
            return UE_RESPONSE_STOPPED, None, None
    
        time.sleep(0.001)
    
    # Check buffer ids match
    if controls[UE_SHARED_MEMORY_REPLAY_BUFFER_ID] != replay_buffer_id:
        logger.error('Invalid replay buffer id, expected %d received %d' % (replay_buffer_id, controls[UE_SHARED_MEMORY_REPLAY_BUFFER_ID]))
        return UE_RESPONSE_UNEXPECTED, None, None

    # Copy experience
    
    episode_num = controls[UE_SHARED_MEMORY_EXPERIENCE_EPISODE_NUM]
    step_num = controls[UE_SHARED_MEMORY_EXPERIENCE_STEP_NUM]
    assert episode_num > 0 and step_num > 0

    episode_starts = replay_buffer.episode_starts[1][:episode_num].reshape([episode_num]).copy()
    episode_lengths = replay_buffer.episode_lengths[1][:episode_num].reshape([episode_num]).copy()
    observations = replay_buffer.observations[0][1][:step_num].reshape([step_num, replay_buffer.observation_nums[0]]).copy()
    actions = replay_buffer.actions[0][1][:step_num].reshape([step_num, replay_buffer.action_nums[0]]).copy()

    if UE_LEARNING_DEVELOPMENT:
        assert np.all(np.isfinite(episode_starts))
        assert np.all(np.isfinite(episode_lengths))
        assert np.all(np.isfinite(observations))
        assert np.all(np.isfinite(actions))

    controls[UE_SHARED_MEMORY_EXPERIENCE_SIGNAL] = 0

    buffer = {
        'obs':          observations,
        'act':          actions,
        'starts':       episode_starts,
        'lengths':      episode_lengths,
    }
    
    return (
        UE_RESPONSE_SUCCESS, 
        buffer,
        None)


class SharedMemoryReplayBuffer:

    def __init__(self, config, shared_memory_config):

        self.max_episode_num = int(config['MaxEpisodeNum'])
        self.max_step_num = int(config['MaxStepNum'])

        self.has_completions = bool(config['HasCompletions'])
        self.has_final_observations = bool(config['HasFinalObservations'])
        self.has_final_memory_states = bool(config['HasFinalMemoryStates'])
        self.is_reinforcement_learning = self.has_completions # This is a little janky

        self.episode_starts = shared_memory_map_array(shared_memory_config['EpisodeStartsGuid'], [self.max_episode_num], np.int32)
        self.episode_lengths = shared_memory_map_array(shared_memory_config['EpisodeLengthsGuid'], [self.max_episode_num], np.int32)

        if self.has_completions:
            self.episode_completion_modes = shared_memory_map_array(shared_memory_config['EpisodeCompletionModesGuid'], [self.max_episode_num], np.uint8)

        self.observation_nums = []
        self.episode_final_observations = []
        self.observations = []
        self.observation_nexts = []
        for index, observation_config in enumerate(config['Observations']):
            observation_id = int(observation_config['Id'])
            observation_name = observation_config['Name']
            observation_schema_id = int(observation_config['SchemaId'])

            observation_num = int(observation_config['VectorDimensionNum'])
            self.observation_nums.append(observation_num)

            if self.has_final_observations:
                self.episode_final_observations.append(
                    shared_memory_map_array(
                        shared_memory_config['EpisodeFinalObservationsGuids'][index],
                        [self.max_episode_num, observation_num],
                        np.float32))
            self.observations.append(
                shared_memory_map_array(
                    shared_memory_config['ObservationsGuids'][index],
                    [self.max_step_num, observation_num],
                    np.float32))

            if self.is_reinforcement_learning:
                self.observation_nexts.append(np.zeros([self.max_step_num, observation_num], dtype=np.float32))

        self.action_nums = []
        self.actions = []
        for index, action_config in enumerate(config['Actions']):
            action_id = int(action_config['Id'])
            action_name = action_config['Name']
            action_schema_id = int(action_config['SchemaId'])
            
            action_num = int(action_config['VectorDimensionNum'])
            self.action_nums.append(action_num)

            self.actions.append(
                shared_memory_map_array(
                    shared_memory_config['ActionsGuids'][index],
                    [self.max_step_num, action_num],
                    np.float32))

        self.action_modifier_nums = []
        self.action_modifiers = []
        for index, action_modifier_config in enumerate(config['ActionModifiers']):
            action_modifier_id = int(action_modifier_config['Id'])
            action_modifier_name = action_modifier_config['Name']
            action_modifier_schema_id = int(action_modifier_config['SchemaId'])
            
            action_modifier_num = int(action_modifier_config['VectorDimensionNum'])
            self.action_modifier_nums.append(action_modifier_num)

            self.action_modifiers.append(
                shared_memory_map_array(
                    shared_memory_config['ActionModifiersGuids'][index],
                    [self.max_step_num, action_modifier_num],
                    np.float32))

        self.memory_state_nums = []
        self.episode_final_memory_states = []
        self.memory_states = []
        self.memory_state_nexts = []
        for index, memory_state_config in enumerate(config['MemoryStates']):
            memory_state_id = int(memory_state_config['Id'])
            memory_state_name = memory_state_config['Name']
            
            memory_state_num = int(memory_state_config['VectorDimensionNum'])
            self.memory_state_nums.append(memory_state_num)

            if self.has_final_memory_states:
                self.episode_final_memory_states.append(shared_memory_map_array(
                    shared_memory_config['EpisodeFinalMemoryStatesGuids'][index],
                    [self.max_episode_num, memory_state_num],
                    np.float32))
            self.memory_states.append(
                shared_memory_map_array(
                    shared_memory_config['MemoryStatesGuids'][index],
                    [self.max_step_num, memory_state_num],
                    np.float32))
            self.memory_state_nexts.append(np.zeros([self.max_step_num, memory_state_num], dtype=np.float32))

        self.reward_nums = []
        self.rewards = []
        for index, reward_config in enumerate(config['Rewards']):
            reward_id = int(reward_config['Id'])
            reward_name = reward_config['Name']

            reward_num = int(reward_config['VectorDimensionNum'])
            self.reward_nums.append(reward_num)

            self.rewards.append(
                shared_memory_map_array(
                    shared_memory_config['RewardsGuids'][index],
                    [self.max_step_num, reward_num],
                    np.float32))

        if self.is_reinforcement_learning:
            self.terminated_buffer = np.zeros([self.max_step_num], dtype=bool)
            self.truncated_buffer = np.zeros([self.max_step_num], dtype=bool)


class SharedMemoryProcess:

    def __init__(self, controls_guid, controls_handle, controls, config_dir):
        self.controls_handle = controls_handle
        self.controls = controls
        self.wait_for_config()
        
        shared_mem_config_file = os.path.join(config_dir, "shared-memory-" + controls_guid + ".json")
        with open(shared_mem_config_file) as f:
            shared_mem_config = json.load(f, object_pairs_hook=OrderedDict)

        data_config_file = os.path.join(config_dir, "data-config.json")
        with open(data_config_file) as f:
            data_config = json.load(f, object_pairs_hook=OrderedDict)

        trainer_config_file = os.path.join(config_dir, "trainer-config.json")
        with open(trainer_config_file) as f:
            trainer_config = json.load(f, object_pairs_hook=OrderedDict)
        
        # Merge configs to reduce impact on rest of codebase
        self.config = {"SharedMemory": {**shared_mem_config}, **data_config, **trainer_config}
        
        # Setup Neural Networks

        self.network_shared_mem_by_id = dict()
        self.network_info_by_id = dict()

        for network_config in self.config['Networks']:
            network_id = int(network_config['Id'])
            network_name = network_config['Name']
            network_max_bytes = int(network_config['MaxByteNum'])

            network_guid_config = self.config['SharedMemory']['NetworkGuids'][network_id]
            network_guid = network_guid_config['Guid']

            self.network_shared_mem_by_id[network_id] = shared_memory_map_array(network_guid, [network_max_bytes], np.uint8)
            self.network_info_by_id[network_id] = (network_name, network_max_bytes)

        # Setup Replay Buffers

        self.replay_buffers_by_id = dict()

        for index, replay_buffer_config in enumerate(self.config['ReplayBuffers']):
            shared_memory_config = self.config['SharedMemory']['ReplayBuffers'][index]

            replay_buffer_id = int(replay_buffer_config['Id'])
            self.replay_buffers_by_id[replay_buffer_id] = SharedMemoryReplayBuffer(replay_buffer_config, shared_memory_config)

    def __del__(self):
        # We need to forget about the shared memories arrays so they can be GC'd
        self.controls = None
        self.network_shared_mem_by_id = None


    def wait_for_config(self):
        while not self.controls[UE_SHARED_MEMORY_CONFIG_SIGNAL]:
            time.sleep(0.001)
        
        return UE_RESPONSE_SUCCESS
    
    def send_network(self, network_id, network):
        while self.controls[UE_SHARED_MEMORY_NETWORK_SIGNAL]:
            time.sleep(0.001)
        
        save_snapshot(self.network_shared_mem_by_id[network_id][1], network)

        self.controls[UE_SHARED_MEMORY_NETWORK_ID] = network_id
        self.controls[UE_SHARED_MEMORY_NETWORK_SIGNAL] = 1
        
        return UE_RESPONSE_SUCCESS

    def receive_network(self, network_id, network):
        self.controls[UE_SHARED_MEMORY_NETWORK_SIGNAL] = 1
        
        while self.controls[UE_SHARED_MEMORY_NETWORK_SIGNAL]:
            time.sleep(0.001)

        if network_id != self.controls[UE_SHARED_MEMORY_NETWORK_ID]:
            logger.error('Invalid network id, expected %d received %d' % (network_id, self.controls[UE_SHARED_MEMORY_NETWORK_ID]))
            return UE_RESPONSE_UNEXPECTED
        
        load_snapshot(self.network_shared_mem_by_id[network_id][1], network)
        
        return UE_RESPONSE_SUCCESS

    def send_complete(self):
        self.controls[UE_SHARED_MEMORY_COMPLETE_SIGNAL] = 1
        return UE_RESPONSE_SUCCESS

    def has_stop(self):
        return self.controls[UE_SHARED_MEMORY_STOP_SIGNAL]

    def receive_stop(self):
        while not self.controls[UE_SHARED_MEMORY_STOP_SIGNAL]:
            time.sleep(0.001)

        self.controls[UE_SHARED_MEMORY_STOP_SIGNAL] = 0
    
        return UE_RESPONSE_SUCCESS

    def send_ping(self):
        self.controls[UE_SHARED_MEMORY_PING_SIGNAL] = 1
        return UE_RESPONSE_SUCCESS
    
    def receive_experience(self, replay_buffer_id, trim_episode_start, trim_episode_end):
        replay_buffer = self.replay_buffers_by_id[replay_buffer_id]

        # Wait until experience is ready
        while not self.controls[UE_SHARED_MEMORY_EXPERIENCE_SIGNAL]:
        
            if self.controls[UE_SHARED_MEMORY_STOP_SIGNAL]:
                self.controls[UE_SHARED_MEMORY_STOP_SIGNAL] = 0
                return UE_RESPONSE_STOPPED, None, None
        
            time.sleep(0.001)
        
        # Check buffer ids match
        if self.controls[UE_SHARED_MEMORY_REPLAY_BUFFER_ID] != replay_buffer_id:
            logger.error('Invalid replay buffer id, expected %d received %d' % (replay_buffer_id, self.controls[UE_SHARED_MEMORY_REPLAY_BUFFER_ID]))
            return UE_RESPONSE_UNEXPECTED, None, None

        episode_num = self.controls[UE_SHARED_MEMORY_EXPERIENCE_EPISODE_NUM]
        step_num = self.controls[UE_SHARED_MEMORY_EXPERIENCE_STEP_NUM]
        assert episode_num > 0 and step_num > 0

        # Reshape data from buffer
        episode_starts = replay_buffer.episode_starts[1][0:episode_num]
        episode_lengths = replay_buffer.episode_lengths[1][0:episode_num]

        if replay_buffer.has_completions:
            episode_completion_modes = replay_buffer.episode_completion_modes[1]
        
        episode_final_observations = [episode_final_observations[1] for episode_final_observations in replay_buffer.episode_final_observations]
        episode_final_memory_states = [episode_final_memory_states[1] for episode_final_memory_states in replay_buffer.episode_final_memory_states]
        observations = [observations[1] for observations in replay_buffer.observations]
        action_modifiers = [action_modifiers[1] for action_modifiers in replay_buffer.action_modifiers]
        actions = [actions[1] for actions in replay_buffer.actions]
        memory_states = [memory_states[1] for memory_states in replay_buffer.memory_states]
        rewards = [rewards[1] for rewards in replay_buffer.rewards]

        avg_rewards = [0.0 for _ in replay_buffer.rewards]
        avg_reward_sums = [0.0 for _ in replay_buffer.rewards]
        avg_episode_length = 0.0
        total_episode_num = 0

        # Compute Buffer Size
        buffer_size = 0

        for ei in range(episode_num):
            ep_len = episode_lengths[ei] - trim_episode_end - trim_episode_start
            if ep_len > 0:
                buffer_size += ep_len
            
        buffer = {
            'obs':          [np.zeros([buffer_size, obs_num], dtype=np.float32) for obs_num in replay_buffer.observation_nums],
            'obs_next':     [np.zeros([buffer_size, obs_num], dtype=np.float32) for obs_num in replay_buffer.observation_nums],
            'act':          [np.zeros([buffer_size, act_num], dtype=np.float32) for act_num in replay_buffer.action_nums],
            'mod':          [np.zeros([buffer_size, mod_num], dtype=np.float32) for mod_num in replay_buffer.action_modifier_nums],
            'mem':          [np.zeros([buffer_size, mem_num], dtype=np.float32) for mem_num in replay_buffer.memory_state_nums],
            'mem_next':     [np.zeros([buffer_size, mem_num], dtype=np.float32) for mem_num in replay_buffer.memory_state_nums],
            'rew':          [np.zeros([buffer_size, rew_num], dtype=np.float32) for rew_num in replay_buffer.reward_nums],
            'terminated':   np.zeros([buffer_size], dtype=bool),
            'truncated':    np.zeros([buffer_size], dtype=bool),
            'starts':       episode_starts.copy(),
            'lengths':      episode_lengths.copy(),
        }
        
        # Fill Buffer
        
        buffer_offset = 0

        for ei in range(episode_num):
            
            ep_start = episode_starts[ei] + trim_episode_start
            ep_end = episode_starts[ei] + episode_lengths[ei] - trim_episode_end
            ep_len = episode_lengths[ei] - trim_episode_end - trim_episode_start
            
            if ep_len > 0:
            
                obs_list = [observation[ep_start:ep_end] for observation in observations]
                act_list = [action[ep_start:ep_end] for action in actions]
                mod_list = [action_modifier[ep_start:ep_end] for action_modifier in action_modifiers]
                mem_list = [memory_state[ep_start:ep_end] for memory_state in memory_states]
                rew_list = [reward[ep_start:ep_end] for reward in rewards]
                
                if UE_LEARNING_DEVELOPMENT:
                    for obs in obs_list:
                        assert np.all(np.isfinite(obs))
                    for act in act_list:
                        assert np.all(np.isfinite(act))
                    for mod in mod_list:
                        assert np.all(np.isfinite(mod))
                    for mem in mem_list:
                        assert np.all(np.isfinite(mem))
                    for rew in rew_list:
                        assert np.all(np.isfinite(rew))
                
                for index, rew in enumerate(rew_list):
                    avg_rewards[index] += rew.mean()
                    avg_reward_sums[index] += rew.sum()

                avg_episode_length += float(ep_len)
                total_episode_num += 1
                
                obs_nexts = [obs_next_buffer[:ep_len] for obs_next_buffer in replay_buffer.observation_nexts]
                for index, obs_next in enumerate(obs_nexts):
                    obs_next[:-1] = obs_list[index][1:]
                    obs_next[-1] = episode_final_observations[index][ei]

                mem_nexts = [mem_next_buffer[:ep_len] for mem_next_buffer in replay_buffer.memory_state_nexts]
                for index, mem_next in enumerate(mem_nexts):
                    mem_next[:-1] = mem_list[index][1:]
                    mem_next[-1] = episode_final_memory_states[index][ei]

                terminated = replay_buffer.terminated_buffer[:ep_len]
                terminated[:-1] = False
                terminated[-1] = (episode_completion_modes[ei] == UE_COMPLETION_TERMINATED)

                truncated = replay_buffer.truncated_buffer[:ep_len]
                truncated[:-1] = False
                truncated[-1] = (episode_completion_modes[ei] == UE_COMPLETION_TRUNCATED)

                for index, obs in enumerate(obs_list):
                    buffer['obs'][index][buffer_offset:buffer_offset+ep_len] = obs
                    buffer['obs_next'][index][buffer_offset:buffer_offset+ep_len] = obs_nexts[index]

                for index, act in enumerate(act_list):
                    buffer['act'][index][buffer_offset:buffer_offset+ep_len] = act

                for index, mod in enumerate(mod_list):
                    buffer['mod'][index][buffer_offset:buffer_offset+ep_len] = mod

                for index, mem in enumerate(mem_list):
                    buffer['mem'][index][buffer_offset:buffer_offset+ep_len] = mem
                    buffer['mem_next'][index][buffer_offset:buffer_offset+ep_len] = mem_nexts[index]
                
                for index, rew, in enumerate(rew_list):
                    buffer['rew'][index][buffer_offset:buffer_offset+ep_len] = rew
                
                buffer['terminated'][buffer_offset:buffer_offset+ep_len] = terminated
                buffer['truncated'][buffer_offset:buffer_offset+ep_len] = truncated
                
                buffer_offset += ep_len

        self.controls[UE_SHARED_MEMORY_NETWORK_ID] = -1
        self.controls[UE_SHARED_MEMORY_EXPERIENCE_SIGNAL] = 0
        
        assert buffer_offset == buffer_size
        
        stats = {
            'experience/avg_reward':  [0.0 if total_episode_num == 0 else avg_reward / total_episode_num for avg_reward in avg_rewards],
            'experience/avg_reward_sum':  [0.0 if total_episode_num == 0 else avg_reward_sum / total_episode_num for avg_reward_sum in avg_reward_sums],
            'experience/avg_episode_length':  0.0 if total_episode_num == 0 else avg_episode_length / total_episode_num,
        }
        
        return UE_RESPONSE_SUCCESS, buffer, stats


class SharedMemoryCommunicator(AbstractCommunicator):

    def __init__(self, controls_guid, process_num, task_dir, task_name, create_mem, make_task_dir):

        self.name = 'sharedmemory'
        self.task_name = task_name
        self.task_dir = task_dir
        self.process_num = process_num

        if not controls_guid:
            controls_guids = ['{' + str(uuid.uuid4()).upper() + '}' for _ in range(process_num)]
        else:
            controls_guids = [controls_guid]

        self.shared_memory_processes = []
        tuples = []
        for controls_guid in controls_guids:
            handle, controls = shared_memory_map_array(controls_guid, [UE_SHARED_MEMORY_CONTROL_NUM], np.int32, create_mem)
            tuples.append((controls_guid, handle, controls))
        
        if make_task_dir:
            (self.task_dir, _) = create_task_dir(task_dir, task_name)
            self.task_config_dir = os.path.join(self.task_dir, "Configs")

            if not os.path.exists(self.task_config_dir):
                os.makedirs(self.task_config_dir)
        else:
            self.task_config_dir = os.path.join(task_dir, "Configs")

        for (controls_guid, handle, controls) in tuples:
            self.shared_memory_processes.append(SharedMemoryProcess(controls_guid, handle, controls, self.task_config_dir))

    def get_network(self, network_id):
        # TODO: move "Device" out of PPOSettings
        if "PPOSettings" in self.shared_memory_processes[0].config:
            device = self.shared_memory_processes[0].config["PPOSettings"]["Device"]
        else:
            device = self.shared_memory_processes[0].config["BehaviorCloningSettings"]["Device"]

        network = build_network(device)
        response = self.receive_network(network_id, network)
        assert response == UE_RESPONSE_SUCCESS
        return network
    
    def set_network(self, network_id, network, version):
        self.send_network(network_id, network, version) 

    def send_network(self, network_id, network, version):
        for process in self.shared_memory_processes:
            result = process.send_network(network_id, network)
            if result != UE_RESPONSE_SUCCESS:
                return result

        return UE_RESPONSE_SUCCESS
        
    def receive_network(self, network_id, network):
        for process in self.shared_memory_processes:
            result = process.receive_network(network_id, network)
            if result != UE_RESPONSE_SUCCESS:
                return result

        return UE_RESPONSE_SUCCESS

    def receive_experience(self, replay_buffer_id, trim_episode_start, trim_episode_end):
        replay_buffer = self.shared_memory_processes[0].replay_buffers_by_id[replay_buffer_id]
        if replay_buffer.is_reinforcement_learning:
            buffers = []
            stats = []

            for process in self.shared_memory_processes:
                result, buffer, stat = process.receive_experience(replay_buffer_id, trim_episode_start, trim_episode_end)
                
                if result != UE_RESPONSE_SUCCESS:
                    return result, None, None
                
                buffers.append(buffer)
                stats.append(stat)
            
            merged_buffer = get_merged_buffer(buffers)
            merged_buffer['lengths'] = np.concatenate([buffer['lengths'] for buffer in buffers])
            
            # We need to re-number the starts since concatenating shifts all the starts after the first worker process
            merged_buffer['starts'] = np.cumsum(np.append(0, merged_buffer['lengths'])[0:-1])
            
            logger.info('\rEpisode Num: %i | Total Step Num: %i' % (merged_buffer['starts'].shape[0], merged_buffer['obs'][0].shape[0]))

            return UE_RESPONSE_SUCCESS, merged_buffer, get_merged_stats(stats)
        else:
            return shared_memory_receive_experience_behavior_cloning(
                self.shared_memory_processes[0].controls,
                replay_buffer_id,
                replay_buffer)

    def send_complete(self):
        for process in self.shared_memory_processes:
            result = process.send_complete()
            if result != UE_RESPONSE_SUCCESS:
                return result
        return UE_RESPONSE_SUCCESS
    
    def send_ping(self):
        for process in self.shared_memory_processes:
            result = process.send_ping()
            if result != UE_RESPONSE_SUCCESS:
                return result
        return UE_RESPONSE_SUCCESS
    
    def has_stop(self):
        for process in self.shared_memory_processes:
            if process.has_stop():
                return True
        return False
    
    def receive_stop(self):
        for process in self.shared_memory_processes:
            result = process.receive_stop()
            if result != UE_RESPONSE_SUCCESS:
                return result
        return UE_RESPONSE_SUCCESS

    def get_batch_size(self):
        return self.process_num
