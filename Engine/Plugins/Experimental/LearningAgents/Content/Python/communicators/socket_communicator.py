# -*- coding: utf-8 -*-
'''
Copyright Epic Games, Inc. All Rights Reserved.
'''

import socket
import threading
import select
import numpy as np
import logging
import json
from collections import OrderedDict
import time
import queue
from contextlib import contextmanager

from train_common import AbstractCommunicator
from train_common import (UE_RESPONSE_SUCCESS, 
                          UE_RESPONSE_UNEXPECTED, 
                          UE_RESPONSE_STOPPED,
                          UE_LEARNING_DEVELOPMENT,
                          UE_COMPLETION_TERMINATED,
                          UE_COMPLETION_TRUNCATED)
from train_common import save_snapshot, load_snapshot, build_network, get_alive_threads, get_merged_buffer, get_merged_stats

logger = logging.getLogger("LearningAgents")

UE_SOCKET_SIGNAL_INVALID            = 0
UE_SOCKET_SIGNAL_SEND_CONFIG        = 1
UE_SOCKET_SIGNAL_SEND_EXPERIENCE    = 2
UE_SOCKET_SIGNAL_RECEIVE_NETWORK    = 3
UE_SOCKET_SIGNAL_SEND_NETWORK       = 4
UE_SOCKET_SIGNAL_RECEIVE_COMPLETE   = 5
UE_SOCKET_SIGNAL_SEND_STOP          = 6
UE_SOCKET_SIGNAL_RECEIVE_PING       = 7

SOCKET_HEARTBEAT_ACTIVE = 1
SOCKET_HEARTBEAT_INACTIVE = 0
SOCKET_HEARTBEAT_STOPPED = -1

NETWORKS_VERSION_UNTRAINED = -1
NETWORKS_VERSION_STALE = -10


def socket_receive_all(sock, byte_num):
    data = b''
    while len(data) < byte_num:
        data += sock.recv(byte_num - len(data))
    return data
    
def socket_send_all(sock, data):
    sent = 0
    while sent < len(data):
        sent += sock.send(data[sent:])

def socket_send_network(socket, network_id, network, version, buffer, network_signal):
    save_snapshot(buffer, network)
    socket_send_all(socket, b'%c' % network_signal)
    socket_send_all(socket, network_id.to_bytes(4, 'little'))
    socket_send_all(socket, version.to_bytes(4, 'little', signed=True))
    socket_send_all(socket, buffer.tobytes())
    return UE_RESPONSE_SUCCESS

def socket_receive_network(socket, network_id, network, buffer, network_signal):
    signal = ord(socket_receive_all(socket, 1))
    
    if signal != network_signal:
        return UE_RESPONSE_UNEXPECTED
    
    id = int.from_bytes(socket_receive_all(socket, 4), byteorder='little')

    if id != network_id:
        return UE_RESPONSE_UNEXPECTED

    buffer[:] = np.frombuffer(socket_receive_all(socket, len(buffer)), dtype=np.uint8)
    
    load_snapshot(buffer, network)
    
    return UE_RESPONSE_SUCCESS

def socket_send_complete(socket):
    socket_send_all(socket, b'%c' % UE_SOCKET_SIGNAL_RECEIVE_COMPLETE)
    return UE_RESPONSE_SUCCESS

def socket_send_ping(socket):
    socket_send_all(socket, b'%c' % UE_SOCKET_SIGNAL_RECEIVE_PING)
    return UE_RESPONSE_SUCCESS

def socket_receive_experience_reinforcement(
    socket, 
    replay_buffer_id,
    replay_buffer,
    trim_episode_start, 
    trim_episode_end):
    
    signal = ord(socket_receive_all(socket, 1))
    
    if signal == UE_SOCKET_SIGNAL_SEND_STOP:
        return UE_RESPONSE_STOPPED, None, None, None
        
    if signal != UE_SOCKET_SIGNAL_SEND_EXPERIENCE:
        print(f'signal error {signal}')
        return UE_RESPONSE_UNEXPECTED, None, None, None
    
    networks_version = int.from_bytes(socket_receive_all(socket, 4), byteorder='little', signed=True)

    id = int.from_bytes(socket_receive_all(socket, 4), byteorder='little')

    if id != replay_buffer_id:
        print(f'id error {id} // {replay_buffer_id}')
        return UE_RESPONSE_UNEXPECTED, None, None, None
    
    episode_num = int.from_bytes(socket_receive_all(socket, 4), byteorder='little')
    step_num = int.from_bytes(socket_receive_all(socket, 4), byteorder='little')
    assert episode_num > 0 and step_num > 0

    episode_starts = np.frombuffer(socket_receive_all(socket, episode_num * np.dtype(np.int32).itemsize), dtype=np.int32).reshape([episode_num])
    episode_lengths = np.frombuffer(socket_receive_all(socket, episode_num * np.dtype(np.int32).itemsize), dtype=np.int32).reshape([episode_num])

    episode_completion_modes = np.frombuffer(socket_receive_all(socket, episode_num * np.dtype(np.uint8).itemsize), dtype=np.uint8).reshape([episode_num])

    episode_final_observations = []
    for observation_num in replay_buffer.observation_nums:
        episode_final_observations.append(np.frombuffer(socket_receive_all(socket, episode_num * observation_num * np.dtype(np.float32).itemsize), dtype=np.float32).reshape([episode_num, observation_num]))

    episode_final_memory_states = []
    for memory_state_num in replay_buffer.memory_state_nums:
        episode_final_memory_states.append(np.frombuffer(socket_receive_all(socket, episode_num * memory_state_num * np.dtype(np.float32).itemsize), dtype=np.float32).reshape([episode_num, memory_state_num]))

    observations = []
    for observation_num in replay_buffer.observation_nums:
        observations.append(np.frombuffer(socket_receive_all(socket, step_num * observation_num * np.dtype(np.float32).itemsize), dtype=np.float32).reshape([step_num, observation_num]))
    
    actions = []
    for action_num in replay_buffer.action_nums:
        actions.append(np.frombuffer(socket_receive_all(socket, step_num * action_num * np.dtype(np.float32).itemsize), dtype=np.float32).reshape([step_num, action_num]))
    
    action_modifiers = []
    for action_modifier_num in replay_buffer.action_modifier_nums:
        action_modifiers.append(np.frombuffer(socket_receive_all(socket, step_num * action_modifier_num * np.dtype(np.float32).itemsize), dtype=np.float32).reshape([step_num, action_modifier_num]))
    
    memory_states = []
    for memory_state_num in replay_buffer.memory_state_nums:
        memory_states.append(np.frombuffer(socket_receive_all(socket, step_num * memory_state_num * np.dtype(np.float32).itemsize), dtype=np.float32).reshape([step_num, memory_state_num]))
    
    rewards = []
    for reward_num in replay_buffer.reward_nums:
        rewards.append(np.frombuffer(socket_receive_all(socket, step_num * reward_num * np.dtype(np.float32).itemsize), dtype=np.float32).reshape([step_num, reward_num]))

    avg_rewards = [0.0 for _ in rewards]
    avg_reward_sums = [0.0 for _ in rewards]
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
            
            obs_nexts = [obs_next_buffer[:ep_len] for obs_next_buffer in replay_buffer.observation_next_buffers]
            for index, obs_next in enumerate(obs_nexts):
                obs_next[:-1] = obs_list[index][1:]
                obs_next[-1] = episode_final_observations[index][ei]

            mem_nexts = [mem_next_buffer[:ep_len] for mem_next_buffer in replay_buffer.mem_next_buffers]
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
    
    assert buffer_offset == buffer_size
    
    stats = {
        'experience/avg_reward':  [0.0 if total_episode_num == 0 else avg_reward / total_episode_num for avg_reward in avg_rewards],
        'experience/avg_reward_sum':  [0.0 if total_episode_num == 0 else avg_reward_sum / total_episode_num for avg_reward_sum in avg_reward_sums],
        'experience/avg_episode_length':  0.0 if total_episode_num == 0 else avg_episode_length / total_episode_num,
    }
    
    return UE_RESPONSE_SUCCESS, buffer, stats, networks_version


def socket_receive_experience_behavior_cloning(
    socket,
    replay_buffer_id,
    replay_buffer):
    
    observation_num = replay_buffer.observation_nums[0]
    action_num = replay_buffer.action_nums[0]

    signal = ord(socket_receive_all(socket, 1))
        
    if signal != UE_SOCKET_SIGNAL_SEND_EXPERIENCE:
        return UE_RESPONSE_UNEXPECTED, None, None
    
    networks_version = int.from_bytes(socket_receive_all(socket, 4), byteorder='little')
    id = int.from_bytes(socket_receive_all(socket, 4), byteorder='little')

    if id != replay_buffer_id:
        return UE_RESPONSE_UNEXPECTED, None, None

    episode_num = int.from_bytes(socket_receive_all(socket, 4), byteorder='little')
    step_num = int.from_bytes(socket_receive_all(socket, 4), byteorder='little')
    assert episode_num > 0 and step_num > 0

    episode_starts = np.frombuffer(socket_receive_all(socket, episode_num * np.dtype(np.int32).itemsize), dtype=np.int32).reshape([episode_num]).copy()
    episode_lengths = np.frombuffer(socket_receive_all(socket, episode_num * np.dtype(np.int32).itemsize), dtype=np.int32).reshape([episode_num]).copy()
    observations = np.frombuffer(socket_receive_all(socket, step_num * observation_num * np.dtype(np.float32).itemsize), dtype=np.float32).reshape([step_num, observation_num]).copy()
    actions = np.frombuffer(socket_receive_all(socket, step_num * action_num * np.dtype(np.float32).itemsize), dtype=np.float32).reshape([step_num, action_num]).copy()

    if UE_LEARNING_DEVELOPMENT:
        assert np.all(np.isfinite(episode_starts))
        assert np.all(np.isfinite(episode_lengths))
        assert np.all(np.isfinite(observations))
        assert np.all(np.isfinite(actions))

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


def socket_receive_config(socket):

    signal = ord(socket_receive_all(socket, 1))
    if signal != UE_SOCKET_SIGNAL_SEND_CONFIG:
        return UE_RESPONSE_UNEXPECTED, None

    config_length = int.from_bytes(socket_receive_all(socket, 4), byteorder='little')
    
    config = socket_receive_all(socket, config_length)

    return UE_RESPONSE_SUCCESS, config
    
    
def socket_has_stop(socket):
    r, _, _ = select.select([socket],[],[],0)
    return socket in r

def socket_receive_stop(socket):

    signal = ord(socket_receive_all(socket, 1))
    if signal != UE_SOCKET_SIGNAL_SEND_STOP:
        return UE_RESPONSE_UNEXPECTED, None

    return UE_RESPONSE_SUCCESS


class SocketReplayBuffer:

    def __init__(self, config):

        self.max_episode_num = int(config['MaxEpisodeNum'])
        self.max_step_num = int(config['MaxStepNum'])

        self.has_completions = bool(config['HasCompletions'])
        self.has_final_observations = bool(config['HasFinalObservations'])
        self.has_final_memory_states = bool(config['HasFinalMemoryStates'])
        self.is_reinforcement_learning = self.has_completions # This is a little janky

        self.observation_nums = []
        self.observation_next_buffers = []
        for observation_config in config['Observations']:
            observation_id = int(observation_config['Id'])
            observation_name = observation_config['Name']
            observation_schema_id = int(observation_config['SchemaId'])

            observation_num = int(observation_config['VectorDimensionNum'])
            self.observation_nums.append(observation_num)

            if self.is_reinforcement_learning:
                self.observation_next_buffers.append(np.zeros([self.max_step_num, observation_num], dtype=np.float32))

        self.action_nums = []
        for action_config in config['Actions']:
            action_id = int(action_config['Id'])
            action_name = action_config['Name']
            action_schema_id = int(action_config['SchemaId'])
            action_num = int(action_config['VectorDimensionNum'])

            self.action_nums.append(action_num)

        self.action_modifier_nums = []
        for action_modifier_config in config['ActionModifiers']:
            action_modifier_id = int(action_modifier_config['Id'])
            action_modifier_name = action_modifier_config['Name']
            action_modifier_schema_id = int(action_modifier_config['SchemaId'])
            action_modifier_num = int(action_modifier_config['VectorDimensionNum'])

            self.action_modifier_nums.append(action_modifier_num)

        self.memory_state_nums = []
        self.mem_next_buffers = []
        for memory_state_config in config['MemoryStates']:
            memory_state_id = int(memory_state_config['Id'])
            memory_state_name = memory_state_config['Name']
            memory_state_num = int(memory_state_config['VectorDimensionNum'])

            self.memory_state_nums.append(memory_state_num)
            self.mem_next_buffers.append(np.zeros([self.max_step_num, memory_state_num], dtype=np.float32))

        self.reward_nums = []
        for reward_config in config['Rewards']:
            reward_id = int(reward_config['Id'])
            reward_name = reward_config['Name']
            reward_num = int(reward_config['VectorDimensionNum'])

            self.reward_nums.append(reward_num)

        self.terminated_buffer = np.zeros([self.max_step_num], dtype=bool)
        self.truncated_buffer = np.zeros([self.max_step_num], dtype=bool)


class RLSocketReplayBuffer(SocketReplayBuffer):
    def __init__(self, config):
        super().__init__(config)
        for observation_config in config['Observations']:
            observation_num = int(observation_config['VectorDimensionNum'])
            self.observation_next_buffers.append(np.zeros([self.max_step_num, observation_num], dtype=np.float32))


class ClientSocket():
    def __init__(self, socket: socket.socket):
        self.socket = socket
        self.name = socket.getpeername()
        self._networks_version = NETWORKS_VERSION_UNTRAINED
        self._ready_to_receive_networks = False    
    
    @property
    def networks_version(self):
        return self._networks_version
    
    @property
    def ready_to_receive_networks(self):
        return self._ready_to_receive_networks
    
    def set_ready_to_receive_networks(self, is_ready):
        self._ready_to_receive_networks = is_ready

    def set_networks_version(self, networks_version):
        self._networks_version = networks_version

    def __eq__(self, other):
        if not isinstance(other, ClientSocket):
            return False
        try:
            return self.socket == other.socket
        except ConnectionError as e: 
            return False
        
    def __hash__(self):
        return hash(self.socket)

    def __call__(self):
        return self.socket
        

class SocketCommunicator(AbstractCommunicator):

    def __init__(self, host, port, min_batch_size=1, max_batch_size=1, connection_waiting_timeout=None, discard_stale_experiences=True):        
        self.name = 'socket'
        self.lock = threading.Lock()

        self.server_socket = None
        self.client_sockets = []

        self.host = host
        self.port = port

        self.listener_active = True
        self.initialized = False
        self.sender_active = True
        self.sending = False

        self.connection_waiting_timeout = connection_waiting_timeout

        self.discard_stale_experiences = discard_stale_experiences

        self.network_buffers_by_id = dict()
        self.network_info_by_id = dict()
        self.networks_by_id = dict() # updated to keep the latest networks
        self.networks_version = NETWORKS_VERSION_UNTRAINED

        self.replay_buffers_by_id = dict()

        self.config = None
        self.device = None
        
        self.trim_episode_start = 0
        self.trim_episode_end = 0

        self.min_batch_size = min_batch_size if min_batch_size and min_batch_size > 0 else 1
        self.max_batch_size = max_batch_size if max_batch_size and max_batch_size >= min_batch_size else 1

        # Daemon heartbeat thread to keep UE processes that are waiting for network from timing out
        self.heartbeat_status = SOCKET_HEARTBEAT_INACTIVE
        self.heartbeat_interval = 5  # seconds
        self.heartbeat_thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.heartbeat_thread.start()

        self.batch_size = 0

        self.buffers_queue_by_replay_ids: dict[int, queue.Queue]= {}

    def start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        
        # accept new connections
        listener_thread = threading.Thread(target=self._listener, daemon=True)
        listener_thread.start()

        # send networks
        sender_thread = threading.Thread(target=self._sender, daemon=True)
        sender_thread.start()
    
        logger.info(f"Server listening on {self.host}:{self.port}")

        while not self.initialized:
            time.sleep(1)

        return True

    def get_config(self):
        wait_time = 0
        while not self.config:
            time.sleep(1)
            wait_time += 1
            if self.connection_waiting_timeout and wait_time > self.connection_waiting_timeout:
                raise Exception(f'Timeout waiting for config after {self.connection_waiting_timeout} seconds...')
        return self.config

    def get_network(self, network_id):
        wait_time = 0
        while not self.networks_by_id:
            time.sleep(1)
            wait_time += 1
            if self.connection_waiting_timeout and wait_time > self.connection_waiting_timeout:
                raise Exception(f'Timeout waiting for networks_by_id after {self.connection_waiting_timeout} seconds...')
        return self.networks_by_id[network_id]

    def set_network(self, network_id, network, version):
        with self.lock:
            self.networks_by_id[network_id] = network
            self.networks_version = version

    def receive_experience(self, replay_buffer_id, trim_episode_start, trim_episode_end):
        replay_buffer = self.replay_buffers_by_id[replay_buffer_id]
        if replay_buffer.is_reinforcement_learning:
            buffers = []
            stats = []
            self.heartbeat_status = SOCKET_HEARTBEAT_ACTIVE

            try:
                while True:
                    with self.lock:
                        buffers_queue = self.buffers_queue_by_replay_ids.get(replay_buffer_id)
                        while len(buffers) < self.max_batch_size:
                            if not buffers_queue or buffers_queue.empty():
                                break
                            buffer, stat, networks_version, socket_name = buffers_queue.get()
                            if self.discard_stale_experiences:
                                if networks_version < self.networks_version: 
                                    logger.info(f"Discarding stale buffer with networks version [{networks_version}] from connection {socket_name}; Latest networks version is [{self.networks_version}]")
                                    continue
                                elif networks_version > self.networks_version:
                                    raise Exception(f'Networks version received {networks_version} is greater than the current networks version {self.networks_version}. Something really wrong has happened...')
                            buffers.append(buffer)
                            stats.append(stat)
                    if len(buffers) < self.min_batch_size:
                        time.sleep(1)
                    else:
                        break
            finally:         
                self.heartbeat_status = SOCKET_HEARTBEAT_INACTIVE

            self.batch_size = len(buffers)
            return UE_RESPONSE_SUCCESS, get_merged_buffer(buffers), get_merged_stats(stats)
        else:
            return socket_receive_experience_behavior_cloning(self.client_sockets[0](), replay_buffer_id, replay_buffer)

    def send_network(self, network_id, network, version):
        response = UE_RESPONSE_SUCCESS
        with self.lock:
            self.networks_by_id[network_id] = network
            self.networks_version = version
            for client_socket in reversed(self.client_sockets):
                try: # Send and Receive are swapped in the signal constant because they refer to what is happening on the C++ side.
                    new_response = socket_send_network(client_socket(), network_id, network, version, self.network_buffers_by_id[network_id], UE_SOCKET_SIGNAL_RECEIVE_NETWORK)
                    if new_response != UE_RESPONSE_SUCCESS:
                        response = new_response
                except ConnectionError as e:
                    self._handle_connection_error(client_socket)
                    logger.warning(f"Cannot send network to a connection because the connection is no longer valid: {e}")
                    response = UE_RESPONSE_UNEXPECTED
        return response

    def send_complete(self):
        responses = []
        with self.lock:
            for client_socket in reversed(self.client_sockets):
                try:
                    responses.append(socket_send_complete(client_socket()))
                except ConnectionError as e:
                    self._handle_connection_error(client_socket)
                    responses.append(UE_RESPONSE_UNEXPECTED)
                    logger.warning(f"Cannot send complete to a connection because it's no longer valid: {e}")
                
        if all(response == UE_RESPONSE_SUCCESS for response in responses):
            return UE_RESPONSE_SUCCESS
        else:
            return UE_RESPONSE_UNEXPECTED

    def send_ping(self):
        responses = []
        with self.lock:
            if not self.sending:
                for client_socket in reversed(self.client_sockets):
                    try:
                        responses.append(socket_send_ping(client_socket()))
                    except ConnectionError as e:
                        self._handle_connection_error(client_socket)
                        responses.append(UE_RESPONSE_UNEXPECTED)
                        logger.warning(f"Cannot send ping to a connection because it's no longer valid: {e}")

        if all(response == UE_RESPONSE_SUCCESS for response in responses):
            return UE_RESPONSE_SUCCESS
        else:
            return UE_RESPONSE_UNEXPECTED

    def has_stop(self):
        responses = []
        with self.lock:
            for client_socket in self.client_sockets:
                responses.append(socket_has_stop(client_socket()))
 
        if all(response == UE_RESPONSE_SUCCESS for response in responses):
            return UE_RESPONSE_SUCCESS
        else:
            return UE_RESPONSE_UNEXPECTED
    
    def receive_stop(self):
        responses = []
        with self.lock:
            for client_socket in self.client_sockets:
                responses.append(socket_receive_stop(client_socket()))
 
        if all(response == UE_RESPONSE_SUCCESS for response in responses):
            return UE_RESPONSE_SUCCESS
        else:
            return UE_RESPONSE_UNEXPECTED

    def get_batch_size(self):
        return self.batch_size

    def _listener(self):
        while self.listener_active:
            try:
                client_socket, addr = self.server_socket.accept()
                logger.info(f"Accepted connection from {addr}!")

                # because of synchronous socket communication, consume config & first networks buffer from all connections even though config & networks are recorded only from the first connection 
                with self.lock:
                    config = self._receive_config(client_socket)
                    networks_by_id = self._receive_networks(client_socket)

                    if not self.client_sockets:
                        self.config = config
                        self.networks_by_id = networks_by_id
                        self.initialized = True

                    client_socket_wrapper = ClientSocket(client_socket)
                    self.client_sockets.append(client_socket_wrapper)

                    worker_thread = threading.Thread(target=self._worker, args=(client_socket_wrapper,), daemon=True)
                    worker_thread.start()

                self.server_socket.settimeout(1)
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Error accepting connection: {e}")
        
    def _sender(self):
        while self.sender_active:
            time.sleep(1)
            with self.lock, self._sending_context():
                for client_socket in self.client_sockets:
                    try:
                        if client_socket.ready_to_receive_networks and client_socket.networks_version < self.networks_version:
                            logger.info(f"Sending networks to connection {client_socket.name}... Network IDs: {list(self.networks_by_id.keys())} | Version: {self.networks_version}")
                            self._send_networks(client_socket())
                            client_socket.set_ready_to_receive_networks(False)
                            client_socket.set_networks_version(self.networks_version)
                    except ConnectionError as e:
                        self._handle_connection_error(client_socket)
                        logger.warning(f"Cannot send networks to a connection because it's no longer valid: {e}")

    def _worker(self, client_socket: ClientSocket):
        try: 
            while True:
                if client_socket.ready_to_receive_networks and self._test_connection(client_socket): # do not read socket until client has received new networks
                    time.sleep(1)
                    continue
                for replay_buffer_id, replay_buffer in self.replay_buffers_by_id.items():
                    result, buffer, stat, networks_version = socket_receive_experience_reinforcement(
                        client_socket(),
                        replay_buffer_id,
                        replay_buffer,
                        self.trim_episode_start,
                        self.trim_episode_end
                    )
                    if result == UE_RESPONSE_SUCCESS:
                        buffers_queue = self.buffers_queue_by_replay_ids.setdefault(replay_buffer_id, queue.Queue())
                        buffers_queue.put((buffer, stat, networks_version, client_socket.name))
                    else: 
                        raise ConnectionError(f"Receive experience was unsuccessful: UE RESPONSE <{result}>")

                with self.lock:
                    if len(self.client_sockets) < self.min_batch_size:
                        networks_version = NETWORKS_VERSION_STALE # set learning socket's networks version to stale so that sender will send this connection the latest networks again
                    client_socket.set_ready_to_receive_networks(True)
                    client_socket.set_networks_version(networks_version)

        except ConnectionError as e: 
            self._handle_connection_error(client_socket)
            logger.warning(f"Worker cannot receive experience from a connection because it's no longer valid: {e}")


    def _heartbeat(self):
        while self.heartbeat_status != SOCKET_HEARTBEAT_STOPPED:
            if self.heartbeat_status == SOCKET_HEARTBEAT_ACTIVE:
                self.send_ping()
                time.sleep(self.heartbeat_interval)
            else:
                time.sleep(0.1)
    
    def _receive_config(self, socket: socket.socket):        
        logger.info('Receiving Config...')
        response, json_config = socket_receive_config(socket)

        if response != UE_RESPONSE_SUCCESS:
            raise Exception('Failed to get config...')

        config = json.loads(json_config, object_pairs_hook=OrderedDict)

        # Reduce nesting to reduce impact on rest of codebase
        for k,v in config['TrainerSettings'].items():
            config[k] = v
        
        # Setup Neural Networks
        for network_config in config['Networks']:
            network_id = int(network_config['Id'])
            network_name = network_config['Name']
            network_max_bytes = int(network_config['MaxByteNum'])

            self.network_buffers_by_id[network_id] = np.empty([network_max_bytes], dtype=np.uint8)
            self.network_info_by_id[network_id] = (network_name, network_max_bytes)

        # Setup Replay Buffers
        for replay_buffer_config in config['ReplayBuffers']:
            replay_buffer_id = int(replay_buffer_config['Id'])
            self.replay_buffers_by_id[replay_buffer_id] = self._get_replay_buffer(replay_buffer_config)

        # Setup Neural Network Device
        # TODO: move "Device" out of PPOSettings
        if "PPOSettings" in config:
            self.device = config['PPOSettings']['Device']
            self.trim_episode_start = int(config['PPOSettings']['TrimEpisodeStartStepNum'])
            self.trim_episode_end = int(config['PPOSettings']['TrimEpisodeEndStepNum'])
        else:
            self.device = config['BehaviorCloningSettings']['Device']

        return config
    
    def _send_networks(self, socket: socket.socket):
        response = UE_RESPONSE_SUCCESS
        for network_id, network in self.networks_by_id.items():
            new_response = socket_send_network(socket, network_id, network, self.networks_version, self.network_buffers_by_id[network_id], UE_SOCKET_SIGNAL_RECEIVE_NETWORK) # Send and Receive are swapped in the signal constant because they refer to what is happening on the C++ side.
            if new_response != UE_RESPONSE_SUCCESS:
                response = new_response
        return response
    
    def _receive_networks(self, socket: socket.socket):
        assert self.device is not None
        networks_by_id = {}
        for network_id in self.network_info_by_id.keys():
            logger.info(f'Receiving {self.network_info_by_id[network_id][0]}...')
            network = build_network(self.device)
            socket_receive_network(socket, network_id, network, self.network_buffers_by_id[network_id], UE_SOCKET_SIGNAL_SEND_NETWORK) # Send and Receive are swapped in the signal constant because they refer to what is happening on the C++ side.
            networks_by_id[network_id] = network
        return networks_by_id
    
    @contextmanager
    def _sending_context(self):
        self.sending = True
        try:
            yield
        finally:
            self.sending = False

    def _test_connection(self, client_socket: ClientSocket):	
        if not self.sending: # do not send ping if we are already sending networks	
            return socket_send_ping(client_socket()) == UE_RESPONSE_SUCCESS	
        return True
    
    def _get_replay_buffer(self, replay_buffer_config):
        return SocketReplayBuffer(replay_buffer_config)
            
    def _handle_connection_error(self, client_socket: ClientSocket):
        self.client_sockets.remove(client_socket) if client_socket in self.client_sockets else None

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.heartbeat_status = SOCKET_HEARTBEAT_STOPPED
        self.listener_active = False
        self.sender_active = False
        self.server_socket.close()

        with self.lock:
            for client_socket in self.client_sockets:
                client_socket().close()

        logger.info('Socket Communicator Exiting...')
        logging.shutdown()
