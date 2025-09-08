# -*- coding: utf-8 -*-
'''
Copyright Epic Games, Inc. All Rights Reserved.
'''

import json
import numpy as np
import sys
import torch
import os

from ppo import PPOTrainer
from train_common import Profile, AbstractExperimentTracker, AbstractCommunicator
from train_common import save_snapshot_to_file
from train_common import UE_RESPONSE_SUCCESS, UE_RESPONSE_STOPPED

import logging
logger = logging.getLogger("LearningAgents")


def train(config, communicator:AbstractCommunicator, trackers: list[AbstractExperimentTracker] = []):
    task_directory = config['TaskDirectory']
    
    # For our default PPO implementation, we will assume exactly one obs, action, etc.
    observation_schema = config['Schemas']['Observations'][0]['Schema']
    action_schema = config['Schemas']['Actions'][0]['Schema']

    ppo_config = config['PPOSettings']
    niterations = int(ppo_config['IterationNum'])
    lr_policy = ppo_config['LearningRatePolicy']
    lr_critic = ppo_config['LearningRateCritic']
    lr_gamma = ppo_config['LearningRateDecay']
    weight_decay = ppo_config['WeightDecay']
    policy_batch_size = int(ppo_config['PolicyBatchSize'])
    critic_batch_size = int(ppo_config['CriticBatchSize'])
    policy_window = int(ppo_config['PolicyWindow'])
    iterations_per_gather = int(ppo_config['IterationsPerGather'])
    iterations_critic_warmup = int(ppo_config['CriticWarmupIterations'])
    eps_clip = ppo_config['EpsilonClip']
    action_surr_weight = ppo_config['ActionSurrogateWeight']
    action_reg_weight = ppo_config['ActionRegularizationWeight']
    action_ent_weight = ppo_config['ActionEntropyWeight']
    return_reg_weight = ppo_config['ReturnRegularizationWeight']
    gae_lambda = ppo_config['GaeLambda']
    advantage_normalization = ppo_config['AdvantageNormalization']
    advantage_min = ppo_config['AdvantageMin']
    advantage_max = ppo_config['AdvantageMax']
    use_grad_norm_max_clipping = ppo_config['UseGradNormMaxClipping']
    grad_norm_max = ppo_config['GradNormMax']
    trim_episode_start = int(ppo_config['TrimEpisodeStartStepNum'])
    trim_episode_end = int(ppo_config['TrimEpisodeEndStepNum'])
    seed = int(ppo_config['Seed'])
    discount_factor = ppo_config['DiscountFactor']
    save_snapshots = ppo_config['SaveSnapshots']

    logger.info(json.dumps(config, indent=4))

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    # Receive initial policy
    policy_network_id = 0
    logger.info('Receiving Policy...')
    policy_network = communicator.get_network(policy_network_id)

    # Receive initial critic
    critic_network_id = 1
    logger.info('Receiving Critic...')
    critic_network = communicator.get_network(critic_network_id)
    
    # Receive initial encoder
    encoder_network_id = 2
    logger.info('Receiving Encoder...')
    encoder_network = communicator.get_network(encoder_network_id)

    # Receive initial decoder
    decoder_network_id = 3
    logger.info('Receiving Decoder...')
    decoder_network = communicator.get_network(decoder_network_id)

    # Create Optimizer

    optimizer_policy = torch.optim.AdamW(
        list(policy_network.parameters()) + 
        list(encoder_network.parameters()) +
        list(decoder_network.parameters()),
        lr=lr_policy,
        amsgrad=True,
        weight_decay=weight_decay)

    optimizer_critic = torch.optim.AdamW(
        list(critic_network.parameters()) +
        list(encoder_network.parameters()),
        lr=lr_critic,
        amsgrad=True,
        weight_decay=weight_decay)

    scheduler_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer_policy, gamma=lr_gamma)
    scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(optimizer_critic, gamma=lr_gamma)

    # Create PPO Policy
    logger.info('Creating PPO Policy...')

    ppo_trainer = PPOTrainer(
        observation_schema,
        action_schema,
        policy_network,
        critic_network,
        encoder_network,
        decoder_network,
        optimizer_policy,
        optimizer_critic,
        discount_factor=discount_factor,
        gae_lambda=gae_lambda,
        eps_clip=eps_clip,
        advantage_normalization=advantage_normalization,
        advantage_min=advantage_min,
        advantage_max=advantage_max,
        use_grad_norm_max_clipping=use_grad_norm_max_clipping,
        grad_norm_max=grad_norm_max,
        action_surr_weight=action_surr_weight,
        action_reg_weight=action_reg_weight,
        action_ent_weight=action_ent_weight,
        return_reg_weight=return_reg_weight)

    for tracker in trackers:
        tracker.initialize_tracker()
    
    # Training Loop
    logger.info('Begin Training...')

    ti = 0
    replay_buffer_id = 0

    while True:

        # Pull Experience
        with Profile('Pull Experience'):
            response, buffer, exp_stats = communicator.receive_experience(
                replay_buffer_id,
                trim_episode_start, 
                trim_episode_end)
            
        if response == UE_RESPONSE_STOPPED:
            break
        else:
            assert response == UE_RESPONSE_SUCCESS
            avg_reward = exp_stats['experience/avg_reward'][0]
            avg_reward_sum = exp_stats['experience/avg_reward_sum'][0]
            avg_episode_length = exp_stats['experience/avg_episode_length']

        # Check for completion
        if ti >= niterations:
            response = communicator.send_complete()
            assert response == UE_RESPONSE_SUCCESS
            break
        
        # Buffer Manipulation
        # Our PPO implementation expects one of each of the data buffers
        buffer['obs'] = buffer['obs'][0]
        buffer['obs_next'] = buffer['obs_next'][0]
        buffer['act'] = buffer['act'][0]
        buffer['mod'] = buffer['mod'][0]
        buffer['mem'] = buffer['mem'][0]
        buffer['mem_next'] = buffer['mem_next'][0]
        buffer['rew'] = buffer['rew'][0].squeeze()

        # Train
        with Profile('Training'):
            
            if ti == 0 and iterations_critic_warmup > 0:
                ppo_trainer.warmup_critic(
                    buffer=buffer,
                    critic_batch_size=critic_batch_size,
                    recompute_returns_iterations=iterations_critic_warmup,
                    update_func=lambda: communicator.send_ping())
            
            stats = ppo_trainer.train(
                buffer=buffer, 
                policy_batch_size=policy_batch_size,
                critic_batch_size=critic_batch_size,
                iterations=iterations_per_gather, 
                policy_window=policy_window,
                update_func=lambda: communicator.send_ping())
                
            avg_return = stats['experience/avg_return']

        networks_version = ti

        logger.info(f"Setting latest networks in communicator... IDs: [{policy_network_id}, {critic_network_id}, {encoder_network_id}, {decoder_network_id}] | Version: {networks_version}")
        communicator.set_network(policy_network_id, policy_network, networks_version)
        communicator.set_network(critic_network_id, critic_network, networks_version)
        communicator.set_network(encoder_network_id, encoder_network, networks_version)
        communicator.set_network(decoder_network_id, decoder_network, networks_version)

        # Log stats

        with Profile('Logging'):

            logger.info('\rIter: %7i | Avg Rew: %7.5f | Avg Rew Sum: %7.5f | Avg Return: %7.5f | Avg Episode Len: %7.5f | Batch Size: %d' % 
                (ti, avg_reward, avg_reward_sum, avg_return, avg_episode_length, communicator.get_batch_size()))
            sys.stdout.flush()
            
            for tracker in trackers:
                tracked_stats = [
                    ('experience/avg_reward', avg_reward, ti),
                    ('experience/avg_reward_sum', avg_reward_sum, ti),
                    ('experience/avg_return', avg_return, ti),
                    ('experience/avg_episode_length', avg_episode_length, ti)
                ]
                tracker.track(tracked_stats)                

            for bi in range(len(stats['loss/policy'])):
                
                for tracker in trackers:
                    tracked_stats = [
                        ('grads/encoder', stats['grads/encoder'][bi], ti),
                        ('grads/decoder', stats['grads/decoder'][bi], ti),
                        ('grads/policy', stats['grads/policy'][bi], ti),
                        ('grads/critic', stats['grads/critic'][bi], ti),
                        ('loss/critic_ret', stats['loss/critic_ret'][bi], ti),
                        ('loss/critic_reg', stats['loss/critic_reg'][bi], ti),
                        ('loss/critic', stats['loss/critic'][bi], ti),
                        ('loss/policy_surr', stats['loss/policy_surr'][bi], ti),
                        ('loss/policy_reg', stats['loss/policy_reg'][bi], ti),
                        ('loss/policy_ent', stats['loss/policy_ent'][bi], ti),
                        ('loss/policy', stats['loss/policy'][bi], ti)
                    ]
                    tracker.track(tracked_stats)
                
                # Write Snapshot
                
                if save_snapshots and ti % 1000 == 0:
                    save_path = os.path.join(task_directory, "Snapshots")

                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    
                    policy_snapshot = os.path.join(save_path, 'policy_' + str(ti) + '.bin')
                    save_snapshot_to_file(policy_network, policy_snapshot)
                    logger.info('Saved Policy Snapshot to: "%s"' % policy_snapshot)

                    critic_snapshot = os.path.join(save_path, 'critic_' + str(ti) + '.bin')
                    save_snapshot_to_file(critic_network, critic_snapshot)
                    logger.info('Saved Critic Snapshot to: "%s"' % critic_snapshot)

                    encoder_snapshot = os.path.join(save_path, 'encoder_' + str(ti) + '.bin')
                    save_snapshot_to_file(encoder_network, encoder_snapshot)
                    logger.info('Saved Encoder Snapshot to: "%s"' % encoder_snapshot)

                    decoder_snapshot = os.path.join(save_path, 'decoder_' + str(ti) + '.bin')
                    save_snapshot_to_file(decoder_network, decoder_snapshot)
                    logger.info('Saved Decoder Snapshot to: "%s"' % decoder_snapshot)

                    for tracker in trackers:
                        snapshots = [policy_snapshot, critic_snapshot, encoder_snapshot, decoder_snapshot]
                        tracker.track_snapshots(snapshots)
                    
                ti += 1
                
                # Update lr schedulers
                
                if ti % 1000 == 0:
                    scheduler_policy.step()
                    scheduler_critic.step()
                    
    for tracker in trackers:
        tracker.close()

    logger.info("Done!")
