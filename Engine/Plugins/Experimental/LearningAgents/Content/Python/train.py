# -*- coding: utf-8 -*-
'''
Copyright Epic Games, Inc. All Rights Reserved.
'''

import argparse
import json
import logging
import os
import sys
from importlib import import_module

from train_common import create_task_dir, get_experiment_trackers

logger = logging.getLogger("LearningAgents")


def save_config(config):
    config_dir = os.path.join(config['TaskDirectory'], 'Configs')
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    with open(os.path.join(config_dir, '%s_%s_%s_%s.json' % (
        config['TaskName'], 
        config['TrainerMethod'], 
        config['CommunicationType'], 
        config['TimeStamp'])), 'w') as f:
        
        json.dump(config, f, indent=4)


if __name__ == '__main__':

    # Set the logging immediately in case anything bad happens we can log the error
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")
    
    parser=argparse.ArgumentParser(description="Learning Agents Training Process")
    parser.add_argument('task_name', help='The name of this training task. Use a unique name to disambiguate results.')
    parser.add_argument('-p', '--trainer-path', help='Optional path where custom trainer files can be found')
    parser.add_argument('-m', '--trainer-module', help='Optional module where custom trainer can be found', default='train_ppo')
    parser.add_argument('-l', '--log', action='store_true', help='Should verbose logging be enabled? Overrides the config.')
    parser.add_argument('--nne-cpu-path', help='Optional path where NNE Runtime Basic CPU can be found.')
    subparsers = parser.add_subparsers(dest='communication_type')
    
    # Shared Memory Parser
    parser_shared_mem = subparsers.add_parser('SharedMemory', help='Train using the shared memory communications protocol')
    parser_shared_mem.add_argument('task_dir', help='The absolute path to the task\'s working directory. Should be a base directory if --make-task-dir if true.')
    parser_shared_mem.add_argument('-d', '--make-task-dir', action='store_true', help='If true, then we need to make the task dir and config dir.')
    parser_shared_mem.add_argument('-n', '--num-processes', type=int, default=1, help='Optional number of game processes that will connect')
    parser_shared_mem.add_argument('-g', '--controls-guid', help='Optional GUID for the shared memory controls region. If not provided, then this trainer will allocate process_num amount of control memory arrays.')
    parser_shared_mem.add_argument('-c', '--create-mem', action='store_true', help='If true, then this process will create the controls memory. Otherwise, we will map the existing memory.')

    # Socket Parser
    parser_socket = subparsers.add_parser('Socket', help='Train using the socket communications protocol')
    parser_socket.add_argument('address', help='The address for the socket server to listen on')
    parser_socket.add_argument('temp_directory', help='Temporary directory where this process can create folders and save files')
    parser_socket.add_argument('-n', '--num-processes', type=int, default=1, help='Optional number of game processes that will connect')
    parser_socket.add_argument('--min-batch-size', type=int, default=1, help='Minimum batch size to fill before a training step')
    parser_socket.add_argument('--max-batch-size', type=int, default=256, help='Maximum batch size to fill before a training step')

    args=parser.parse_args()

    if args.nne_cpu_path:
        sys.path.append(args.nne_cpu_path)
    else:
        sys.path.append(os.path.dirname(__file__) + '/../../../NNERuntimeBasicCpu/Content/Python/')
    
    if args.trainer_path:
        sys.path.append(args.trainer_path)
    
    # Import the trainer module and find the train function
    module = import_module(args.trainer_module)
    train = getattr(module, 'train')

    if args.log:
        logger.setLevel(logging.INFO)
    
    if args.communication_type == 'SharedMemory':
        
        from communicators.shared_memory_communicator import SharedMemoryCommunicator
        communicator = SharedMemoryCommunicator(args.controls_guid, args.num_processes, args.task_dir, args.task_name, args.create_mem, args.make_task_dir)
        
        config = communicator.shared_memory_processes[0].config
        config['TaskName'] = args.task_name
        config["TaskDirectory"] = communicator.task_dir
        config['CommunicationType'] = args.communication_type

        train(communicator.shared_memory_processes[0].config, communicator, get_experiment_trackers(config))

        logger.info('Exiting...')
        logging.shutdown()
        
    elif args.communication_type == 'Socket':  

        from communicators.socket_communicator import SocketCommunicator
        logger.info('Starting Socket Communicator...')
                
        config = dict()
        task_dir, task_id = create_task_dir(args.temp_directory, args.task_name)
        config['TaskName'] = args.task_name
        config['TaskDirectory'] = task_dir
        config['TempDirectory'] = args.temp_directory
        config['CommunicationType'] = args.communication_type

        config['MinBatchSize'] = args.min_batch_size
        config['MaxBatchSize'] = args.max_batch_size

        host, port = args.address.split(':')
        port = int(port)
        logger.info('Creating Socket Trainer Server (%s:%i)...' % (host, port))

        with SocketCommunicator(host, port, config['MinBatchSize'], config['MaxBatchSize']) as communicator:
            if communicator.start_server():
                config.update(communicator.get_config())
                save_config(config)
                train(config, communicator, get_experiment_trackers(config))        
    else:
        raise Exception('Unknown Communicator Type %s' % args.communication_type)
