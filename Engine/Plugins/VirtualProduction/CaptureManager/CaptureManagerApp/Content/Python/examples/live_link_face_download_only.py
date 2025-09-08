#  Copyright Epic Games, Inc. All Rights Reserved.

import argparse
import os
import socket
import sys
import tempfile

import unreal

from capture.devices import LiveLinkFaceDevice
from capture.ingest import download_takes
from capture.unreal_endpoint_manager import UnrealEndpointManager


def print_diagnostic_info():
    banner_width = 120

    unreal.log('-' * banner_width)
    unreal.log('Python diagnostic information')
    unreal.log('-' * banner_width)

    unreal.log('Script: {}'.format(sys.argv[0]))
    unreal.log('Executing from {}'.format(os.getcwd()))
    unreal.log('Module search paths:')

    for path in sys.path:
        unreal.log(os.path.abspath(path))

    unreal.log('-' * banner_width)


def main():
    parser = argparse.ArgumentParser(description='Example Live Link Face device download', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ip-address', required=True, type=str, help='Address of the Live Link Face device')
    parser.add_argument('--port', type=int, default=14785, help='Port used by the Live Link Face device')
    args = parser.parse_args()

    print_diagnostic_info()

    with LiveLinkFaceDevice(args.ip_address, args.port, start_timeout=10) as app:
        download_directory = os.path.join(tempfile.gettempdir(), 'ScriptedIngestDownload')
        download_takes(app, download_directory)
        

if __name__ == '__main__':
    main()
