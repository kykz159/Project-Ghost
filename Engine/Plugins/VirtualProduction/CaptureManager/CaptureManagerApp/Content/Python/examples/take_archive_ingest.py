#  Copyright Epic Games, Inc. All Rights Reserved.

import argparse
import os
import socket
import sys
import tempfile

import unreal

from capture.devices import TakeArchiveIngestDevice
from capture.ingest import ingest_takes
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
    parser = argparse.ArgumentParser(description='Example Take Archive Ingest device')
    parser.add_argument('--archive-path', required=True, type=str, help='Path to the take archive')
    args = parser.parse_args()

    print_diagnostic_info()

    # Just connect to a local client for this example
    unreal_hostname = socket.gethostname()

    # Start up the Unreal endpoint manager so we can discover and connect to unreal clients
    with UnrealEndpointManager(connect_timeout=10) as endpoint_manager:
        # Wait here until we have discovered the unreal client (or the discovery times out)
        endpoint_manager.wait_for_endpoint_by_host_name(
            unreal_hostname,
            discovery_timeout=10
        )

        unreal.log(f'Discovered {unreal_hostname}')

        settings = unreal.IngestCapability_Options()
        settings.working_directory = os.path.join(tempfile.gettempdir(), 'ScriptedIngestConversion')
        settings.audio.format = 'wav'
        settings.audio.file_name_prefix = 'audio'
        settings.video.format = 'jpeg'
        settings.video.file_name_prefix = 'frame'
        settings.upload_host_name = unreal_hostname

        with TakeArchiveIngestDevice(args.archive_path) as archive:
            ingest_takes(archive, settings)


if __name__ == '__main__':
    main()
