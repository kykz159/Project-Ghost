# Copyright Epic Games, Inc. All Rights Reserved.
from typing import Callable

import unreal


class UnrealEndpointManager:

    def __init__(self, *, connect_timeout: int = 120):
        """
        Creates an Unreal Endpoint Manager - used to discover Unreal clients.
        
        :param connect_timeout: Timeout (in seconds) for an individual connection attempt
        """
        self._connect_timeout = connect_timeout
        self._endpoint_manager = unreal.CaptureManagerUnrealEndpointManager()

    def wait_for_endpoint_by_host_name(
        self,
        host_name : str,
        *,
        discovery_timeout: int = 120
    ):
        """
        Blocks until a connection is made to the requested Unreal endpoint or 
        the timeout is reached. 

        :param host_name: The host name of the Unreal endpoint
        :param discovery_timeout: Timeout (in seconds)
        :raises TimeoutError
        """
        discovery_timeout_ms : int = discovery_timeout * 1_000
        success = self._endpoint_manager.wait_for_endpoint_by_host_name(
            host_name,
            discovery_timeout_ms
        )

        if not success:
            raise TimeoutError(f'Timed out waiting for endpoint: {host_name}')

    def get_endpoints(self):
        """
        Gets the list of already discovered endpoints.
        
        :return: A list of unreal.CaptureManagerUnrealEndpointInfo
        """
        return self._endpoint_manager.get_endpoints()

    def __enter__(self):
        self._endpoint_manager.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._endpoint_manager.stop()
