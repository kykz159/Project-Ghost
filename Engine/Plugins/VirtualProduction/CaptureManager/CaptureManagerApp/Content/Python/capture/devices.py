#  Copyright Epic Games, Inc. All Rights Reserved.

import typing
from itertools import count

import unreal

_DEFAULT_ARCHIVE_TIMEOUT = 5
_DEFAULT_NETWORK_TIMEOUT = 30


class LiveLinkHubCaptureDevice:
    """ Class for all capture devices """

    def __init__(self, device: unreal.LiveLinkHubCaptureDevice, start_timeout: int):
        self._device = device
        self._start_timeout = start_timeout

    @property
    def name(self):
        return self._device.name

    def fetch_takes(self) -> typing.List[unreal.LiveLinkHubTakeMetadata]:
        """ Fetch a list of the available takes from the device """
        result = self._device.fetch_takes()
        if result.status.is_error():
            raise RuntimeError(result.status.message)
        return result.takes

    def ingest_take(self, take: unreal.LiveLinkHubTakeMetadata, conversion_settings: unreal.IngestCapability_Options):
        """ Ingest the specified take to the ingest client """
        status = self._device.ingest_take(take, conversion_settings)
        if status.is_error():
            raise RuntimeError(status.message)
        
    def download_take(self, take: unreal.LiveLinkHubTakeMetadata, download_directory: str):
        """ Download the specified take """
        status = self._device.download_take(take, download_directory)
        if status.is_error():
            raise RuntimeError(status.message)

    def __enter__(self):
        status = self._device.start(self._start_timeout)
        if status.is_error():
            raise RuntimeError(status.message)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = self._device.stop()
        if status.is_error():
            raise RuntimeError(status.message)


class LiveLinkHubNetworkCaptureDevice(LiveLinkHubCaptureDevice):
    """ A Live Link Face device """

    _num_instances_by_class = {}

    # noinspection PyMethodOverriding
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._num_instances_by_class[cls] = count(1)

    @classmethod
    def _default_name(cls):
        return '{}_{}'.format(cls.__name__, next(cls._num_instances_by_class[cls]))

    def __init__(self, ip_address: str, port: int, *, name: str = None, start_timeout: int = _DEFAULT_NETWORK_TIMEOUT):
        """
        :param ip_address: IP address of the device
        :param port: Port the device is using
        :param name: Device name
        :param start_timeout: Start timeout (in seconds)
        """
        
        device_name = name if name is not None else self._default_name()

        settings = unreal.LiveLinkFaceDeviceSettings()
        settings.display_name = device_name
        settings.ip_address.ip_address_string = ip_address
        settings.port = port

        factory = unreal.LiveLinkHubCaptureDeviceFactory()
        device = factory.create_device_by_class(device_name, unreal.LiveLinkFaceDevice.static_class(), settings)

        super().__init__(device, start_timeout)


class LiveLinkHubArchiveCaptureDevice(LiveLinkHubCaptureDevice):
    """ A take archive ingest device """

    _num_instances_by_class = {}

    # noinspection PyMethodOverriding
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._num_instances_by_class[cls] = count(1)

    @classmethod
    def _default_name(cls):
        return '{}_{}'.format(cls.__name__, next(cls._num_instances_by_class[cls]))

    def __init__(self, archive_path: str, *, name: str = None, start_timeout: int = _DEFAULT_ARCHIVE_TIMEOUT):
        """
        :param archive_path: Path to the archive
        :param name: Device name
        :param start_timeout: Start timeout (in seconds)
        """
        device_name = name if name is not None else self._default_name()

        settings = unreal.TakeArchiveIngestDeviceSettings()
        settings.display_name = device_name
        settings.take_directory.path = archive_path

        factory = unreal.LiveLinkHubCaptureDeviceFactory()
        device = factory.create_device_by_class(device_name, unreal.TakeArchiveIngestDevice.static_class(), settings)

        super().__init__(device, start_timeout)


class MonoVideoIngestDeviceBase(LiveLinkHubCaptureDevice):
    """ An mono video ingest device """

    _num_instances_by_class = {}

    # noinspection PyMethodOverriding
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._num_instances_by_class[cls] = count(1)

    @classmethod
    def _default_name(cls):
        return '{}_{}'.format(cls.__name__, next(cls._num_instances_by_class[cls]))

    def __init__(self, archive_path: str, video_discovery_expression: str, *, name: str = None, start_timeout: int = _DEFAULT_ARCHIVE_TIMEOUT):
        """
        :param archive_path: Path to the mono video
        :param video_discovery_expression: Video file discovery expression
        :param name: Device name
        :param start_timeout: Start timeout (in seconds)
        """
        device_name = name if name is not None else self._default_name()

        settings = unreal.MonoVideoIngestDeviceSettings()
        settings.display_name = device_name
        settings.video_discovery_expression.value = video_discovery_expression
        settings.take_directory.path = archive_path

        factory = unreal.LiveLinkHubCaptureDeviceFactory()
        device = factory.create_device_by_class(device_name, unreal.MonoVideoIngestDevice.static_class(), settings)

        super().__init__(device, start_timeout)


class TakeArchiveIngestDevice(LiveLinkHubArchiveCaptureDevice):
    """ A convenience wrapper for Take Archive Ingest devices """

    def __init__(self, archive_path: str, *, name: str = None, start_timeout: int = _DEFAULT_ARCHIVE_TIMEOUT):
        """
        :param archive_path: Path to the take archive
        :param name: Device name
        :param start_timeout: Start timeout (in seconds)
        """
        super().__init__(archive_path, name=name, start_timeout=start_timeout)


class LiveLinkFaceDevice(LiveLinkHubNetworkCaptureDevice):
    """ A convenience wrapper for Generic CPS (Capture Protocol Stack) devices """

    def __init__(self, ip_address: str, port: int, *, name: str = None, start_timeout: int = _DEFAULT_NETWORK_TIMEOUT):
        """
        :param ip_address: IP address of the device
        :param port: Port the device is using
        :param name: Device name
        :param start_timeout: Start timeout (in seconds)
        """
        super().__init__(ip_address, port, name=name, start_timeout=start_timeout)


class MonoVideoIngestDevice(MonoVideoIngestDeviceBase):
    """ A convenience wrapper for Mono Video Capture devices """

    def __init__(self, archive_path: str, video_discovery_expression: str, *, name: str = None, start_timeout: int = _DEFAULT_ARCHIVE_TIMEOUT):
        """
        :param archive_path: Path to the archive
        :param video_discovery_expression: Video file discovery expression
        :param name: Device name
        :param start_timeout: Start timeout (in seconds)
        """
        super().__init__(archive_path, video_discovery_expression, name=name, start_timeout=start_timeout)