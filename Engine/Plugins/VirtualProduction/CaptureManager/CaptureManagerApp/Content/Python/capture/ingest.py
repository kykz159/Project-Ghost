#  Copyright Epic Games, Inc. All Rights Reserved.

import math
from datetime import datetime, timedelta
from typing import Callable, List, Optional

import unreal

from capture.devices import LiveLinkHubCaptureDevice


def is_from_today(take: unreal.LiveLinkHubTakeMetadata) -> bool:
    date_time_string = take.metadata.get_date_time_string()
    try:
        take_date = datetime.fromisoformat(date_time_string).date()
        return take_date == datetime.utcnow().date()

    except ValueError:
        unreal.log_error('Failed to parse take date as ISO format: {}'.format(date_time_string))
        return False


def is_from_yesterday(take: unreal.LiveLinkHubTakeMetadata) -> bool:
    date_time_string = take.metadata.get_date_time_string()
    try:
        take_date = datetime.fromisoformat(date_time_string).date()
        return take_date == (datetime.utcnow().date() - timedelta(days=1))

    except ValueError:
        unreal.log_error('Failed to parse take date as ISO format: {}'.format(date_time_string))
        return False


def _any_take(_: unreal.LiveLinkHubTakeMetadata) -> bool:
    return True


def _filter_takes(
    takes: List[unreal.LiveLinkHubTakeMetadata],
    take_filter: Callable[[unreal.LiveLinkHubTakeMetadata], bool]
) -> List[unreal.LiveLinkHubTakeMetadata]:
    included = []
    excluded = []

    for take in takes:
        if take_filter(take):
            included.append(take)
        else:
            excluded.append(take)

    if len(included) > 0:
        unreal.log('Takes to be ingested:')

        for take in included:
            unreal.log('  {} #{}'.format(take.metadata.slate_name, take.metadata.take_number))

    if len(excluded) > 0:
        unreal.log('Takes excluded by filter:')

        for take in excluded:
            unreal.log('  {} #{}'.format(take.metadata.slate_name, take.metadata.take_number))

    return included


def _do_ingest(
    device: LiveLinkHubCaptureDevice,
    settings: unreal.IngestCapability_Options,
    takes: List[unreal.LiveLinkHubTakeMetadata],
) -> List[unreal.LiveLinkHubTakeMetadata]:
    failed_takes = []
    max_digits = int(math.log10(len(takes))) + 1

    for take_num, take in enumerate(takes, start=1):
        unreal.log(
            'Ingesting {:{width}d}/{:{width}d}: {} #{}'.format(
                take_num,
                len(takes),
                take.metadata.slate_name,
                take.metadata.take_number,
                width=max_digits
            )
        )

        try:
            device.ingest_take(take, settings)

        except Exception as ex:
            unreal.log_error('Ingest failed: {}'.format(ex))
            failed_takes.append(take)

    return failed_takes


def _do_download(
    device: LiveLinkHubCaptureDevice,
    download_directory: str,
    takes: List[unreal.LiveLinkHubTakeMetadata],
) -> List[unreal.LiveLinkHubTakeMetadata]:
    failed_takes = []
    max_digits = int(math.log10(len(takes))) + 1

    for take_num, take in enumerate(takes, start=1):
        unreal.log(
            'Downloading {:{width}d}/{:{width}d}: {} #{}'.format(
                take_num,
                len(takes),
                take.metadata.slate_name,
                take.metadata.take_number,
                width=max_digits
            )
        )

        try:
            device.download_take(take, download_directory)

        except Exception as ex:
            # Conversion and import are currently one step, so we report "ingest failure" rather than "import failure"
            unreal.log_error('Ingest failed: {}'.format(ex))
            failed_takes.append(take)

    return failed_takes


def ingest_takes(
    device: LiveLinkHubCaptureDevice,
    conversion_settings: unreal.IngestCapability_Options,
    *,
    take_filter: Optional[Callable[[unreal.LiveLinkHubTakeMetadata], bool]] = None,
) -> List[unreal.LiveLinkHubTakeMetadata]:
    """
    :param device: Capture device instance
    :param conversion_settings: Conversion settings
    :param take_filter: Optional predicate to filter the take list. If not 
        specified all takes will be ingested.
    
    :return: List of takes which failed to ingest
    """
    unreal.log('Ingesting takes from {}'.format(device.name))

    unreal.log('Fetching take list...')
    takes = device.fetch_takes()

    if len(takes) == 0:
        unreal.log_warning('No takes found')
        return []

    if take_filter is None:
        take_filter = _any_take

    filtered_takes = _filter_takes(takes, take_filter)

    if len(filtered_takes) == 0:
        unreal.log_warning('No takes left to ingest after filtering the list')
        return []

    failed_takes = _do_ingest(device, conversion_settings, filtered_takes)

    if len(failed_takes) == 0:
        unreal.log('Ingest complete')

    else:
        unreal.log('Ingest complete, failed to ingest the following takes:')

        for failed_take in failed_takes:
            unreal.log('{} #{}'.format(failed_take.metadata.slate_name, failed_take.metadata.take_number))

    return failed_takes


def download_takes(
    device: LiveLinkHubCaptureDevice,
    download_directory: str,
    *,
    take_filter: Optional[Callable[[unreal.LiveLinkHubTakeMetadata], bool]] = None,
) -> List[unreal.LiveLinkHubTakeMetadata]:
    """
    :param device: Capture device instance
    :param conversion_settings: Conversion settings
    :param take_filter: Optional predicate to filter the take list. If not 
        specified all takes will be ingested.
    
    :return: List of takes which failed to ingest
    """
    unreal.log('Ingesting takes from {}'.format(device.name))

    unreal.log('Fetching take list...')
    takes = device.fetch_takes()

    if len(takes) == 0:
        unreal.log_warning('No takes found')
        return []

    if take_filter is None:
        take_filter = _any_take

    filtered_takes = _filter_takes(takes, take_filter)

    if len(filtered_takes) == 0:
        unreal.log_warning('No takes left to ingest after filtering the list')
        return []

    failed_takes = _do_download(device, download_directory, filtered_takes)

    if len(failed_takes) == 0:
        unreal.log('Ingest complete')

    else:
        unreal.log('Ingest complete, failed to ingest the following takes:')

        for failed_take in failed_takes:
            unreal.log('{} #{}'.format(failed_take.metadata.slate_name, failed_take.metadata.take_number))

    return failed_takes