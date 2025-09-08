# Copyright Epic Games, Inc. All Rights Reserved.

# This module contains utility functions that could be used by other scripts in MetaHuman plugin content folder

import unreal
import os
import shutil

content_directory = unreal.SystemLibrary.get_project_content_directory()
asset_tools = unreal.AssetToolsHelpers.get_asset_tools()

def create_or_recreate_asset(in_asset_name, in_package_path, in_asset_class, in_factory):
    """Helper for creating a new asset and overwriting in acse it already exists"""
    path_to_asset = os.path.join(in_package_path, in_asset_name)

    if unreal.EditorAssetLibrary.does_asset_exist(path_to_asset):
        print('Deleting an existing asset before creating a new one with the same name')
        unreal.EditorAssetLibrary.delete_asset(path_to_asset)

    created_asset = asset_tools.create_asset(asset_name=in_asset_name, package_path=in_package_path, 
                                           asset_class=in_asset_class, factory=in_factory)

    if created_asset is None:
        unreal.log_error("Error creating asset '{0}'".format(path_to_asset))
        exit()

    return created_asset

def get_or_create_asset(in_asset_name, in_package_path, in_asset_class, in_factory):
    """Returns an existing asset or returns a newly created one with the input path, class & name"""
    asset_subsystem = unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)
    path_to_asset = os.path.join(in_package_path, in_asset_name)

    asset = None
    if unreal.EditorAssetLibrary.does_asset_exist(path_to_asset):
        asset = asset_subsystem.load_asset(asset_path=path_to_asset)

    if asset is None:
        asset = asset_tools.create_asset(asset_name=in_asset_name, package_path=in_package_path,
                                         asset_class=in_asset_class, factory=in_factory)
    if asset is None:
        unreal.log_error("Error creating asset '{0}'".format(path_to_asset))
        exit()

    return asset


def run_data_cleanup(relative_data_path : str):
    """
    A clean-up function for MetaHuman animator pipeline. During the import process a set 
    of assets as well as image and depth data is generated. This function cleans up 
    the assets and other data created during the MetaHuman flow.
    Input path to this function is a relative path e.g. "/Game/MHA-Data"
    """
    # Need to manually delete assets first as not all data is imported into UE
    assets_for_deletion = unreal.EditorAssetLibrary.list_assets(relative_data_path)
    unreal.EditorAssetLibrary.delete_loaded_assets(assets_for_deletion)

    # Delete folder on disk to clear image and depth data after loaded assets have been removed
    storage_location = relative_data_path.replace('/Game', '')
    path_to_data = content_directory + '/' + storage_location
    if os.path.exists(path_to_data):
        shutil.rmtree(path_to_data)
