import os
import argparse
import sys

import unreal

def get_asset(package_path: str) -> unreal.Object:
    asset_subsystem = unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)

    asset = None
    if unreal.EditorAssetLibrary.does_asset_exist(package_path):
        asset = asset_subsystem.load_asset(asset_path=package_path)

    return asset

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
    parser = argparse.ArgumentParser(description='Depth Generator Example')
    parser.add_argument('--cd-package-path', required=True, type=str, help='Path to the footage capture data asset')
    args = parser.parse_args()

    print_diagnostic_info()

    depth_options = unreal.MetaHumanGenerateDepthWindowOptions()
    depth_options.asset_name = 'depth_asset_name'
    depth_options.package_path.path = '/Game/DepthGenerator/'
    depth_options.image_sequence_root_path.path = os.path.join(unreal.Paths.project_content_dir(), 'DepthGenerator/DepthImageSequence/')
    # depth_options.reference_camera_calibration = None # Will use calibration from the Footage Capture Data asset (set by default)
    # depth_options.generated_camera_calibration_suffix = '_Generated' # Set by default
    # depth_options.depth_precision = MetaHumanCaptureDepthPrecisionType.EIGHTIETH # Set by default
    # depth_options.depth_resolution = MetaHumanCaptureDepthResolutionType.FULL # Set by default
    # depth_options.min_distance = 10.0 # Set by default
    # depth_options.max_distance = 25.0 # Set by default
    # depth_options.should_compress_depth_files = true # Set by default 

    capture_data_asset = get_asset(args.cd_package_path)
    if capture_data_asset is None:
        raise RuntimeError(f'Capture data asset not found : {args.cd_package_path}')
    
    depth_generator = unreal.MetaHumanDepthGenerator()
    success = depth_generator.process(capture_data_asset, depth_options)

    if not success:
        raise RuntimeError('Failed to generate depth')
    
if __name__ == '__main__':
    main()

