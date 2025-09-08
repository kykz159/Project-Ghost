# Copyright Epic Games, Inc. All Rights Reserved.

from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeInitialLocationConverter(ModuleConverterInterface):
    call_count = 0

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleLocation

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # choose the correct niagara module depending on the converted renderer

        # get all properties from the cascade initial location module
        # noinspection PyTypeChecker
        (start_location_distribution,
         distribute_over_n_points,
         distribute_threshold
         ) = ueFxUtils.get_particle_module_location_props(cascade_module)

        #  todo implement "choose only n particles" dynamic input

        # if distribute over n points is not 0 or 1, special case handle the start location distribution to be over an 
        # equispaced range.
        if distribute_over_n_points != 0.0 and distribute_over_n_points != 1.0:
            range_n_input = c2nUtils.create_script_input_random_range(0.0, distribute_over_n_points)
            n_input = ueFxUtils.create_script_input_int(distribute_over_n_points)

            script_asset = ueFxUtils.create_asset_data(Paths.di_divide_float)
            script_args = ue.CreateScriptContextArgs(script_asset)
            div_float_script = ueFxUtils.create_script_context(script_args)
            div_float_script.set_parameter("A", range_n_input)
            div_float_script.set_parameter("B", n_input)
            indexer_input = ueFxUtils.create_script_input_dynamic(div_float_script, ue.NiagaraScriptInputType.FLOAT)

            options = c2nUtils.DistributionConversionOptions()
            options.set_custom_indexer(indexer_input)
            position_input = c2nUtils.create_script_input_for_distribution(start_location_distribution, options)
        else:
            indexer_input = ueFxUtils.create_script_input_linked_parameter(
                "Emitter.LoopedAge",
                ue.NiagaraScriptInputType.FLOAT)
            options = c2nUtils.DistributionConversionOptions()
            options.set_custom_indexer(indexer_input)
            position_input = c2nUtils.create_script_input_for_distribution(start_location_distribution, options)

        script_asset = ueFxUtils.create_asset_data(Paths.di_transform_vector)
        script_args = ue.CreateScriptContextArgs(script_asset)
        di_transform_vector_script = ueFxUtils.create_script_context(script_args)
        di_transform_vector_script.set_parameter('Vector', position_input)
        di_transform_vector_input = ueFxUtils.create_script_input_dynamic(
            di_transform_vector_script,
            ue.NiagaraScriptInputType.VEC3)

        script_asset = ueFxUtils.create_asset_data(Paths.script_add_vector_to_position)
        script_args = ue.CreateScriptContextArgs(script_asset)
        add_vector_to_position_script = emitter.find_or_add_module_script(
            "AddVectorToPosition_Local" + str(cls.call_count),
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_SPAWN)
        add_vector_to_position_script.set_parameter("Vector", di_transform_vector_input)

        cls.call_count += 1
