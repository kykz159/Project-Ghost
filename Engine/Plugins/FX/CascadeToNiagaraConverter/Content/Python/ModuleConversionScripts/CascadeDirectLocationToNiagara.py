# Copyright Epic Games, Inc. All Rights Reserved.

from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeDirectLocationConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleLocationDirect

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # get all properties from the cascade direct location module
        # noinspection PyTypeChecker
        (location_distribution,
         location_offset_distribution,
         scale_factor_distribution
         ) = ueFxUtils.get_particle_module_location_direct_props(cascade_module)

        # create an input for the location.
        location_input = c2nUtils.create_script_input_for_distribution(location_distribution)

        # if the location offset is non zero, add it to the location.
        if c2nUtils.distribution_always_equals(location_offset_distribution, 0.0) is False:
            script_asset = ueFxUtils.create_asset_data(Paths.di_add_vector)
            script_args = ue.CreateScriptContextArgs(script_asset)
            add_vector_script = ueFxUtils.create_script_context(script_args)
            location_offset_input = c2nUtils.create_script_input_for_distribution(location_offset_distribution)
            add_vector_script.set_parameter("A", location_input)
            add_vector_script.set_parameter("B", location_offset_input)
            location_input = ueFxUtils.create_script_input_dynamic(
                add_vector_script,
                ue.NiagaraScriptInputType.VEC3)

        # if the scale factor distribution is not equal to 1, multiply the 
        # location input.
        if c2nUtils.distribution_always_equals(scale_factor_distribution, 1.0) is False:
            script_asset = ueFxUtils.create_asset_data(Paths.di_multiply_vector)
            script_args = ue.CreateScriptContextArgs(script_asset)
            multiply_vector_script = ueFxUtils.create_script_context(script_args)
            scale_factor_input = c2nUtils.create_script_input_for_distribution(scale_factor_distribution)
            multiply_vector_script.set_parameter("A", location_input)
            multiply_vector_script.set_parameter("B", scale_factor_input)
            location_input = ueFxUtils.create_script_input_dynamic(
                multiply_vector_script,
                ue.NiagaraScriptInputType.VEC3)

        script_asset = ueFxUtils.create_asset_data(Paths.di_vec_to_pos)
        script_args = ue.CreateScriptContextArgs(script_asset)
        vec_to_pos_script = ueFxUtils.create_script_context(script_args)
        vec_to_pos_script.set_parameter('Input Position', location_input)
        vec_to_pos_input = ueFxUtils.create_script_input_dynamic(
            vec_to_pos_script,
            ue.NiagaraScriptInputType.POSITION)

        script_asset = ueFxUtils.create_asset_data(Paths.di_transform_position)
        script_args = ue.CreateScriptContextArgs(script_asset)
        transform_position_script = ueFxUtils.create_script_context(script_args)
        transform_position_script.set_parameter('Position', vec_to_pos_input)
        transform_position_input = ueFxUtils.create_script_input_dynamic(
            transform_position_script,
            ue.NiagaraScriptInputType.POSITION)

        emitter.set_parameter_directly(
            "Particles.Position",
            transform_position_input,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)
