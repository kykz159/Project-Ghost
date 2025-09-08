# Copyright Epic Games, Inc. All Rights Reserved.

from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeColorConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleColor

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()
        
        # make the new module script for initial color
        script_asset = ueFxUtils.create_asset_data(Paths.script_color)
        script_args = ue.CreateScriptContextArgs(script_asset)
        color_script = emitter.find_or_add_module_script(
            "InitialColor",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_SPAWN)

        # get all properties from the cascade color module
        # noinspection PyTypeChecker
        (initial_color,
         initial_alpha,
         b_clamp_alpha
         ) = ueFxUtils.get_particle_module_color_props(cascade_module)

        # make inputs for the initial values
        initial_color_input = c2nUtils.create_script_input_for_distribution(
            initial_color)
        initial_alpha_input = c2nUtils.create_script_input_for_distribution(
            initial_alpha)

        # Cascade's UParticleModuleColor.bClampAlpha seems to be considered only when manipulating the graph, so the clamp settings are not carried over.
        # (If the value is set to -1, it will be clamped when Niagara is executed)

        # combine initial color and alpha into linear color
        script_asset = ueFxUtils.create_asset_data(Paths.di_color_from_vec_and_float)
        script_args = ue.CreateScriptContextArgs(script_asset)
        break_color_script = ueFxUtils.create_script_context(script_args)
        break_color_script.set_parameter("Vector (RGB)", initial_color_input)
        break_color_script.set_parameter("Float (Alpha)", initial_alpha_input)

        # set the color
        break_color_script_input = ueFxUtils.create_script_input_dynamic(
            break_color_script,
            ue.NiagaraScriptInputType.LINEAR_COLOR)
        color_script.set_parameter("Color", break_color_script_input)
