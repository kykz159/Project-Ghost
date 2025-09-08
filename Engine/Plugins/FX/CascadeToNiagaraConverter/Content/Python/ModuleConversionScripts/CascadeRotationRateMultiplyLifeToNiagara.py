# Copyright Epic Games, Inc. All Rights Reserved.

from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeRotationRateMultiplyLifeConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleRotationRateMultiplyLife

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        rotation_rate_module = args.find_cascade_module(ue.ParticleModuleRotationRate)
        if rotation_rate_module:
            # CascadeRotationRateToNiagaraRotationRate.pyで処理を行うので、ここでは何もしない。
            pass
        else:
            life_multiplier = ueFxUtils.get_particle_module_rotation_rate_multiply_life_props(cascade_module)

            script_asset = ueFxUtils.create_asset_data(Paths.script_sprite_rotation_rate)
            script_args = ue.CreateScriptContextArgs(script_asset)
            rotation_rate_script = emitter.find_or_add_module_script(
                "SpriteRotationRate",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_UPDATE)

            options = c2nUtils.DistributionConversionOptions()
            options.set_target_type_width(ue.NiagaraScriptInputType.FLOAT)
            life_multiplier_input = c2nUtils.create_script_input_for_distribution(life_multiplier, options)

            script_asset = ueFxUtils.create_asset_data(Paths.di_angle_conversion)
            script_args = ue.CreateScriptContextArgs(script_asset)
            norm_angle_to_degrees_script = ueFxUtils.create_script_context(script_args)
            norm_angle_to_degrees_script.set_parameter(
                "Angle Input",
                ueFxUtils.create_script_input_enum(Paths.enum_niagara_angle_input, "Normalized Angle (0-1)"))
            norm_angle_to_degrees_script.set_parameter(
                "Angle Output",
                ueFxUtils.create_script_input_enum(Paths.enum_niagara_angle_input, "Degrees"))
            norm_angle_to_degrees_script.set_parameter("Angle", life_multiplier_input)
            rotation_from_spawn_input = ueFxUtils.create_script_input_dynamic(
                norm_angle_to_degrees_script,
                ue.NiagaraScriptInputType.FLOAT)

            rotation_rate_script.set_parameter("Rotation Rate", rotation_from_spawn_input)
