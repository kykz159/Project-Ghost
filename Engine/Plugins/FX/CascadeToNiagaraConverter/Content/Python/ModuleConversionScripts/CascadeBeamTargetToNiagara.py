# Copyright Epic Games, Inc. All Rights Reserved.

from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeBeamTargetConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleBeamTarget

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        (target_distribution,
         b_target_absolute
         ) = ueFxUtils.get_particle_module_beam_target_props(cascade_module)

        source_distribution = None
        source_module = args.find_cascade_module(ue.ParticleModuleBeamSource)
        if source_module:
            source_distribution = ueFxUtils.get_particle_module_beam_source_props(source_module)

        script_asset = ueFxUtils.create_asset_data(Paths.script_beam_emitter_setup)
        script_args = ue.CreateScriptContextArgs(script_asset)
        beam_emitter_setup_script = emitter.find_or_add_module_script(
            "BeamEmitterSetup", 
            script_args,
            ue.ScriptExecutionCategory.EMITTER_UPDATE)

        target_input = c2nUtils.create_script_input_for_distribution(target_distribution)

        script_asset = ueFxUtils.create_asset_data(Paths.di_vec_to_pos)
        script_args = ue.CreateScriptContextArgs(script_asset)
        vec_to_pos_script = ueFxUtils.create_script_context(script_args)
        vec_to_pos_script.set_parameter('Input Position', target_input)
        vec_to_pos_input = ueFxUtils.create_script_input_dynamic(
            vec_to_pos_script,
            ue.NiagaraScriptInputType.POSITION)

        beam_emitter_setup_script.set_parameter("Beam End", vec_to_pos_input)
        beam_emitter_setup_script.set_parameter("Absolute Beam End", ueFxUtils.create_script_input_bool(b_target_absolute))
