# Copyright Epic Games, Inc. All Rights Reserved.

from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils

import Paths
from ModuleConversionScripts import CascadeMeshRotationRateToNiagara

class CascadeMeshRotationRateMultiplyLifeConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleMeshRotationRateMultiplyLife

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()
        
        # get all properties of the mesh rotation rate muliply life module.
        # noinspection PyTypeChecker
        life_multiplier_distribution = ueFxUtils.get_particle_module_mesh_rotation_rate_multiply_life_props(
            cascade_module)

        # create an input for the rotation rate scale.
        life_multiplier_input = c2nUtils.create_script_input_for_distribution(life_multiplier_distribution)

        #  todo handle rotation rate as euler

        mesh_rotation_rate_module = args.find_cascade_module(ue.ParticleModuleMeshRotationRate)
        if mesh_rotation_rate_module is None:
            CascadeMeshRotationRateToNiagara.CascadeMeshRotationRateConverter.create_modules(args, emitter)

        script_asset = ueFxUtils.create_asset_data(Paths.di_multiply_vector)
        script_args = ue.CreateScriptContextArgs(script_asset)
        multiply_vector_script = ueFxUtils.create_script_context(script_args)

        rotation_rate_link_input = ueFxUtils.create_script_input_linked_parameter(
            "Particles.MeshRotationRate",
            ue.NiagaraScriptInputType.VEC3)
        multiply_vector_script.set_parameter('A', rotation_rate_link_input)
        multiply_vector_script.set_parameter('B', life_multiplier_input)

        result_vector_input = ueFxUtils.create_script_input_dynamic(multiply_vector_script, ue.NiagaraScriptInputType.VEC3)

        emitter.set_parameter_directly(
            "Particles.MeshRotationRate",
            result_vector_input,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)
