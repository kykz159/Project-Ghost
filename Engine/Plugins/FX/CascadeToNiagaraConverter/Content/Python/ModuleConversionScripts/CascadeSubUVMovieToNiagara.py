# Copyright Epic Games, Inc. All Rights Reserved.

from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import CascadeToNiagaraHelperMethods as c2nUtils
import Paths


class CascadeSubUVMovieConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleSubUVMovie

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()
        
        # get all properties from the cascade sub uv movie module
        # noinspection PyTypeChecker
        (animation,
         subuv_index_distribution,
         b_use_real_time
         ) = ueFxUtils.get_particle_module_sub_uv_props(cascade_module)

        # noinspection PyTypeChecker
        (b_use_emitter_time,
         framerate_distribution,
         start_frame
         ) = ueFxUtils.get_particle_module_sub_uv_movie_props(cascade_module)

        # find/add the module script for sub uv animation
        script_asset = ueFxUtils.create_asset_data(Paths.script_subuv_animation_v2)
        script_args = ue.CreateScriptContextArgs(script_asset)
        subuv_script = emitter.find_or_add_module_script(
            "SubUVMovie",
            script_args,
            ue.ScriptExecutionCategory.PARTICLE_UPDATE)

        # set the subuv mode to infinite for equivalent behavior to subuv movie.
        script_input = ueFxUtils.create_script_input_enum(Paths.enum_niagara_subuv_lookup_mode_v2, "Infinite Loop")
        subuv_script.set_parameter("SubUV Animation Mode", script_input)


        # set the play rate
        if b_use_real_time:
            subuv_script.log(
                "Failed to set \"Use Emitter Time\": Niagara does not support this mode!",
                ue.NiagaraMessageSeverity.ERROR)
            
            #  todo Divide particle age by world time dilation for the play rate input. Not implemented as Niagara does 
            #  not currently subsume world time dilation.
            pass
            
        fps_playrate = ueFxUtils.create_script_input_enum(Paths.enum_niagara_fps_playrate, "Frames Per Second")
        framerate = ueFxUtils.get_float_distribution_const_values(framerate_distribution)
        framerate_input = ueFxUtils.create_script_input_int(int(framerate))
        subuv_script.set_parameter("Playback Mode", fps_playrate)
        subuv_script.set_parameter("Frames Per Second", framerate_input)
        if start_frame == 0:
            subuv_script.set_parameter("Random Start Frame", ueFxUtils.create_script_input_bool(True))
        else:
            start_frame_input = ueFxUtils.create_script_input_int(start_frame - 1)
            subuv_script.set_parameter("Start Frame Offset", start_frame_input)
