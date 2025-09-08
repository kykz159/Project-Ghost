# Copyright Epic Games, Inc. All Rights Reserved.

from ModuleConverterInterface import ModuleConverterInterface
import unreal as ue
from unreal import FXConverterUtilitiesLibrary as ueFxUtils
import Paths

class CascadeAxisLockConverter(ModuleConverterInterface):

    @classmethod
    def get_input_cascade_module(cls):
        return ue.ParticleModuleOrientationAxisLock

    @classmethod
    def convert(cls, args):
        cascade_module = args.get_cascade_module()
        emitter = args.get_niagara_emitter_context()

        # lock axis is only valid for the sprite renderer, so start by finding 
        # that.
        sprite_renderer_props = emitter.find_renderer(
            "SpriteRenderer")
        if sprite_renderer_props is not None:
            # get all properties from the cascade orientation module
            # noinspection PyTypeChecker
            axis_lock = ueFxUtils.get_particle_module_orientation_axis_lock_props(cascade_module)

           
            # set the sprite facing vector and custom alignment vector depending
            # on the axis lock.
            lock_axis_type = 'None'
            if axis_lock == ue.ParticleAxisLock.EPAL_X:
                lock_axis_type = '+ X'
            elif axis_lock == ue.ParticleAxisLock.EPAL_Y:
                lock_axis_type = '+ Y'
            elif axis_lock == ue.ParticleAxisLock.EPAL_Z:
                lock_axis_type = '+ Z'
            elif axis_lock == ue.ParticleAxisLock.EPAL_NEGATIVE_X:
                lock_axis_type = '- X'
            elif axis_lock == ue.ParticleAxisLock.EPAL_NEGATIVE_Y:
                lock_axis_type = '- Y'
            elif axis_lock == ue.ParticleAxisLock.EPAL_NEGATIVE_Z:
                lock_axis_type = '- Z'
            elif axis_lock == ue.ParticleAxisLock.EPAL_ROTATE_X:
                lock_axis_type = 'Rotate X'
            elif axis_lock == ue.ParticleAxisLock.EPAL_ROTATE_Y:
                lock_axis_type = 'Rotate Y'
            elif axis_lock == ue.ParticleAxisLock.EPAL_ROTATE_Z:
                lock_axis_type = 'Rotate Z'
            else:
                raise NameError("Encountered invalid particle axis lock when "
                                "converting cascade lock axis module!")
            script_asset = ueFxUtils.create_asset_data(Paths.script_sprite_lock_axis)
            script_args = ue.CreateScriptContextArgs(script_asset)
            sprite_lock_axis_script = emitter.find_or_add_module_script(
                "SpriteLockAxis",
                script_args,
                ue.ScriptExecutionCategory.PARTICLE_SPAWN)

            lock_axis_type_input = ueFxUtils.create_script_input_enum(
                Paths.enum_niagara_sprite_lock_axis,
                lock_axis_type)

            sprite_renderer_props.set_editor_property(
                "Alignment",
                ue.NiagaraSpriteAlignment.AUTOMATIC)

            sprite_renderer_props.set_editor_property(
                "FacingMode",
                ue.NiagaraSpriteFacingMode.AUTOMATIC)

            sprite_lock_axis_script.set_parameter('Lock Axis Type', lock_axis_type_input)
