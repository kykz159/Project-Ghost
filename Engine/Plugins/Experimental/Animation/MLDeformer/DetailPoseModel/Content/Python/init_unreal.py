# -*- coding: utf-8 -*-
"""
ML Deformer Detail Pose Model.

Copyright Epic Games, Inc. All Rights Reserved.
"""

from importlib import reload
import unreal

if unreal.is_editor():
    @unreal.uclass()
    class DetailPosePythonTrainingModel(unreal.DetailPoseTrainingModel):
        @unreal.ufunction(override=True)
        def update_available_devices(self):
            import mldeformer.training_helpers
            reload(mldeformer.training_helpers)
            mldeformer.training_helpers.update_training_device_list(self)

        @unreal.ufunction(override=True)
        def train(self):
            assert self.get_model().mode == 1
            print('Training the Detail Pose Model')
            import neuralmorphmodel_global
            reload(neuralmorphmodel_global)
            return neuralmorphmodel_global.train(self)
