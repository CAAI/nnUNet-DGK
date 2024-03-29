#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict

from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
from nnunet.paths import *

# nnUNetData_plans_v2.1_PETasCT_3D_plans_3D.pkl
# nnUNetPlansv2.1_plans_3D_PETasCT
class ExperimentPlannerPETasCT_3D(ExperimentPlanner3D_v21):
    """
    Preprocesses all data in CT mode
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlannerPETasCT_3D, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.1_PETasCT_3D" #nnUNetPlansv2.1_plans_3D_PETasCT_plans_3D.pkl
        self.plans_fname = join(self.preprocessed_output_folder, "nnUNetPlansv2.1_plans_PETasCT" + "_plans_3D.pkl")

    def determine_normalization_scheme(self):
        schemes = OrderedDict()
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        for i in range(num_modalities):
            if modalities[i] == "CT":
                schemes[i] = "CT"
            else:
                schemes[i] = "CT"
        return schemes

