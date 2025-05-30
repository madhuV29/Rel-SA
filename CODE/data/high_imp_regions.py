#Disctionary categorizing AAL3 Atlas regions by relevance to AD and MCI
aal3_regions = {
    "High Relevance": {
        3: "Frontal_Sup_2_L",
        4: "Frontal_Sup_2_R",
        23: "Rectus_L",
        24: "Rectus_R",
        29: "OFCpost_L",
        30: "OFCpost_R",
        31: "OFClat_L",
        32: "OFClat_R",
        33: "Insula_L",
        34: "Insula_R",
        37: "Cingulate_Mid_L",
        38: "Cingulate_Mid_R",
        39: "Cingulate_Post_L",
        40: "Cingulate_Post_R",
        41: "Hippocampus_L",
        42: "Hippocampus_R",
        43: "ParaHippocampal_L",
        44: "ParaHippocampal_R",
        45: "Amygdala_L",
        46: "Amygdala_R",
        57: "Occipital_Inf_L",
        58: "Occipital_Inf_R",
        59: "Fusiform_L",
        60: "Fusiform_R",
        69: "Angular_L",
        70: "Angular_R",
        71: "Precuneus_L",
        72: "Precuneus_R",
        75: "Caudate_L",
        76: "Caudate_R",
        77: "Putamen_L",
        78: "Putamen_R",
        79: "Pallidum_L",
        80: "Pallidum_R",
        121: "Thal_AV_L",
        122: "Thal_AV_R",
        123: "Thal_LP_L",
        124: "Thal_LP_R",
        125: "Thal_VA_L",
        126: "Thal_VA_R",
        127: "Thal_VL_L",
        128: "Thal_VL_R",
        131: "Thal_IL_L",
        132: "Thal_IL_R",
        151: "ACC_sub_L",
        152: "ACC_sub_R",
        153: "ACC_pre_L",
        154: "ACC_pre_R",
        157: "N_Acc_L",
        158: "N_Acc_R"
    }
}



# Dictionary categorizing JHU ICBM-DTI-81 Atlas regions by relevance to AD and MCI
JHU_WM_regions = {
    "High Relevance": {
        4: {"Description": "Genu of Corpus Callosum"},
        5: {"Description": "Body of Corpus Callosum"},
        6: {"Description": "Splenium of Corpus Callosum"},
        7: {"Description": "Fornix (Cres)"},
        10: {"Description": "Medial Lemniscus (Right)"},
        11: {"Description": "Medial Lemniscus (Left)"},
        18: {"Description": "Anterior Limb of Internal Capsule (Right)"},
        19: {"Description": "Anterior Limb of Internal Capsule (Left)"},
        20: {"Description": "Posterior Limb of Internal Capsule (Right)"},
        21: {"Description": "Posterior Limb of Internal Capsule (Left)"},
        22: {"Description": "Retrolenticular Part of Internal Capsule (Right)"},
        23: {"Description": "Retrolenticular Part of Internal Capsule (Left)"},
        24: {"Description": "Anterior Corona Radiata (Right)"},
        25: {"Description": "Anterior Corona Radiata (Left)"},
        28: {"Description": "Posterior Corona Radiata (Right)"},
        29: {"Description": "Posterior Corona Radiata (Left)"},
        30: {"Description": "Posterior Thalamic Radiation (Right)"},
        31: {"Description": "Posterior Thalamic Radiation (Left)"},
        34: {"Description": "External Capsule (Right)"},
        35: {"Description": "External Capsule (Left)"},
        36: {"Description": "Cingulum (Right)"},
        37: {"Description": "Cingulum (Left)"},
        38: {"Description": "Cingulum (Hippocampus) (Right)"},
        39: {"Description": "Cingulum (Hippocampus) (Left)"},
        40: {"Description": "Fornix (Body)"},
        41: {"Description": "Fornix (Cres)"},
        42: {"Description": "Superior Longitudinal Fasciculus (Right)"},
        43: {"Description": "Superior Longitudinal Fasciculus (Left)"},
        44: {"Description": "Superior Fronto-Occipital Fasciculus"},
        45: {"Description": "Uncinate Fasciculus (Right)"},
        46: {"Description": "Uncinate Fasciculus (Left)"}
    }
}


AAL3_high_relevance_regions = list(aal3_regions["High Relevance"].keys())
JHU_WM_high_relevance_regions = list(JHU_WM_regions["High Relevance"].keys())

def get_high_regions():
    return AAL3_high_relevance_regions, JHU_WM_high_relevance_regions
