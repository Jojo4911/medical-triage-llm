from datasets import load_dataset

# MediQA
mediqa = load_dataset("ANR-MALADES/MediQAl", name="mcqu")
print(mediqa["train"][0])
print(mediqa["train"].features)

# FrenchMedMCQA
fmed = load_dataset("qanastek/frenchmedmcqa", revision="refs/convert/parquet")
print(fmed["train"][0])

# MedQuAD
medquad = load_dataset("keivalya/MedQuad-MedicalQnADataset")
print(medquad["train"][0])
print(medquad["train"].features)

# UltraMedical-Preference
ultra_pref = load_dataset("TsinghuaC3I/UltraMedical-Preference")
print(ultra_pref["train"][0])