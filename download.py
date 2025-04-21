from transformers import AutoTokenizer, EsmModel, EsmForProteinFolding

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D")


tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
