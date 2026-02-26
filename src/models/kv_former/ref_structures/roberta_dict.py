from transformers import AutoModelForCausalLM, Blip2ForConditionalGeneration, Blip2Config, RobertaModel

model = AutoModelForCausalLM.from_pretrained("roberta-base")
with open("roberta_structure.txt", "w") as f:
    structure = model.__str__()
    f.write(structure)

config = Blip2Config()
# import pdb; pdb.set_trace()
model = Blip2ForConditionalGeneration(config)
with open("blip2_structure.txt", "w") as f:
    structure = model.__str__()
    f.write(structure)