from deeppavlov import build_model

question = 'what number of religions is chrismation practiced by?'
# sparql = g([sexpr])

model = build_model('kbqa_custom_graph.json', download=True)
print(model([question]))