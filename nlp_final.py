import spacy
import nltk, re, pprint
from nltk.corpus import wordnet 
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

class EntityMatcher(object):
    name = 'entity_matcher'

    def __init__(self, nlp, terms, label):
        patterns = [nlp.make_doc(text) for text in terms]
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add(label, None, *patterns)

    def __call__(self, doc):
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = Span(doc, start, end, label=match_id)
            doc.ents = list(doc.ents) + [span]
        return doc
def getHypernyms(token):
	hypernyms = []
	synsets = wordnet.synsets(token)
	for synset in synsets:
		for h in synset.hypernyms():
			for l in h.lemmas():
				hypernyms.append(l.name())
	return list(set(hypernyms))
def getHyponyms(token):
	hyponyms = []
	synsets = wordnet.synsets(token)
	for synset in synsets:
		for h in synset.hyponyms():
			for l in h.lemmas():
				hyponyms.append(l.name())
	return list(set(hyponyms))
def getMeronyms(token):
	meronyms = []
	synsets = wordnet.synsets(token)
	for synset in synsets:
		for h in synset.part_meronyms():
			for l in h.lemmas():
				meronyms.append(l.name())
	return list(set(meronyms))
def getHolonyms(token):
	holonyms = []
	synsets = wordnet.synsets(token)
	for synset in synsets:
		for h in synset.member_holonyms():
			for l in h.lemmas():
				holonyms.append(l.name())
	return list(set(holonyms))
if __name__ == '__main__':
	synonyms = {
	'killing' : {'shot','fired','killed','murdered', 'murder', 'killing', 'vote_out', 'cleanup', 'defeat', 'vote_down', 'pour_down', 'drink_down', 'kill', 'obliterate', 'wipe_out', 'sidesplitting', 'pop', 'down', 'toss_off', 'belt_down', 'shoot_down', 'bolt_down', 'violent_death', 'stamp_out', 'putting_to_death'}
	}
	nlp = spacy.load('en_core_web_sm')


	filename = 'killing.txt'
	file = open(filename, 'r', encoding="utf-8")
	document = file.read()
	articles = document.split('##')
	data = []
	dependency = []
	template_killing  = dict()
	
	instruments = ['.45-caliber semi-automatic pistol','fire_ship', 'brass_knuckles', 'four-pounder', 'Greek_fire', 'battery', 'projectile', 'stun_gun', 'sword', 'slasher', 'blade', 'steel', 'knuckles', 'W.M.D.', 'knuckle_duster', 'gun', 'field_gun', 'hatchet', 'sling', 'light_arm', 'bow', 'knife', 'brass_knucks', 'knucks', 'WMD', 'bow_and_arrow', 'stun_baton', 'flamethrower', 'shaft', 'weapon_of_mass_destruction', 'pike', 'brand', 'field_artillery', 'tomahawk', 'lance', 'cannon', 'spear', 'missile']
	entity_instruments = EntityMatcher(nlp, instruments, 'INSTRUMENT')
	nlp.add_pipe(entity_instruments) 
	id = 0
	for article in articles:
		killing_victim = set()
		killing_perpetrator = set()
		killing_location = set()
		killing_instrument = set()
		killing_date = set()
		id += 1
		sentences = nltk.sent_tokenize(article)
		for sentence in sentences:
			doc = nlp(sentence)
			tokenList = nltk.word_tokenize(sentence)
			person = set()
			date = set()
			location = set()
			instrument = set()
			for word in tokenList:
				if word in synonyms['killing']:
					for ent in doc.ents:
						if ent.label_ == "PERSON":
							person.add(ent.text)
						if ent.label_ == "DATE":
							killing_date.add(ent.text)
						if ent.label_ == "ORG" or ent.label_ == "GPE":
							killing_location.add(ent.text)
						if ent.label_ == "INSTRUMENT":
							killing_instrument.add(ent.text)
					
					if(re.search('(\S+\s+|^)(\S+\s+|)(kills|for killing|was killed|killing|killing of)(\s+\S+|)(\s+\S+|$)', sentence)):
						s = re.search('(\S+\s+|^)(\S+\s+|)(kills|for killing|was killed|killing|killing of)(\s+\S+|)(\s+\S+|$)', sentence).group(0)
						k = nlp(s)

						for seq in k.ents:
							if seq.label_ == 'PERSON':
								for p in person:
									if seq.text in p:
										killing_victim.add(p)
					if(re.search('(\S+\s+|^)(\S+\s+|)(shot|by|fired by|killed himself|killing by|killed by| was killed by)(\s+\S+|)(\s+\S+|$)', sentence)):
						s = re.search('(\S+\s+|^)(\S+\s+|)(shot|by|fired by|killed himself|killing by|killed by| was killed by)(\s+\S+|)(\s+\S+|$)', sentence).group(0)
						k = nlp(s)
						for seq in k.ents:
							if seq.label_ == 'PERSON':
								for p in person:
									if seq.text in p:
										killing_perpetrator.add(p)
						

			template_killing[id] = ({
				'Victim' : ' '.join(killing_victim),
				'perpetrator': ' '.join(killing_perpetrator),
				'Location' : ' '.join(killing_location),
				'Instrument': ' '.join(killing_instrument),
				'Date' : ' '.join(killing_date)
				})			
			for token in doc:
				#print(token)
				data.append({
					"token": token.text,
					"lemma": token.lemma_,
					"pos": token.pos_ ,
					"tag": token.tag_ ,
					"dependency": token.dep_ ,
					"hypernyms":getHypernyms(str(token)), 
					"holonyms" :getHolonyms(str(token)) ,
					"meronyms" :getMeronyms(str(token)) ,
					"hyponyms" : getHyponyms(str(token))
					})
				dependency.append({
					"Text" :token.text,
					"dependency": token.dep_,
					"Head Text" : token.head.text,
					"Head Pos" : token.head.pos_ ,
					"children" : [child for child in token.children]
					})
				# print(dependency)
	print(template_killing)

