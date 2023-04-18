import torch 
import re 
from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel
from colabcode import ColabCode
from fastapi import FastAPI
from PIL import Image
import nltk

device='cpu'
encoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
decoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
model_checkpoint = "nlpconnect/vit-gpt2-image-captioning"

feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def predict(image,max_length=64, num_beams=4):
  image = image.convert('RGB')
  image = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
  clean_text = lambda x: x.replace('<|endoftext|>','').split('\n')[0]
  caption_ids = model.generate(image, max_length = max_length)[0]
  caption_text = clean_text(tokenizer.decode(caption_ids))

  return caption_text

def extract_nouns_verbs(sentence):
    words = nltk.word_tokenize(sentence)
    tagged_words = nltk.pos_tag(words)
    nouns = [word for word, pos in tagged_words if pos.startswith('N')]
    verbs = [word for word, pos in tagged_words if pos.startswith('V')]
    
    return nouns, verbs

@app.get("/")
async def read_root(image_path: str):
  try:
    image = Image.open(image_path)
    caption = predict(image)
    nouns, verbs = extract_nouns_verbs(caption)
  except Exception as e:
    print(e)
    return {"message": "image not found!",}

  return {"message": caption, "nouns": nouns, "verbs": verbs}
