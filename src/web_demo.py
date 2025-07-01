import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load translation models
print("Loading models...")
tokenizer_en_es = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
model_en_es = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")

tokenizer_es_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
model_es_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")

def translate_en_to_es(text):
    inputs = tokenizer_en_es(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model_en_es.generate(**inputs)
    return tokenizer_en_es.decode(outputs[0], skip_special_tokens=True)

def translate_es_to_en(text):
    inputs = tokenizer_es_en(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model_es_en.generate(**inputs)
    return tokenizer_es_en.decode(outputs[0], skip_special_tokens=True)

with gr.Blocks() as demo:
    gr.Markdown("## 🌐 Cross-lingual Translator (EN ↔ ES)")
    
    with gr.Row():
        with gr.Column():
            text_en = gr.Textbox(label="English Input")
            translate_button_en_es = gr.Button("Translate EN → ES")
            output_es = gr.Textbox(label="Spanish Output")

        with gr.Column():
            text_es = gr.Textbox(label="Spanish Input")
            translate_button_es_en = gr.Button("Translate ES → EN")
            output_en = gr.Textbox(label="English Output")

    translate_button_en_es.click(translate_en_to_es, inputs=text_en, outputs=output_es)
    translate_button_es_en.click(translate_es_to_en, inputs=text_es, outputs=output_en)

demo.launch()
