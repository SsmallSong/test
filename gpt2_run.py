# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, set_seed
from datasets import load_dataset
from trl import SFTTrainer
device = "cuda" # the device to load the model onto

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

#Text Generated:
prompt_text = "Tell me a cute story"
input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

output_ids = model.generate(input_ids, max_length=100, num_return_sequences=3, 
                            pad_token_id=tokenizer.eos_token_id,
                            do_sample=True,  # ??????????????????????
                            top_k=50,         # ????????????????????????????
                            top_p=0.95,       # ????????????????????????
                            temperature=0.7,  # ????????????????????????????????????????????
                            no_repeat_ngram_size=2  # ??????????????????????????n-gram
                            )
generated_text=[]
for i in range(len(output_ids)):
    generated_text.append(tokenizer.decode(output_ids[i], skip_special_tokens=True))
print("Generated Text:")

for i in range(len(generated_text)):
    print(generated_text[i])
    print('')


# #Text Generated in other way:
# generator = pipeline('text-generation', model='gpt2')
# set_seed(42)
# generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)


# #Get the features of given text:
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
