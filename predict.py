from train import tokenizer, pad_sequences, max_sequence_len, model
import numpy as np

start_text = "Today, diabetic patients"
next_word = 50


# prediction
for _ in range(next_word):
    token_list = tokenizer.texts_to_sequences([start_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predict_x=model.predict(token_list) 
    classes_x=np.argmax(predict_x)

    #predicted = (model.predict(token_list) > 0.5).astype("int32")
    #print(classes_x) 
    output_word = ""
    
   
    for word, index in tokenizer.word_index.items():
        if index == classes_x:
            output_word = word
            # print("hi")
            break
    start_text += " " + output_word
print(start_text)