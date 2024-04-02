def predict_Emotions():
    from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
    emo_det = request.form['emotion_analysis']

    tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
    model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
    emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa' , return_all_scores= True)

    emotion_labels = emotion(emo_det)
    needed_emotions=[ "admiration","approval","excitement","optimism","pride","curiosity","desire","embarrassment"]
    emotion_dict = {emotion['label']: round(emotion['score']*100,2) for emotion in emotion_labels[0]}
    emotion_dict_final={}
    for key,value in emotion_dict.items():
        if key in needed_emotions:
            if value<1:
                emotion_dict_final[key]='low'
            elif value<20:
                emotion_dict_final[key]='moderate'
            elif value<60:
                emotion_dict_final[key]='high'
            elif value>59:
                emotion_dict_final[key]='very high'
    print(emotion_dict_final)
    return jsonify(emotion_dict_final)
