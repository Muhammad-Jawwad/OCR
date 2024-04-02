#pip install weasyprint
#pip install pygo
#pip install pangocffi
#pip install flask
#pip install python-doctr
#pip install tensorflow
#pip install tf2onnx
#pip install tensorflow-addons
#pip install rapidfuzz==2.15.1
#installed gtk 3 externally .exe
import os

os.environ['USE_TF']='1'
from flask import Flask,render_template,url_for,request,jsonify
from flask_cors import CORS
import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from werkzeug.utils import secure_filename
import re
import json






app = Flask(__name__)
CORS(app)
model=ocr_predictor(pretrained=True)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_SSC',methods=['POST'])
def predict_SSC():
    matricmarksheet = request.files['SSC']  # Access file object using request.files
    filename_ssc = secure_filename(matricmarksheet.filename)
    filepath_ssc = os.path.join(app.root_path, 'Static', filename_ssc)
    matricmarksheet.save(filepath_ssc)
    
    class_ssc=''
    seatno_ssc=''
    percentage_ssc=''
    total_max_marks_ssc=850
    marks_obtained_ssc=''

    document_ssc = DocumentFile.from_pdf(filepath_ssc)
    result_ssc = model(document_ssc)

    s=str(result_ssc)
    delimiter1 = "value='"
    delimiter2 = "', confidence="

    # Pattern to match the text between the delimiters
    pattern = re.compile(re.escape(delimiter1) + "(.*?)" + re.escape(delimiter2))
    matches_ssc = re.findall(pattern, s)

    for index in range(0,len(matches_ssc)):
        if matches_ssc[index]=='ANNUAL':
            class_ssc=matches_ssc[index+1]

        elif matches_ssc[index]=='NUMBER':
            seatno_ssc=matches_ssc[index+1]

        elif matches_ssc[index]=='TOTAL:':
            temp=matches_ssc[index+1].split('/')
            marks_obtained_ssc=temp[0]
            total_max_marks_ssc=temp[1]

    try:
        percentage_ssc=round((int(marks_obtained_ssc)/int(total_max_marks_ssc)*100),2)
    except:
        percentage_ssc=''

    os.remove(filepath_ssc)

    dic_ssc_to_json={'class':class_ssc,
                 'seatNo':seatno_ssc,
                'totalMarks':total_max_marks_ssc,
                'obtainedMarks':marks_obtained_ssc,
                'percentage':percentage_ssc}

    return jsonify(dic_ssc_to_json)

@app.route('/predict_HSC',methods=['POST'])
def predict_HSC():
    intermarksheet = request.files['HSC']  # Access file object using request.files
    filename_hsc = secure_filename(intermarksheet.filename)
    filepath_hsc = os.path.join(app.root_path, 'Static', filename_hsc)
    intermarksheet.save(filepath_hsc)
    
    document_hsc = DocumentFile.from_pdf(filepath_hsc)
    result_hsc = model(document_hsc)
    hsc_str=str(result_hsc)

    delimiter1 = "value='"
    delimiter2 = "', confidence="

    # Pattern to match the text between the delimiters
    pattern = re.compile(re.escape(delimiter1) + "(.*?)" + re.escape(delimiter2))
    matches_hsc = re.findall(pattern, hsc_str)



    Hsc_Roll=0
    HSC_NO=[]
    hsc_class=''
    for a in range(0,len(matches_hsc)):
        if 'Date' in matches_hsc[a]:
            try:
                for b in range(1,6):
                    if len(matches_hsc[a+b])==4:
                        hsc_class=matches_hsc[a+b]
            except:
                c='' 
        if len(matches_hsc[a])==6:
            try:
                if str(int(matches_hsc[a]))==matches_hsc[a]:
                    Hsc_Roll=matches_hsc[a]
            except:
                c=1

        elif len(matches_hsc[a])==3:
            try:
                if str(int(matches_hsc[a]))==matches_hsc[a]:
                    HSC_NO.append(matches_hsc[a])
            except:
                c=1
    HSC_NO.sort()

    total_max_marks_hsc=1100
    hsc_Marks_obtained=HSC_NO[len(HSC_NO)-1]

    try:
        percentage_hsc=round((int(hsc_Marks_obtained)/int(total_max_marks_hsc)*100),2)
    except:
        percentage_hsc=''




    dict_to_json={
    'class':hsc_class,
    'seatNo':Hsc_Roll,
    'totalMarks': str(total_max_marks_hsc),
    'obtainedMarks':hsc_Marks_obtained,
    'percentage':percentage_hsc
    }

    os.remove(filepath_hsc)
    return jsonify(dict_to_json)


@app.route('/predict_NED',methods=['POST'])
def predict_NED():
    transcript = request.files['NED']  # Access file object using request.files
    filename_ned = secure_filename(transcript.filename)
    filepath_ned = os.path.join(app.root_path, 'Static', filename_ned)
    transcript.save(filepath_ned)

    document_ned = DocumentFile.from_pdf(filepath_ned)
    result_ned = model(document_ned)
    

    Class_NO= ''
    Seat_no=''
    Total_cgpa=4.0
    obtained_cgpa=''
    ned_trans=str(result_ned)
    delimiter1 = "value='"
    delimiter2 = "', confidence="

    # Pattern to match the text between the delimiters
    pattern = re.compile(re.escape(delimiter1) + "(.*?)" + re.escape(delimiter2))
    matches_ned = re.findall(pattern, ned_trans)

    print(matches_ned)

    Total_cgpa=4.0
    for a in range(0,len(matches_ned)):
        if matches_ned[a]=='No'and matches_ned[a-1]=='Seat':
            Seat_no=matches_ned[a+1]
        elif 'NED/' in matches_ned[a]:
        
            Class_NO= matches_ned[a][-4:]
    

        elif len(matches_ned[a])==5:
            temp=matches_ned[a]+'5'
            try:
                if str(float(temp))==temp:
                    # obtained_cgpa=matches_ned[a]
                    obtained_cgpa=str(round(float(matches_ned[a]),1))
            except:
                c=1

    



    dict_to_json_ned={
    'class':  Class_NO, 
    'seatNo':  Seat_no, 
    'totalCGPA':  str(Total_cgpa),
    'obtainedCGPA':  obtained_cgpa
    }
   
    os.remove(filepath_ned)
    return jsonify(dict_to_json_ned)


@app.route('/predict_Emotions',methods=['POST'])
def predict_Emotions():
    from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
    emo_det = str(request.form['emotion_analysis'])
#    emo_det="From a young age, I have nurtured a deep admiration for individuals who dedicate themselves to making a positive impact on the world. Their unwavering commitment and unwavering resolve have always inspired me. As I stand on the precipice of my educational journey, I am filled with a profound sense of approval as I reflect on the accomplishments that have shaped me thus far.The prospect of pursuing higher education fills me with unparalleled excitement and optimism. The opportunity to engage with diverse perspectives, challenge myself academically, and contribute to innovative research ignites a fire within me. With every new door that opens, I am driven by an unyielding desire to make a difference and leave an indelible mark on society.I take immense pride in my achievements and the knowledge that they have been the product of determination and perseverance. These accomplishments have fueled my curiosity and propelled me to explore various fields of study. The multidisciplinary nature of scholarship both intrigues and captivates me, as it presents an avenue to satiate my thirst for knowledge and foster a deeper understanding of the world.As I embark on this scholarship journey, I am driven by an insatiable curiosity to unravel the complexities of the human condition and make meaningful contributions. The prospect of joining a community of scholars who share my passion for growth and learning fills me with anticipation. I yearn to engage in intellectual discourse, collaborate with like-minded individuals, and push the boundaries of knowledge.Above all, I am guided by an unwavering desire to create a positive change in the world. I aspire to leverage the resources and opportunities afforded by this scholarship to drive progress in areas close to my heart. By combining my academic pursuits with a sense of purpose, I aim to leave a lasting legacy that transcends personal achievements.In conclusion, I am fueled by emotions of admiration, approval, excitement, optimism, pride, curiosity, and desire as I embark on this scholarship endeavor. With an unwavering commitment to personal growth, a thirst for knowledge, and a burning desire to make a difference, I am prepared to seize the opportunities that lie ahead and emerge as a catalyst for change"
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
                emotion_dict_final[key]='moderate'
            elif value<20:
                emotion_dict_final[key]='moderate'
            elif value<60:
                emotion_dict_final[key]='high'
            elif value>59:
                emotion_dict_final[key]='very high'
    print(emotion_dict_final)
    return jsonify(emotion_dict_final)

    print(emotion_dict)
#    return render_template('emotions.html', prediction_emotion= emotion_dict)

if __name__ == '__main__':
	app.run(debug=True)