from flask import Flask, render_template, request
import pickle
import string
import numpy as np
import pandas as pd

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('cv.pkl', 'rb') as f:
    cv = pickle.load(f)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question1 = request.form['question1']
        question2 = request.form['question2']
        
        def remove_punctuation(text):
            if isinstance(text, str):  
                  translator = str.maketrans('', '', string.punctuation)
                  return text.translate(translator)
            else:
              return text 

        question1=remove_punctuation(question1)
        question2=remove_punctuation(question2)   
        abbreviation_dict = {
        "ASAP": "As Soon As Possible",
        "FYI": "For Your Information",
        "IMO": "In My Opinion",
        "IMHO": "In My Humble Opinion",
        "TBA": "To Be Announced",
        "TBC": "To Be Confirmed",
        "TBD": "To Be Decided/Determined",
        "ETA": "Estimated Time of Arrival",
        "FAQ": "Frequently Asked Questions",
        "ICYMI": "In Case You Missed It",
        "IDK": "I Don't Know",
        "IIRC": "If I Recall Correctly",
        "N/A": "Not Applicable",
        "BRB": "Be Right Back",
        "BTW": "By The Way",
        "AFAIK": "As Far As I Know",
        "AKA": "Also Known As",
        "NVM": "Never Mind",
        "NP": "No Problem",
        "TIA": "Thanks In Advance",
        "LOL": "Laugh Out Loud",
        "ROFL": "Rolling On the Floor Laughing",
        "LMK": "Let Me Know",
        "GG": "Good Game",
        "AFK": "Away From Keyboard",
        "TTYL": "Talk To You Later",
        "OMW": "On My Way",
        "TBH": "To Be Honest",
        "FTW": "For The Win",
        "DM": "Direct Message",
        "PM": "Private Message",
        "IRL": "In Real Life",
        "TL;DR": "Too Long; Didn't Read",
        "YOLO": "You Only Live Once",
        "SMH": "Shaking My Head",
        "ICYDK": "In Case You Didn't Know",
        "MIA": "Missing In Action",
        "BFF": "Best Friends Forever",
        "JK": "Just Kidding",
        "IDC": "I Don't Care",
        "SMH": "Shaking My Head",
        "BBS": "Be Back Soon",
        "GTG": "Got To Go",
        "OOTD": "Outfit Of The Day",
        "BF": "Boyfriend",
        "GF": "Girlfriend",
        "IDC": "I Don't Care",
        "PLZ": "Please",
        "PPL": "People",
        "RN": "Right Now",
        "SRS": "Serious",
        "SRSLY": "Seriously",
        "WB": "Welcome Back",
        "IDK": "I Don't Know",
        "IKR": "I Know, Right?",
        "WTF": "What The F***",
        "FOMO": "Fear Of Missing Out",
        "BRB": "Be Right Back",
        "TGIF": "Thank God It's Friday",
        "WTH": "What The Heck",
        "LMAO": "Laughing My Ass Off",
        "BFN": "Bye For Now",
        "OMG": "Oh My God",
        "XOXO": "Hugs and Kisses",
        "GR8": "Great",
        "CUL8R": "See You Later",
        "HAND": "Have A Nice Day",
        "POTD": "Picture Of The Day",
        "TMI": "Too Much Information",
        "ICYMI": "In Case You Missed It",
        "BAE": "Before Anyone Else",
        "HBD": "Happy Birthday",
        "IDTS": "I Don't Think So",
        "IYKYK": "If You Know, You Know",
        "YOLO": "You Only Live Once"
    }
        def chat_conversion(text):
            if isinstance(text, str):
                words = text.split()
                converted_words = [abbreviation_dict.get(word.upper(), word) for word in words]
                return ' '.join(converted_words)
            else:
                return text

        question1=chat_conversion(question1)
        question2=chat_conversion(question2)      
        q1_len = len(question1) 
        q2_len = len(question2) 
        q1_num_words =  len(question1 .split(" "))
        q2_num_words =  len(question2.split(" "))
        def num_common_words(question1, question2):
        
            words1 = set(question1.lower().split())
            words2 = set(question2.lower().split())
            common_words = words1.intersection(words2)
            return len(common_words)

        word_common = num_common_words(question1,question2)
    
        def total_words(question1,question2):
            words1 = set(question1.lower().split())
            words2 = set(question2.lower().split())
            all_unique_words = words1.union(words2)
            return len(all_unique_words)


        word_total = total_words(question1,question2)
    
        word_share = round(word_common/word_total,2)
        # convert to dataframe with cols q1_len	q2_len	q1_num_words	q2_num_words	word_common	word_total	word_share
        data = {
            'q1_len': [q1_len],
            'q2_len': [q2_len],
            'q1_num_words': [q1_num_words],
            'q2_num_words': [q2_num_words],
            'word_common': [word_common],
            'word_total': [word_total],
            'word_share': [word_share]
        }

        df = pd.DataFrame(data)
    
        questions = [question1, question2]
        transformed_questions = cv.transform(questions).toarray()
        
        
        
        q1_arr, q2_arr = np.vsplit(transformed_questions, 2)
        temp_df1 = pd.DataFrame(q1_arr, columns=cv.get_feature_names_out())
        temp_df2 = pd.DataFrame(q2_arr, columns=cv.get_feature_names_out())
        temp_df = pd.concat([temp_df1, temp_df2], axis=1)
        
        df = pd.concat([df, temp_df], axis=1)
        
        predictions=model.predict(df)
        if(predictions>=0.5):
            predictions="The Question Pairs are Similar"
        else:
             predictions="The Question Pairs are not Similar"    


        return render_template('index.html', predictions=predictions,question1=question1,question2=question2)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
