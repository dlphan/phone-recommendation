from underthesea import word_tokenize
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import string
import numpy as np
import unidecode
import re
import joblib
import flask
import socket

ip = socket.gethostbyname(socket.gethostname())


app = flask.Flask(__name__)
app.config["DEBUG"] = True

path='Sentiment_Classification/'
stopwords=set(open(path+'data/stopwords.txt',encoding='utf-8').read().split(' ')[:-1])
max_token=joblib.load(path+'model/max_token.pkl')
RNN_model=joblib.load(path+'model/LSTM_model.pkl')
RNN_embedding=joblib.load(path+'model/tokenizer_LSTM.pkl')
SVM_model=joblib.load(path+'model/SVM_model.pkl')
NB_model=joblib.load(path+'model/NB_model.pkl')
TFIDF_embedding=joblib.load(path+'model/tfidf.pkl')

graph = tf.get_default_graph()

def vi_tokenizer(row):
    return word_tokenize(row, format="text")

def remove_stopwords(stopwords,hl_split):
  sent = [s for s in hl_split if s not in stopwords ]
  return sent

def normalize_text(text):
    #Remove extended characters: ex: đẹppppppp
    text = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)

    # lower text
    text = text.lower()

    #Standardize Vietnamese, handle emoj, standardize English
    replace_list = {
        'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ', 'òe': 'oè', 'óe': 'oé','ỏe': 'oẻ',
        'õe': 'oẽ', 'ọe': 'oẹ', 'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ','ụy': 'uỵ', 'uả': 'ủa',
        'ả': 'ả', 'ố': 'ố', 'u´': 'ố','ỗ': 'ỗ', 'ồ': 'ồ', 'ổ': 'ổ', 'ấ': 'ấ', 'ẫ': 'ẫ', 'ẩ': 'ẩ',
        'ầ': 'ầ', 'ỏ': 'ỏ', 'ề': 'ề','ễ': 'ễ', 'ắ': 'ắ', 'ủ': 'ủ', 'ế': 'ế', 'ở': 'ở', 'ỉ': 'ỉ',
        'ẻ': 'ẻ', 'àk': u' à ','aˋ': 'à', 'iˋ': 'ì', 'ă´': 'ắ','ử': 'ử', 'e˜': 'ẽ', 'y˜': 'ỹ', 'a´': 'á',
        #Quy các icon về 2 loại emoj: Tích cực hoặc tiêu cực
        "👹": "nagative", "👻": "positive", "💃": "positive",'🤙': ' positive ', '👍': ' positive ',
        "💄": "positive", "💎": "positive", "💩": "positive","😕": "nagative", "😱": "nagative", "😸": "positive",
        "😾": "nagative", "🚫": "nagative",  "🤬": "nagative","🧚": "positive", "🧡": "positive",'🐶':' positive ',
        '👎': ' nagative ', '😣': ' nagative ','✨': ' positive ', '❣': ' positive ','☀': ' positive ',
        '♥': ' positive ', '🤩': ' positive ', 'like': ' positive ', '💌': ' positive ',
        '🤣': ' positive ', '🖤': ' positive ', '🤤': ' positive ', ':(': ' nagative ', '😢': ' nagative ',
        '❤': ' positive ', '😍': ' positive ', '😘': ' positive ', '😪': ' nagative ', '😊': ' positive ',
        '?': ' ? ', '😁': ' positive ', '💖': ' positive ', '😟': ' nagative ', '😭': ' nagative ',
        '💯': ' positive ', '💗': ' positive ', '♡': ' positive ', '💜': ' positive ', '🤗': ' positive ',
        '^^': ' positive ', '😨': ' nagative ', '☺': ' positive ', '💋': ' positive ', '👌': ' positive ',
        '😖': ' nagative ', '😀': ' positive ', ':((': ' nagative ', '😡': ' nagative ', '😠': ' nagative ',
        '😒': ' nagative ', '🙂': ' positive ', '😏': ' nagative ', '😝': ' positive ', '😄': ' positive ',
        '😙': ' positive ', '😤': ' nagative ', '😎': ' positive ', '😆': ' positive ', '💚': ' positive ',
        '✌': ' positive ', '💕': ' positive ', '😞': ' nagative ', '😓': ' nagative ', '️🆗️': ' positive ',
        '😉': ' positive ', '😂': ' positive ', ':v': '  positive ', '=))': '  positive ', '😋': ' positive ',
        '💓': ' positive ', '😐': ' nagative ', ':3': ' positive ', '😫': ' nagative ', '😥': ' nagative ',
        '😃': ' positive ', '😬': ' 😬 ', '😌': ' 😌 ', '💛': ' positive ', '🤝': ' positive ', '🎈': ' positive ',
        '😗': ' positive ', '🤔': ' nagative ', '😑': ' nagative ', '🔥': ' nagative ', '🙏': ' nagative ',
        '🆗': ' positive ', '😻': ' positive ', '💙': ' positive ', '💟': ' positive ',
        '😚': ' positive ', '❌': ' nagative ', '👏': ' positive ', ';)': ' positive ', '<3': ' positive ',
        '🌝': ' positive ',  '🌷': ' positive ', '🌸': ' positive ', '🌺': ' positive ',
        '🌼': ' positive ', '🍓': ' positive ', '🐅': ' positive ', '🐾': ' positive ', '👉': ' positive ',
        '💐': ' positive ', '💞': ' positive ', '💥': ' positive ', '💪': ' positive ',
        '💰': ' positive ',  '😇': ' positive ', '😛': ' positive ', '😜': ' positive ',
        '🙃': ' positive ', '🤑': ' positive ', '🤪': ' positive ','☹': ' nagative ',  '💀': ' nagative ',
        '😔': ' nagative ', '😧': ' nagative ', '😩': ' nagative ', '😰': ' nagative ', '😳': ' nagative ',
        '😵': ' nagative ', '😶': ' nagative ', '🙁': ' nagative ',
        #Chuẩn hóa 1 số sentiment words/English words
        ':))': '  positive ', ':)': ' positive ', 'ô kêi': ' ok ', 'okie': ' ok ', ' o kê ': ' ok ',
        'okey': ' ok ', 'ôkê': ' ok ', 'oki': ' ok ', ' oke ':  ' ok ',' okay':' ok ','okê':' ok ',
        ' tks ': u' cám ơn ', 'thks': u' cám ơn ', 'thanks': u' cám ơn ', 'ths': u' cám ơn ', 'thank': u' cám ơn ',
        '⭐': 'star ', '*': 'star ', '🌟': 'star ', '🎉': u' positive ',
        'kg ': u' không ',' not ': u' không ', u' kg ': u' không ', '"k ': u' không ',' kh ':u' không ','kô':u' không ','hok':u' không ',' kp ': u' không phải ',u' kô ': u' không ', '"ko ': u' không ', u' ko ': u' không ', u' k ': u' không ', 'khong': u' không ', u' hok ': u' không ',
        'he he': ' positive ','hehe': ' positive ','hihi': ' positive ', 'haha': ' positive ', 'hjhj': ' positive ',
        ' lol ': ' nagative ',' cc ': ' nagative ','cute': u' dễ thương ','huhu': ' nagative ', ' vs ': u' với ', 'wa': ' quá ', 'wá': u' quá', ' j ': u' gì ', '“': ' ',
        ' sz ': u' cỡ ', 'size': u' cỡ ', u' đx ': u' được ', 'dk': u' được ', 'dc': u' được ', 'đk': u' được ',
        'đc': u' được ','authentic': u' chuẩn chính hãng ',u' aut ': u' chuẩn chính hãng ', u' auth ': u' chuẩn chính hãng ', 'thick': u' positive ', 'store': u' cửa hàng ',
        'shop': u' cửa hàng ', 'sp': u' sản phẩm ', 'gud': u' tốt ','god': u' tốt ','wel done':' tốt ', 'good': u' tốt ', 'gút': u' tốt ',
        'sấu': u' xấu ','gut': u' tốt ', u' tot ': u' tốt ', u' nice ': u' tốt ', 'perfect': 'rất tốt', 'bt': u' bình thường ',
        'time': u' thời gian ', 'qá': u' quá ', u' ship ': u' giao hàng ', u' m ': u' mình ', u' mik ': u' mình ',
        'ể': 'ể', 'product': 'sản phẩm', 'quality': 'chất lượng','chat':' chất ', 'excelent': 'hoàn hảo', 'bad': 'tệ','fresh': ' tươi ','sad': ' tệ ',
        'date': u' hạn sử dụng ', 'hsd': u' hạn sử dụng ','quickly': u' nhanh ', 'quick': u' nhanh ','fast': u' nhanh ','delivery': u' giao hàng ',u' síp ': u' giao hàng ',
        'beautiful': u' đẹp tuyệt vời ', u' tl ': u' trả lời ', u' r ': u' rồi ', u' shopE ': u' cửa hàng ',u' order ': u' đặt hàng ',
        'chất lg': u' chất lượng ',u' sd ': u' sử dụng ',u' dt ': u' điện thoại ',u'đt ': u' điện thoại ' ,u' wfi ': u' wifi ',u' nt ': u' nhắn tin ',u' tl ': u' trả lời ',u' sài ': u' xài ',u'bjo':u' bao giờ ',
        'thik': u' thích ',u' sop ': u' cửa hàng ', ' fb ': ' facebook ', ' face ': ' facebook ', ' very ': u' rất ',u'quả ng ':u' quảng  ',
        'dep': u' đẹp ',u' xau ': u' xấu ','delicious': u' ngon ', u'hàg': u' hàng ', u'qủa': u' quả ', u'%': u' phần trăm ', u'lác': u' lag ', u'lắc': u' lag ', u'wep': u' web ',
        'iu': u' yêu ','fake': u' giả mạo ', 'trl': 'trả lời', '><': u' positive ',
        ' por ': u' tệ ',' poor ': u' tệ ', 'ib':u' nhắn tin ', 'rep':u' trả lời ',u'fback':' feedback ','fedback':' feedback ', u'hqa': u' hôm qua ',
        #less than 3 * converted to 1 *, over 3 * converted to 5 *
        u'mìk': u' mình ', u'ròy': u' rồi ', u'hk': u' không ', 'dt ': u' điện thoại ', 'mún': u' muốn ', 'youtobe': u' youtube ',' s ': u' sao ', ' tuột ': u' tụt ', 'nv': u' nhân viên ',
        '6 sao': ' 5star ','6 star': ' 5star ', '5star': ' 5star ','5 sao': ' 5star ','5sao': ' 5star ',
        'starstarstarstarstar': ' 5star ', '1 sao': ' 1star ', '1sao': ' 1star ','2 sao':' 1star ','2sao':' 1star ',
        '2 starstar':' 1star ','1star': ' 1star ', '0 sao': ' 1star ', '0star': ' 1star ',}

    for k, v in replace_list.items():
      text = text.replace(k, v)
    #remove punctuation 
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)
    # Replace whitespace between terms with a single space
    re_space=re.compile('\s+')
    text=re.sub(re_space,' ',text)
    text=text.strip()
    return text

def standardize_data(df):
    hl_cleansed=[]
    for row in df:
        row=normalize_text(row)
        row=vi_tokenizer(row)
        row=remove_stopwords(stopwords,row.split())
        hl_cleansed.append(row)
    return hl_cleansed

def input_vectorization(input_token,choice):
  if choice=='RNN':
    sequences = RNN_embedding.texts_to_sequences(input_token)
    X = pad_sequences(sequences, maxlen=max_token)
  else:
    list_token=[' '.join(token) for token in input_token ]
    X = TFIDF_embedding.transform(list_token)
  return X
       
def predict(text,choice):
  token=standardize_data([text])
  if choice=='RNN':
    X=input_vectorization(token,'RNN')
    if np.around(RNN_model.predict(X))==1:
      print(text)
      print("=============> Positive\n")
    else:
      print(text)
      print("=============> Negative\n")
  elif choice=='SVM':
    X=input_vectorization(token,'SVM')
    if SVM_model.predict(X)[0]==1:
      print(text)
      print("=============> Positive\n")
    else:
      print(text)
      print("=============> Negative\n")
  else:
    X=input_vectorization(token,'NB')
    if NB_model.predict(X)[0]==1:
      print(text)
      print("=============> Positive\n")
    else:
      print(text)
      print("=============> Negative\n")

def multi_predict(list_data,choice):
  result=dict()
  list_token=standardize_data(list_data)
  rs=[]
  num_positive=None
  num_negative=None
  X=input_vectorization(list_token,choice)
  if choice=='RNN':
    global graph
    with graph.as_default():
      rs=np.around(RNN_model.predict(X))
  elif choice=='SVM':
    rs = SVM_model.predict(X)
  elif choice=='NB':
    rs = NB_model.predict(X)
    
  num_positive=np.count_nonzero(rs == 1)
  num_negative=np.count_nonzero(rs == 0)
  rs=rs.flatten().astype(int)
  
  percentage_of_positive= round(num_positive/(num_positive + num_negative)*100,2)
  index_of_positive=[i for i, e in enumerate(rs) if e == 1]
  index_of_negative=[i for i, e in enumerate(rs) if e == 0]
  recommend='Máy được khen trên '+str(percentage_of_positive)+'%, mua đi !!' if percentage_of_positive > 60 \
                            else 'Máy bị chê quá nhiều ('+str(100-percentage_of_positive)+'%), không nên mua'

  result['total_positive']=num_positive
  result['total_negative']=num_negative
  result['recommend']=recommend
  result['index_of_positive']= index_of_positive
  result['index_of_negative']= index_of_negative
  result['model']= choice


  return result

@app.route('/result', methods=['POST'])
def home():
    listData = flask.request.get_json()
    return multi_predict(listData['reviews'],listData['algorithm'])
app.run(ip)