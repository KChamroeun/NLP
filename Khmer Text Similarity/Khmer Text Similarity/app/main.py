import re
import numpy as np
from typing import Optional
from scipy.spatial import distance
from gensim.models import FastText
from supabase import create_client, Client
from fastapi.staticfiles import StaticFiles
from db.database import fetch_document_vectors
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Form, UploadFile, Request


url: str = "https://evdaaabkuvrgufbvhmsd.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImV2ZGFhYWJrdXZyZ3VmYnZobXNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTc0MzkyMjUsImV4cCI6MjAzMzAxNTIyNX0.Yd9weYFXPJEIeR293pKHh8bOBaGVuQjcH5gj14dR8Gw"
supabase: Client = create_client(url, key)

app = FastAPI()

app.mount("/static", StaticFiles(directory="website/static"), name="static")
templates = Jinja2Templates(directory="website/templates")

@app.on_event("startup")
def load_model():
    global fasttext_model
    fasttext_model = FastText.load(
        "/Users/bormeychanchem/Desktop/fyp2/app/website/model/config.fasttext.v5.bin")
    
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/process-text")
async def process_text(request: Request, text: Optional[str] = Form(""), file: Optional[UploadFile] = None):
    content = ""
    print("Received text:", text)
    print(file)
    if file.filename != "" and text:
        return templates.TemplateResponse("home.html", {"request": request, "message": "Please provide only one input type, either a file or plain text.", "result": None})

    if file.filename != "":
        content = (await file.read()).decode('utf-8')
    elif text:
        content = text
        print("content from text", content)
    else:
        return templates.TemplateResponse("home.html", {"request": request, "message": "No text or file provided.", "result": None})
    
    try:
        print(content)
        words = preprocess(content)
        vectors = [fasttext_model.wv[word] for word in words]
        
        if vectors:
            source_vector = np.mean(np.array(vectors), axis=0).tolist()
            document_vectors = fetch_document_vectors()
            
            # print("len(document_vectors)")
            # print(len(document_vectors))
        
            similarities = [(title, round(cal_cosine_sim(source_vector, target), 3), content)
                            for title, target, content in document_vectors]
            top_matches = sorted(
                similarities, key=lambda x: x[1], reverse=False)[:10]
            print(top_matches)
            return templates.TemplateResponse("home.html", {"request": request, "result": top_matches, "message": None})
        else:
            return templates.TemplateResponse("home.html", {"request": request, "message": "No valid words found in input.", "result": None})
    except Exception as e:
        print(e)
        return templates.TemplateResponse("home.html", {"request": request, "message": str(e), "result": None})

def cal_cosine_sim(source_vector, target):
    dist = distance.cosine(source_vector, target)
    return dist

def preprocess(text):
    text = re.sub("[/#]", '\u0020', text)
    text = re.sub(r"\d+", '\u0020', text)
    text = text.replace('\xa0', '\u0020')
    text = re.sub('\u0020+', '\u0020', text)
    text = re.sub('[()“«»]', '', text)
    text = re.sub('៕។៛ៗ៚៙៘,.?!', '', text)
    text = re.sub('០១២៣៤៥៦៧៨៩0123456789', '', text)
    text = re.sub('᧠᧡᧢᧣᧤᧥᧦᧧᧨᧩᧪᧫᧬᧭᧮᧯᧰᧱᧲᧳᧴᧵᧶᧷᧸᧹᧺᧻᧼᧽᧾᧿', '', text)
    khmer_stopwords = set([" ", "ៗ", "។ល។", "៚", "។", "៕", "៖", "៙", "-", "...", "+", "=", ",", "–", "/", "", "!", "!!", "!!!",
                           "នេះ", "នោះ", "អ្វី", "ទាំង", "គឺ", "ដ៏", "ណា", "បើ",
                           "ខ្ញុំ", "អ្នក", "គាត់", "នាង", "ពួក", "យើង", "ពួកគេ", "លោក", "គេ", "នៃ", "នឹង", "នៅ",
                           "ដែល", "ដោយ", "ក៏", "ហើយ", "ដែរ", "ទេ", "ផង", "វិញ", "ខាង", "អស់",
                          "និង", "ដែល", "ជា", "តែ", "ដើម្បី", "បាន", "យ៉ាង", "ទៀត",
                           "ប៉ុន្តែ", "ដោយសារ", "ពេលដែល", "ហើយ", "ដូចជា", "ដូច្នេះ", "ពេលណាមួយ", "ទៅវិញ", "តែម្តង",
                           "ជាមួយ", "ដូចគ្នានិង", "រួចហើយទេ", "ឬ", "ដោយ", 'ជាមួយនឹង'
                           ])
    text = text.lower()
    words = text.split(" ")
    words = [w for w in words if w not in khmer_stopwords]
    return words


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

