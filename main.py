from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
from business_logic import HaystackPipeline
from models import Query, ProcessedExhibit
from haystack.reader.farm import FARMReader
from haystack.pipeline import ExtractiveQAPipeline

app = FastAPI()
haystack_pipeline = HaystackPipeline()

app = FastAPI()

origins = [
    "http://exhibitmanagernlp.bitstone.eu"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def index():
    return {'message': 'If you wanna check available routes put /docs after base url'}

@app.post('/upload-exhibit-json')
async def upload_exhibit_json(exhibit_json: ProcessedExhibit):
  haystack_pipeline.updateProcessedObject(exhibit_json.exhibit_array)
  haystack_pipeline.retriever = haystack_pipeline._retriever_init()
  haystack_pipeline.reader = FARMReader(model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad", use_gpu=True)
  haystack_pipeline.pipe = ExtractiveQAPipeline(haystack_pipeline.reader, haystack_pipeline.retriever)
  return {"message": "Pipe is ready for prediction"}

@app.post('/predict')
async def predict(query: Query):
    question_dict = query.dict()
    prediction = haystack_pipeline.predict(question_dict['question'])
    return prediction