from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.retriever.dense import DensePassageRetriever
from haystack.pipeline import ExtractiveQAPipeline
import urllib.request

class HaystackPipeline():

    def __init__(self):
        self.exhibitProcessedArray = []
        # self.exhibitInfoArray = self._getGithubRawObject()

        # These should be initialised after processed exhibit was sent to the API, happening in App Controller now.
        # self.retriever = self._retriever_init()
        # self.reader = FARMReader(model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad", use_gpu=True)
        # self.pipe = ExtractiveQAPipeline(self.reader, self.retriever)

    def updateProcessedObject(self, processedArray):
      self.exhibitProcessedArray = self._format_retriever_input(processedArray)

    # Method used to extract processed object from github, but now it can be taken from postman/frontendapp

    # def _getGithubRawObject(self):
    #   exhibitInfoArray = []
    #   with urllib.request.urlopen("https://raw.githubusercontent.com/motionDew/exhibit/master/response.json") as url:
    #         exhibitInfoArray = json.loads(url.read().decode())
      
    #   finalExhibitInfo = self._format_retriever_input(exhibitInfoArray)
    #   return finalExhibitInfo

    def _format_retriever_input(self, requestObject):
      class Identifier:
        idx = 0
      def format(element):
        new_element = {
            "text": element["text"],
            "footnotes": element["meta"]["footnotes"],
            "id": Identifier.idx
        }
        Identifier.idx += 1
        return new_element

      finalExhibitInfo = map(format,requestObject)
      return list(finalExhibitInfo)

    def _retriever_init(self):
        # init FAISS Document store
        document_store = FAISSDocumentStore(
            faiss_index_factory_str="Flat"
        )

        haystackContexts = []
        for info in self.exhibitProcessedArray:
          if len(info['text']) > 300 :
            haystackContexts.append({
              'id': info["id"],
              'text': clean_wiki_text(info["text"].replace('\r\n', ' ')),
              'meta': { 
                  'paragraph_id': info["id"],
                }
            })

        # write into documentstore
        document_store.write_documents(haystackContexts)

        retriever = DensePassageRetriever(document_store=document_store,
                                        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                        batch_size=16,
                                        use_gpu=True,
                                        embed_title=True,
                                        use_fast_tokenizers=True)

        # Important: 
        # Now that after we have the DPR initialized, we need to call update_embeddings() to iterate over all
        # previously indexed documents and update their embedding representation. 
        # While this can be a time consuming operation (depending on corpus size), it only needs to be done once. 
        # At query time, we only need to embed the query and compare it the existing doc embeddings which is very fast.
        document_store.update_embeddings(retriever)
        # document_store.save("embeddings")
        return retriever

    def _format_prediction(self, haystack_prediction):
      formatted_response = {"predictions": []}
      for prediction in haystack_prediction["answers"]:
        formatted_response["predictions"].append(
            {
                "answer": prediction["answer"],
                "context": self.exhibitProcessedArray[int(prediction["meta"]["paragraph_id"])]["text"],
                "footnotes": self.exhibitProcessedArray[int(prediction["meta"]["paragraph_id"])]["footnotes"],
                "score": prediction["score"],
            }
        )
      return formatted_response

    def predict(self, question):
      predictions = self.pipe.run(question)
      formatted_response = self._format_prediction(predictions)
      return formatted_response
