# langchain==0.3.18
# langchain-core==0.3.35
# langchain-text-splitters==0.3.6
# langsmith==0.3.8
import os
from dotenv import load_dotenv

#openai api를 활용하여 채팅 
from openai import OpenAI

#모델과의 대화를 주도
from langchain_community.chat_models import ChatOpenAI
#대화 시 프롬프트를 미리 세팅, 전달할
from langchain.prompts import ChatPromptTemplate

#RAG관련된 라이브러리
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import retrieval_qa
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader

import tiktoken
#문서를 가져왔을 때, 단어를 나누는 작업(토큰화) -> 문자를 나누는 작업을 도와주는 역할
from langchain.text_splitter import RecursiveCharacterTextSplitter

def titoken_len(text):
  tokenizer = tiktoken.get_encoding('cl100k_base')
  tokens = tokenizer.encode(text)
  return len(tokens)

def default_chat(query):
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('API_KEY') 
    Chat = ChatOpenAI(model='gpt-4o-mini', temperature=0.5)

    template_string = '''
                        입력되는 내용 : {inputs}

                        사용자가 inputs란 내용으로 너에게 질문을 할거야.
                        너는 상냥하고 따뜻한, 긍정적인 어조로 사용자의 질문에 답변해줘.

                        혹시라도, 입력되는 inputs에
                        욕설이 있으면 제거하고 *로 대체해서 보여줘.

                        예시 : 나쁜 새끼... -> 나쁜 **
                        '''
    
    prompt_template = ChatPromptTemplate.from_template(template_string)

    #실제로 포맷된 메시지를 전달하기 위한 작업
    chat_message = prompt_template.format_messages(inputs=query)

    response = Chat(chat_message)
    return response.content