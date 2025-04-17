import streamlit as st
import os
import tempfile
import time
import uuid
import glob

# importações do langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

# variáveis:------
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Diretório de PDFs
PDF_DIR = "./Programa_Desenrola"

# Inicializar estado da sessão para histórico de chat e documentos carregados
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "loaded_pdfs" not in st.session_state:
    st.session_state.loaded_pdfs = {}  # Dicionário para rastrear PDFs carregados

# Estilo CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4F8BF9;
        margin-bottom: 2rem;
    }
    .card {
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .highlight {
        background-color: #e9f7fe;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4F8BF9;
        margin-bottom: 20px;
    }
    .timer {
        color: #6c757d;
        font-size: 0.9rem;
        font-style: italic;
    }
    .response-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1.5rem;
        color: #4F8BF9;
    }
    .btn-custom {
        background-color: #4F8BF9;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }
    .input-area {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 10px;
        border-radius: 8px;
        background-color: white;
        border: 1px solid #dee2e6;
        margin-bottom: 20px;
        position: relative;
        display: block;
        width: 100%;
    }
    .chat-wrapper {
        position: relative;
        margin-bottom: 30px;
    }
    .form-container {
        margin-top: 30px;
        position: relative;
        z-index: 10;
    }
    .message-container {
        overflow: hidden;
        width: 100%;
        margin-bottom: 10px;
        clear: both;
    }
    .user-message {
        background-color: #e9f7fe;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin: 5px 0;
        display: inline-block;
        float: right;
        clear: both;
        max-width: 80%;
    }
    .assistant-message {
        background-color: #f1f3f4;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin: 5px 0;
        display: inline-block;
        float: left;
        clear: both;
        max-width: 80%;
    }
</style>
""", unsafe_allow_html=True)

# Cabeçalho do aplicativo com estilo personalizado
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image("./gemini-native-image.png", width=100, use_container_width=True)
    st.markdown("<h1 class='main-header'>Programa Desenrola</h1>", unsafe_allow_html=True)

# Inicializar modelo de linguagem (LLM)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", google_api_key=GOOGLE_API_KEY)

prompt = ChatPromptTemplate.from_template(
    """
    Você é um Procurador Federal, que é um advogado representante de Autarquias e Fundações Públicas Federais, e está ajudando um devedor desses órgãos governamentais a negociar as dívidas.
    Você deve responder às perguntas do devedor com base no contexto fornecidos.
    Certifique-se de que sua resposta seja relevante para o contexto.
    Se não souber a resposta, diga "Não encontrei essa informação nos documentos. Poderia reformular sua pergunta de forma mais específica, com exemplos se possível?".
    
    <contexto>
    {context}
    </contexto>

    Histórico do chat:
    {chat_history}
    
    Pergunta atual:
    {input}
    """
)

def load_pdfs_from_directory():
    """Carrega todos os PDFs da pasta especificada"""
    # Verifica se o diretório existe
    if not os.path.exists(PDF_DIR):
        st.error(f"A pasta '{PDF_DIR}' não foi encontrada. Por favor, crie-a e adicione os documentos PDF.")
        return False
        
    # Lista todos os arquivos PDF no diretório
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    
    if not pdf_files:
        st.error(f"Nenhum arquivo PDF encontrado na pasta '{PDF_DIR}'.")
        return False
        
    st.info(f"Encontrados {len(pdf_files)} arquivos PDF para processar.")
    return pdf_files

def vector_embedding(force_reload=False, clear_pdfs=False):
    # Limpar PDFs se solicitado
    if clear_pdfs:
        if "vectors" in st.session_state:
            del st.session_state.vectors
        if "docs" in st.session_state:
            del st.session_state.docs
        if "final_documents" in st.session_state:
            del st.session_state.final_documents
        st.session_state.loaded_pdfs = {}  # Limpar lista de PDFs carregados
        st.session_state.chat_history = []  # Limpar histórico de chat
        return True
        
    # Processar PDFs da pasta local
    if "vectors" not in st.session_state or force_reload:
        with st.spinner('Processando PDFs do Programa Desenrola... Isso pode levar um momento.'):
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=GOOGLE_API_KEY
            )
            
            # Carregar PDFs da pasta local
            pdf_files = load_pdfs_from_directory()
            if not pdf_files:
                return False
                
            all_docs = []  # Lista para todos os documentos
            
            # Processar cada arquivo encontrado
            for pdf_path in pdf_files:
                try:
                    # Extrair nome do arquivo para exibição
                    pdf_name = os.path.basename(pdf_path)
                    
                    # Carregar e processar o PDF
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    
                    # Adicionar metadados para rastrear a origem do documento
                    for doc in docs:
                        doc.metadata["source"] = pdf_name
                    
                    if docs:
                        all_docs.extend(docs)
                        # Registrar o PDF carregado
                        unique_id = str(uuid.uuid4())
                        st.session_state.loaded_pdfs[unique_id] = pdf_name
                    else:
                        st.warning(f"Não foi possível extrair conteúdo do PDF: {pdf_name}")
                    
                except Exception as e:
                    st.error(f"Erro ao processar PDF {pdf_name}: {str(e)}")
            
            # Verificar se conseguimos extrair algum documento
            if not all_docs:
                st.error("Não foi possível extrair conteúdo de nenhum dos PDFs carregados.")
                return False
                
            # Dividir documentos em chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(all_docs)
            
            # Verificar se os documentos foram divididos com sucesso
            if not st.session_state.final_documents:
                st.error("Falha ao dividir o conteúdo dos documentos.")
                return False
                
            # Criar vetores
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents, st.session_state.embeddings
            )
            
            # Guardar todos os documentos carregados
            st.session_state.docs = all_docs
            
            # Resetar histórico de chat ao carregar novos documentos
            st.session_state.chat_history = []
            return True

# Barra lateral com configurações
with st.sidebar:
    st.markdown("<h2>Configurações do Documento</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Processar PDFs"):
            # Forçar recarregamento com os PDFs da pasta local
            if vector_embedding(force_reload=True):
                st.success("PDFs processados com sucesso!")
    
    with col2:
        if st.button("Limpar PDFs"):
            # Limpar todos os PDFs carregados
            vector_embedding(clear_pdfs=True)
            st.success("Todos os PDFs foram removidos.")
            st.rerun()
    
    # Exibir PDFs carregados
    if st.session_state.loaded_pdfs:
        st.markdown("### PDFs Carregados")
        for pdf_name in st.session_state.loaded_pdfs.values():
            st.info(f"📄 {pdf_name}")
    
    if st.button("Limpar Histórico de Conversa"):
        st.session_state.chat_history = []
        st.success("Histórico de conversa limpo!")

# Interface principal do chat
st.markdown("<h2>Converse e esclareça dúvidas com o chatbot sobre o Programa Desenrola</h2>", unsafe_allow_html=True)

# Inicializar chave para controlar fluxo de chat
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# Mensagem inicial ou container de chat
chat_placeholder = st.empty()  # Usa um placeholder em vez de condicional

# Exibir histórico de chat ou mensagem informativa
if st.session_state.chat_history:
    chat_content = "<div class='chat-container'>"
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            chat_content += f"<div class='message-container'><div class='user-message'>{message['content']}</div></div>"
        else:
            chat_content += f"<div class='message-container'><div class='assistant-message'>{message['content']}</div></div>"
    chat_content += "</div>"
    chat_placeholder.markdown(chat_content, unsafe_allow_html=True)
else:
    chat_placeholder.info("Carregue os PDFs do Programa Desenrola e faça perguntas para iniciar a conversa.")

# Abordagem de formulário para evitar reenvio automático
with st.form(key="chat_form"):
    col1, col2 = st.columns([4, 1])
    with col1:
        # Usar uma chave dinamicamente alterada para forçar a reinicialização do input
        user_question = st.text_input(
            "Digite sua pergunta sobre os documentos:", 
            key=f"user_question_{st.session_state.input_key}"
        )
    with col2:
        submit_button = st.form_submit_button("Enviar", use_container_width=True)

# Processar o envio - agora só acontece quando o formulário é enviado
if submit_button:
    if not user_question:
        st.warning("⚠️ Por favor, digite uma pergunta antes de enviar.")
    elif "vectors" not in st.session_state:
        st.warning("⚠️ Por favor, processe os PDFs primeiro usando o painel lateral.")
    else:
        # Adicionar mensagem do usuário ao histórico de chat
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Formatar histórico de chat para contexto
        formatted_history = ""
        for msg in st.session_state.chat_history[:-1]:  # Excluir pergunta atual
            prefix = "Usuário: " if msg["role"] == "user" else "Assistente: "
            formatted_history += f"{prefix}{msg['content']}\n\n"
        
        with st.spinner('Buscando a melhor resposta...'):
            # Criar cadeia de recuperação
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 5})
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Obter resposta
            start = time.process_time()
            response = retrieval_chain.invoke({
                "input": user_question,
                "chat_history": formatted_history
            })
            response_time = time.process_time() - start
            
            # Adicionar resposta ao histórico de chat
            st.session_state.chat_history.append({"role": "assistant", "content": response.get('answer', 'Não consegui encontrar uma resposta nos documentos carregados.')})
            
            # Armazenar tempo de resposta para informações de depuração
            st.session_state.last_response_time = response_time
            
        # Incrementar a chave de entrada para forçar um novo widget de entrada de texto no rerun
        st.session_state.input_key += 1
        
        # Forçar um rerun para mostrar o chat atualizado e limpar a entrada
        st.rerun()

# Exibir informações de depuração no expansor
with st.expander("ℹ️ Informações de Depuração", expanded=False):
    if "vectors" in st.session_state:
        st.write(f"Número de documentos carregados: {len(st.session_state.docs)}")
        st.write(f"Número de chunks após divisão: {len(st.session_state.final_documents)}")
        if "last_response_time" in st.session_state:
            st.write(f"Tempo da última resposta: {st.session_state.last_response_time:.2f} segundos")

# Adicionar um rodapé
st.markdown("---")
st.markdown("<center>Desenvolvido com Gemini + LangChain + Streamlit</center>", unsafe_allow_html=True)