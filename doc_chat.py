import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
from embedding import LocalEmbedding
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# 设置Streamlit应用的页面标题和布局
st.set_page_config(page_title="文档回答", layout="wide")
# 设置应用的标题
st.title("文档回答")

# 上传txt文件，允许上传多个文件
uploaded_files = st.sidebar.file_uploader(
    label="上传md文件", type=["md"], accept_multiple_files=True
)
# 如果没有上传文件，提示用户上传文件并停止运行
if not uploaded_files:
    st.info("请先上传md文档")
    st.stop()


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # 读取上传的文档，并写入一个临时目录
    docs = []
    temp_dir = tempfile.TemporaryDirectory(dir="./data")
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        # 使用TextLoader加载文本文件
        loader = TextLoader(temp_filepath, encoding="utf-8")
        docs.extend(loader.load())

    # 进行文档分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 使用向量模型生成文档的向量表示
    # embedding = SentenceTransformer("BAAI/bge-m3")
    # 指定本地路径（假设模型文件在 ./models/bge-m3 目录下）
    # model_path = "D:\\huggingface\\hub\\models--BAAI--bge-m3\\snapshots\\5617a9f61b028005a4858fdac845db406aefb181"
    # model_path = "./model/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
    # embedder = LocalEmbedding(model_path)
    # NVIDIAEmbeddings.get_available_models()
    embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
    vectordb = FAISS.from_documents(splits, embedder)

    # 创建文档检索器
    retriever = vectordb.as_retriever()

    return retriever


# 配置检索器
retriever = configure_retriever(uploaded_files)

# 如果session_state中没有消息记录或用户点击了清空聊天记录按钮，则初始化消息记录
if "messages" not in st.session_state or st.sidebar.button("清空聊天记录"):
    st.session_state["messages"] = [{"role": "assistant", "content": "您好，我是文档回答助手"}]

# 加载历史聊天记录
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 创建检索工具
from langchain.tools.retriever import create_retriever_tool

# 创建用于文档检索的工具
tool = create_retriever_tool(
    retriever,
    "文档检索",
    "用于检索用户提出的问题，并基于检索到的文档内容进行回复"
)
tools = [tool]

# 创建聊天消息历史记录
msgs = StreamlitChatMessageHistory()
# 创建对话缓存区内存
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)

# 指令模板
instructions = """您是一个设计用于查询文档来回答问题的代理。
您可以使用文档检索工具，并基于检索内容来回答问题
您可能不查询文档就知道答案，但是您仍然应该查询文档来获得答案。
如果您从文档中找不到任何信息用于回答问题，则根据您已有的知识来回复答案。
"""

# 基础提示模板
base_prompt_template = """
{instructions}

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

'''
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: {input}
Observation: the result of the action
'''

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

'''
Thought: Do I need to use a tool? No
Final Answer: [your response here]
'''

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""

# 创建基础提示模板
base_prompt = PromptTemplate.from_template(base_prompt_template)

# 创建部分填充的提示模板
prompt = base_prompt.partial(instructions=instructions)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# 创建llm
# ChatNVIDIA.get_available_models()
llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")

# 创建react Agent
agent = create_react_agent(llm, tools, prompt)

# 创建Agent执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,
                               handle_parsing_errors=True)

# 创建聊天输入框
user_query = st.chat_input(placeholder="请开始提问吧！")

# 如果有用户输入的查询
if user_query:
    # 添加用户消息到session_state
    st.session_state.messages.append({"role": "user", "content": user_query})
    # 显示用户消息
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        # 创建Streamlit回答处理器
        st_cb = StreamlitCallbackHandler(st.container())
        # agent 执行过程日志回调显示在Streamlit Container (如思考、选择工具、执行查询、观察结果等)
        config = {"callbacks": [st_cb]}
        # 执行Agent并获取响应
        response = agent_executor.invoke({"input": user_query}, config=config)
        # 添加助手消息到session_state
        st.session_state.messages.append({"role": "assistant", "content": response["output"]})
        # 显示助手响应
        st.write(response["output"])
