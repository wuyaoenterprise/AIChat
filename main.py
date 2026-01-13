import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from supabase import create_client, Client
import os

# ==========================================
# 1. 页面配置与初始化
# ==========================================
st.set_page_config(page_title="双核心 AI 聚合站 (Cloud)", page_icon="☁️", layout="wide")

st.title("🤖 双核心 AI 聚合终端 Pro")
st.markdown("### ChatGPT (OpenAI) | Gemini (Google) | ☁️ 云端同步版")
st.markdown("---")

# ==========================================
# 2. 安全与数据库连接
# ==========================================
try:
    OPENAI_KEY = st.secrets["keys"]["openai_api_key"]
    GOOGLE_KEY = st.secrets["keys"]["google_api_key"]
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
except Exception as e:
    st.error("❌ 缺少配置！请检查 .streamlit/secrets.toml 是否包含 [keys] 和 [supabase]。")
    st.stop()

# 初始化 Supabase 客户端
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase: Client = init_supabase()

# ==========================================
# 3. 用户身份识别 (关键逻辑)
# ==========================================
# Streamlit Cloud 会自动通过 Google Login 提供 user.email
user_email = None

if st.experimental_user.email:
    # 线上环境：直接获取登录用户的邮箱
    user_email = st.experimental_user.email
    st.sidebar.success(f"👤 已登录: {user_email}")
else:
    # 本地环境：提供一个模拟登录框方便你测试
    st.sidebar.warning("⚠️ 本地开发模式")
    user_email = st.sidebar.text_input("请输入测试邮箱 (模拟登录):", "test@example.com")

if not user_email:
    st.warning("👈 请先在侧边栏输入邮箱，或登录后开始对话。")
    st.stop()

# ==========================================
# 4. 历史记录管理 (Supabase)
# ==========================================

def load_history(email, model):
    """从数据库加载历史记录"""
    try:
        response = supabase.table("chat_history")\
            .select("*")\
            .eq("user_email", email)\
            .order("created_at", desc=False)\
            .execute()
        # 转换回 Streamlit 需要的格式
        messages = []
        for row in response.data:
            messages.append({"role": row["role"], "content": row["content"]})
        return messages
    except Exception as e:
        st.error(f"加载历史失败: {e}")
        return []

def save_message(email, model, role, content):
    """保存单条消息到数据库"""
    try:
        supabase.table("chat_history").insert({
            "user_email": email,
            "model_name": model,
            "role": role,
            "content": content
        }).execute()
    except Exception as e:
        st.error(f"保存失败: {e}")

def clear_history(email):
    """清空该用户的云端记录"""
    try:
        supabase.table("chat_history").delete().eq("user_email", email).execute()
        st.session_state["messages"] = []
        st.rerun()
    except Exception as e:
        st.error(f"删除失败: {e}")

# ==========================================
# 5. 模型与逻辑控制
# ==========================================
with st.sidebar:
    st.markdown("---")
    model_choice = st.radio(
        "选择 AI 模型:",
        ("ChatGPT-5", "Gemini 3 Pro"), # 界面显示的名字
        index=1
    )
    
    # 状态管理：如果还没加载过或者换了用户/模型，重新加载历史
    # 这里我们简化逻辑：所有模型共享一个历史，或者你可以选择过滤 'model_name'
    if "messages" not in st.session_state or st.sidebar.button("🔄 刷新/加载云端记录"):
        st.session_state["messages"] = load_history(user_email, "shared_history")
    
    if st.button("🗑️ 清空我的云端记录"):
        clear_history(user_email)

# ==========================================
# 6. AI 响应函数
# ==========================================

def get_chatgpt_response(messages):
    client = OpenAI(api_key=OPENAI_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-5", 
            messages=messages,
            stream=True 
        )
        return response
    except Exception as e:
        return f"ChatGPT Error: {str(e)}"

def get_gemini_response(messages):
    genai.configure(api_key=GOOGLE_KEY)
    model = genai.GenerativeModel('gemini-3-pro-preview') 
    
    gemini_history = []
    for msg in messages[:-1]: 
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})
    
    chat = model.start_chat(history=gemini_history)
    
    try:
        response = chat.send_message(messages[-1]["content"], stream=True)
        return response
    except Exception as e:
        return f"Gemini Error: {str(e)}"

# ==========================================
# 7. 聊天界面
# ==========================================

# 显示历史
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 处理输入
if prompt := st.chat_input("说点什么..."):
    # 1. 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})
    # 2. 存入云端 (用户)
    save_message(user_email, model_choice, "user", prompt)

    # 3. AI 回复
    with st.chat_message("assistant"):
        response_placeholder = st.empty() 
        full_response = ""
        
        if model_choice == "ChatGPT-5":
            stream = get_chatgpt_response(st.session_state["messages"])
        elif model_choice == "Gemini 3 Pro":
            stream = get_gemini_response(st.session_state["messages"])
            
        # 统一流处理
        if isinstance(stream, str): # 报错了
            response_placeholder.error(stream)
            full_response = stream
        else:
            try:
                for chunk in stream:
                    content = ""
                    if model_choice == "ChatGPT-5":
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                    else: # Gemini
                         content = chunk.text
                    
                    full_response += content
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            except Exception as e:
                response_placeholder.error(f"生成中断: {e}")
                full_response = str(e)

    # 4. 存入云端 (AI)
    st.session_state["messages"].append({"role": "assistant", "content": full_response})
    save_message(user_email, model_choice, "assistant", full_response)